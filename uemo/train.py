"""Fresh-base LoRA training with explicit global-batch gradient SUM reduction."""
import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

from .io import sha256, write_json


def packed_batches(processor, rows, root, micro_batch, token_budget):
    import torch
    from .data import prepare, batch_prepare
    prepared = []
    for row in rows:
        item = prepare(processor, row, root)
        single = batch_prepare(processor, [item])
        ids = single['input_ids'][0][single['attention_mask'][0].bool()].clone()
        prepared.append((row, item, ids))
    prepared.sort(key=lambda item: len(item[2]))

    def pack(items):
        inputs = batch_prepare(processor, [p for _, p, _ in items])
        same = all(torch.equal(inputs['input_ids'][i][inputs['attention_mask'][i].bool()], ids)
                   for i, (_, _, ids) in enumerate(items))
        if not same and len(items) > 1:
            middle = len(items)//2
            yield from pack(items[:middle])
            yield from pack(items[middle:])
            return
        if not same:
            raise RuntimeError('Singleton preprocessing changed input tokens')
        yield inputs, [r['choices'].index(r['gt_answer']) for r, _, _ in items], [len(r['choices']) for r, _, _ in items]

    pending = []
    for item in prepared:
        length = len(item[2])
        if pending and (len(pending) >= micro_batch or length*(len(pending)+1) > token_budget):
            yield from pack(pending)
            pending = []
        pending.append(item)
        if length > token_budget or len(pending) >= micro_batch:
            yield from pack(pending)
            pending = []
    if pending:
        yield from pack(pending)


def learning_rate(step, total, peak, warmup):
    return peak*min(1., (step+1)/max(1, warmup))*.5*(1+math.cos(math.pi*max(0, step-warmup)/max(1, total-warmup)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', required=True)
    p.add_argument('--media-root')
    p.add_argument('--base-model', required=True, help='Local original Qwen2.5-Omni-3B directory')
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/train.json')
    p.add_argument('--resume', help='Complete checkpoint-step-* directory from this output run')
    p.add_argument('--max-steps', type=int, help='Stop after this many total optimizer steps; keep the full schedule')
    p.add_argument('--max-samples', type=int, help='Use a deterministic subset for a smoke test')
    args = p.parse_args()
    if args.max_steps is not None and args.max_steps < 1 or args.max_samples is not None and args.max_samples < 1:
        p.error('Limits must be positive')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC', '1')
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from .data import load_records
    from .model import load_train_model, choice_token_ids, move_inputs, forward_logits, choice_loss
    from .io import model_fingerprint
    rank, world, local_rank = (int(os.environ.get(k, d)) for k, d in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=device)
    def barrier():
        if world > 1:
            dist.barrier()
    config = json.loads(Path(args.config).read_text())
    if (config['global_batch_size'] < world or config['epochs'] < 1
            or not 1 <= config['micro_batch_size'] <= 8 or config['padded_token_budget'] < 1
            or config['save_steps'] < 1 or config['learning_rate'] <= 0):
        raise ValueError('Invalid training configuration')
    metadata = Path(args.metadata).resolve()
    root = Path(args.media_root).resolve() if args.media_root else metadata.parent
    rows = load_records(metadata)
    order = np.random.default_rng(config['seed']).permutation(len(rows)).tolist()
    if args.max_samples:
        order = order[:args.max_samples]
    order = order * config['epochs']
    total = math.ceil(len(order)/config['global_batch_size'])
    output = Path(args.output).resolve()
    contract = {'config': config, 'metadata_sha256': sha256(metadata), 'samples': len(order),
                'world_size': world, 'max_samples': args.max_samples, 'media_root': str(root),
                'base_model': str(Path(args.base_model).resolve()), 'total_steps': total}
    if rank == 0:
        contract['base_files'] = model_fingerprint(args.base_model)
        if args.resume:
            if json.loads((output/'run.json').read_text()) != contract:
                raise ValueError('Resume configuration, data, base model, or world size changed')
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('Use a new output directory or explicitly --resume')
            output.mkdir(parents=True, exist_ok=True)
            write_json(output/'run.json', contract)
    barrier()
    contract_sha = sha256(output/'run.json')
    random.seed(config['seed']); np.random.seed(config['seed'])
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    model, processor = load_train_model(args.base_model, str(device))
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    token_ids = choice_token_ids(processor)
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=config['weight_decay'],
                                  betas=(.9, .999), eps=1e-8, fused=True)
    step = cursor = 0
    if args.resume:
        cp = Path(args.resume).resolve()
        if cp.parent != output:
            raise ValueError('Resume checkpoint must belong to the same output directory')
        checks = json.loads((cp/'complete.json').read_text())
        for rel, digest in checks['files'].items():
            if sha256(cp/rel) != digest:
                raise ValueError(f'Checkpoint hash mismatch: {rel}')
        state = torch.load(cp/'training_state.pt', map_location='cpu', weights_only=True)
        if state['contract_sha256'] != contract_sha or state['parameter_names'] != names:
            raise ValueError('Resume state belongs to a different run')
        set_peft_model_state_dict(model, load_file(str(cp/'adapter/adapter_model.safetensors')))
        optimizer.load_state_dict(state['optimizer'])
        step, cursor = state['step'], state['cursor']
        torch.set_rng_state(state['rng'][rank]['cpu'])
        torch.cuda.set_rng_state(state['rng'][rank]['cuda'], device)
        random.setstate(state['rng'][rank]['python'])

    def save_checkpoint():
        rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device), 'python': random.getstate()}
        states = [None]*world
        if world > 1:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        cp = output/f'checkpoint-step-{step:07d}'
        if rank == 0:
            temp = cp.with_name(cp.name+'.partial')
            temp.mkdir(exist_ok=False)
            model.save_pretrained(temp/'adapter', safe_serialization=True)
            torch.save({'step': step, 'cursor': cursor, 'optimizer': optimizer.state_dict(),
                        'rng': states, 'parameter_names': names, 'contract_sha256': contract_sha}, temp/'training_state.pt')
            write_json(temp/'complete.json', {'step': step, 'cursor': cursor, 'files': {
                str(f.relative_to(temp)): sha256(f) for f in temp.rglob('*') if f.is_file()}})
            temp.rename(cp)
            write_json(output/'latest_checkpoint.json', {'path': str(cp), 'step': step, 'cursor': cursor})
        barrier()

    try:
        while cursor < len(order) and (args.max_steps is None or step < args.max_steps):
            start = time.monotonic()
            indices = order[cursor:cursor+config['global_batch_size']]
            local_rows = [rows[i] for i in indices[rank::world]]
            optimizer.zero_grad(set_to_none=True)
            loss_sum, local_count = 0.0, 0
            for inputs, targets, counts in packed_batches(processor, local_rows, root, config['micro_batch_size'], config['padded_token_budget']):
                inputs = move_inputs(inputs, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = choice_loss(forward_logits(model, inputs), targets, token_ids, counts)
                    loss = losses.sum()/len(indices)
                loss.backward()
                loss_sum += float(loss.detach())
                local_count += len(targets)
                del inputs, loss, losses
            if local_count != len(local_rows):
                raise RuntimeError('Training batch dropped or duplicated a sample')
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
            if world > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            if not bool(torch.isfinite(flat).all()):
                raise RuntimeError('Non-finite gradients')
            pos = 0
            for parameter in params:
                parameter.grad = flat[pos:pos+parameter.numel()].view_as(parameter)
                pos += parameter.numel()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            lr = learning_rate(step, total, config['learning_rate'], config['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            loss_value = torch.tensor(loss_sum, device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            cursor += len(indices); step += 1
            if rank == 0:
                event = {'step': step, 'samples_seen': cursor, 'total_samples': len(order), 'loss': float(loss_value),
                         'gradient_norm': float(norm), 'learning_rate': lr, 'seconds': time.monotonic()-start}
                with (output/'train_log.jsonl').open('a') as log:
                    log.write(json.dumps(event, allow_nan=False)+'\n')
                print(json.dumps(event), flush=True)
            if step % config['save_steps'] == 0 or cursor == len(order) or step == args.max_steps:
                save_checkpoint()
        if rank == 0 and cursor == len(order):
            write_json(output/'completed.json', {'steps': step, 'samples_seen': cursor, 'checkpoint': str(output/f'checkpoint-step-{step:07d}'), 'temperature': 1.0})
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
