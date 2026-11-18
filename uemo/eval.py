"""Save every decision, raw option logits, calibrated probabilities and metrics."""
import argparse
import json
import math
import time
from pathlib import Path

from .data import LETTERS, load_records
from .io import model_fingerprint, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local complete UEmo-D1 directory, or base directory with --adapter')
    parser.add_argument('--adapter', help='Optional locally trained adapter directory')
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--media-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--resume', action='store_true')
    temperature = parser.add_mutually_exclusive_group()
    temperature.add_argument('--temperature', type=float)
    temperature.add_argument('--temperature-file', help='JSON with a fitted temperature')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max-samples must be positive')
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error('--temperature must be finite and positive')
    import torch
    from .data import prepare, batch_prepare
    from .model import load_eval_model, choice_token_ids, move_inputs, forward_logits
    from .metrics import report
    torch.set_num_threads(4)
    metadata = Path(args.metadata).resolve()
    root = Path(args.media_root).resolve() if args.media_root else metadata.parent
    rows = load_records(metadata)
    if args.max_samples:
        rows = rows[:args.max_samples]
    output = Path(args.output).resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory, or --resume with the same inputs')
    model, processor, default_temperature = load_eval_model(args.model, args.device, args.adapter)
    fitted = json.loads(Path(args.temperature_file).read_text()) if args.temperature_file else None
    value = float(fitted['temperature']) if fitted else (args.temperature if args.temperature is not None else default_temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Temperature must be finite and positive')
    contract = {'model': str(Path(args.model).resolve()), 'model_files': model_fingerprint(args.model),
                'adapter': str(Path(args.adapter).resolve()) if args.adapter else None,
                'adapter_files': model_fingerprint(args.adapter) if args.adapter else None,
                'metadata_sha256': sha256(metadata), 'media_root': str(root),
                'sample_count': len(rows), 'max_samples': args.max_samples, 'temperature': value,
                'temperature_file_sha256': sha256(args.temperature_file) if fitted else None}
    completed = []
    predictions = output/'predictions.jsonl'
    if args.resume:
        if json.loads((output/'run.json').read_text()) != contract:
            raise ValueError('Cannot resume with different inputs, weights, temperature or limits')
        if predictions.exists():
            with predictions.open(encoding='utf-8') as handle:
                for line in handle:
                    completed.append(json.loads(line))
        if len(completed) > len(rows):
            raise ValueError('Saved output has more records than this run')
        for saved, row in zip(completed, rows):
            for key in ('id', 'choices', 'gt_answer', 'source', 'source_id', 'modality'):
                if saved[key] != row[key]:
                    raise ValueError(f'Resume output differs from metadata: {key}')
            if saved.get('subtask') != row.get('subtask'):
                raise ValueError('Resume subtask differs from metadata')
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'run.json', contract)
    if fitted and fitted.get('model_files') and (fitted['model_files'] != contract['model_files']
            or fitted.get('adapter_files') != contract['adapter_files']):
        raise ValueError('Calibration belongs to a different model/adapter')
    ids = choice_token_ids(processor)
    with predictions.open('a', encoding='utf-8', buffering=1) as handle:
        for row in rows[len(completed):]:
            start = time.monotonic()
            try:
                inputs = move_inputs(batch_prepare(processor, [prepare(processor, row, root)]), args.device)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = forward_logits(model, inputs)[0, ids[:len(row['choices'])]]
                    probs = (logits/value).softmax(-1)
                if not bool(torch.isfinite(logits).all() and torch.isfinite(probs).all()):
                    raise RuntimeError('Non-finite option logits or probabilities')
                winner = int(logits.argmax())
                saved = {key: row[key] for key in ('id', 'source', 'source_id', 'modality', 'choices', 'gt_answer')}
                if 'subtask' in row:
                    saved['subtask'] = row['subtask']
                saved.update(prediction_letter=LETTERS[winner], prediction=row['choices'][winner],
                             correct=row['choices'][winner] == row['gt_answer'],
                             option_logits=logits.cpu().tolist(), probabilities=probs.cpu().tolist(),
                             temperature=value, seconds=time.monotonic()-start)
                handle.write(json.dumps(saved, ensure_ascii=False, allow_nan=False)+'\n')
                completed.append(saved)
                del inputs, logits, probs
                if len(completed) % 50 == 0 or len(completed) == len(rows):
                    print(json.dumps({'completed': len(completed), 'total': len(rows)}), flush=True)
            except Exception as error:
                with (output/'errors.jsonl').open('a', encoding='utf-8') as errors:
                    errors.write(json.dumps({'id': row['id'], 'type': type(error).__name__, 'error': str(error)}, ensure_ascii=False)+'\n')
                raise  # Never silently skip a sample or truncate its media.
    metrics = report(completed)
    metrics.update(temperature=value, sample_count=len(completed))
    write_json(output/'metrics.json', metrics)
    write_json(output/'completed.json', {'samples': len(completed), 'predictions_sha256': sha256(predictions),
                                         'metrics_sha256': sha256(output/'metrics.json')})
    print(json.dumps(metrics['mean_over_settings'], indent=2))


if __name__ == '__main__':
    main()
