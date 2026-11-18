"""Load the released complete model, or a fresh base with decoder-only LoRA."""
import types
from pathlib import Path
import torch
from transformers import AutoModel, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from .data import LETTERS

MODEL_SOURCE_SHA256 = '830855078d0c0b159a47041f8942949fd1a6efd39c62127c1387cc3b84058fa3'
RELEASE_TEMPERATURE = 1.0790077648040843


def processor_for(path):
    processor = Qwen2_5OmniProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    return processor


def choice_token_ids(processor):
    encoded = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != 26:
        raise ValueError('Expected 26 distinct, single-token option letters')
    return [ids[0] for ids in encoded]


def move_inputs(inputs, device):
    return {k: v.to(device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()}


def _bounded_mlp(self, hidden):
    if hidden.shape[-2] <= 16384:
        return self._uemo_original_forward(hidden)
    if self.training and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return torch.cat([checkpoint(self._uemo_original_forward, part, use_reentrant=False)
                          for part in hidden.split(2048, dim=-2)], dim=-2)
    return torch.cat([self._uemo_original_forward(part) for part in hidden.split(2048, dim=-2)], dim=-2)


def base_with_attention(path, device):
    from .uemo_integration import install
    model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': device}, attn_implementation='flash_attention_2', output_loading_info=True)
    if info['missing_keys'] or info['mismatched_keys'] or info.get('error_msgs'):
        raise RuntimeError(f'Incomplete base checkpoint: {info}')
    install(model, source_sha256=MODEL_SOURCE_SHA256)
    for layer in model.model.layers:
        layer.mlp._uemo_original_forward = layer.mlp.forward
        layer.mlp.forward = types.MethodType(_bounded_mlp, layer.mlp)
    model.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
    model.config.use_cache = False
    return model


def load_train_model(base_path, device):
    from peft import LoraConfig, get_peft_model
    model = base_with_attention(base_path, device)
    targets = [name for name, module in model.named_modules()
               if name.startswith('model.layers.') and isinstance(module, torch.nn.Linear)
               and name.rsplit('.', 1)[-1] in {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}]
    if len(targets) != 252:
        raise ValueError('Expected the 252 Qwen2.5-Omni-3B decoder LoRA targets')
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                          target_modules=targets, bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    model.get_base_model().audio_tower.eval()
    model.get_base_model().visual.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != 29933568 or any(p.dtype != torch.float32 for p in trainable):
        raise ValueError('Unexpected trainable parameters or precision')
    return model, processor_for(base_path)


def load_eval_model(model_path, device, adapter_path=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA and FlashAttention 2 are required')
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_with_attention(model_path, device), adapter_path,
                                          is_trainable=False, local_files_only=True)
        temperature = 1.0  # A newly trained adapter has not been temperature-calibrated.
    else:
        model, info = AutoModel.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map={'': device},
            attn_implementation='flash_attention_2', output_loading_info=True)
        if info['missing_keys'] or info['unexpected_keys'] or info['mismatched_keys'] or info.get('error_msgs'):
            raise RuntimeError(f'Incomplete UEmo-D1 checkpoint: {info}')
        if model.config.model_type != 'uemo_d1':
            raise ValueError('Use the complete wudq/UEmo-D1 release, or supply --adapter with the original base')
        temperature = float(model.config.uemo_temperature)
    model.eval()
    model.requires_grad_(False)
    return model, processor_for(model_path), temperature


def forward_logits(model, inputs):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    base.rope_deltas = None
    return model(**inputs, use_audio_in_video=True, use_cache=False,
                 mer_prefix_tokens=inputs['input_ids'].shape[-1]).logits[:, -1, :].float()


def choice_loss(logits, targets, token_ids, counts):
    """Per-example valid-choice CE; invalid options have exactly zero gradient."""
    ids = torch.as_tensor(token_ids, device=logits.device)
    counts = torch.as_tensor(counts, device=logits.device)
    targets = torch.as_tensor(targets, device=logits.device)
    if (logits.ndim != 2 or len(targets) != len(logits)
            or counts.shape != targets.shape or not bool(torch.isfinite(logits).all())
            or not bool(((counts >= 2) & (counts <= len(ids)) & (targets >= 0) & (targets < counts)).all())):
        raise ValueError('Invalid choice logits/targets/counts')
    z = logits if logits.dtype == torch.float64 else logits.float()
    scores = z.index_select(1, ids)
    valid = torch.arange(len(ids), device=z.device)[None, :] < counts[:, None]
    log_probs = scores.masked_fill(~valid, float('-inf')).log_softmax(-1)
    return -log_probs.gather(1, targets[:, None]).squeeze(1)
