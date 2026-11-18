"""UEmo-D1 inference attention; preserved from the validated final model."""
from .uemo_mask import prefix_attention_plan


def factorized_attention(
    original_attention,
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    mer_prefix_tokens,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    **kwargs,
):
    import torch

    if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in (query, key, value)):
        raise ValueError('Expected native B,H,S,D query/key/value tensors')
    batch, query_heads, query_length, head_dim = query.shape
    if min(batch, query_heads, query_length, head_dim) < 1:
        raise ValueError('Empty query dimensions are not supported')
    if key.shape != value.shape or key.shape[0] != batch or key.shape[3] != head_dim:
        raise ValueError('Inconsistent native QKV geometry')
    if key.shape[1] < 1 or query_heads % key.shape[1]:
        raise ValueError('Invalid native grouped-query head counts')
    if not (query.dtype == key.dtype == value.dtype and query.device == key.device == value.device):
        raise ValueError('Mixed QKV device or dtype')
    if dropout != 0.0 or sliding_window is not None:
        raise ValueError('This experiment requires original zero-dropout full attention')
    if kwargs.get('position_ids') is not None:
        raise ValueError('Packed text positions are outside the native un-packed recipe')
    for name in ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k'):
        if kwargs.get(name) is not None:
            raise ValueError('Packed attention metadata is not admitted')
    if kwargs.get('output_attentions', False) or kwargs.get('head_mask') is not None:
        raise ValueError('Unsupported native FA2 attention inspection request')
    if kwargs.pop('is_causal', None) is not None:
        raise ValueError('Conflicting causality override')
    key_length = key.shape[2]
    plan = prefix_attention_plan(query_length, key_length, mer_prefix_tokens)
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (batch, key_length):
            raise ValueError('Only the original two-dimensional padding mask is supported')
        if attention_mask.device != query.device:
            raise ValueError('Padding mask must share the QKV device')
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('Padding mask must be binary')
        valid = attention_mask.bool()
        if bool((valid[:, :-1] & ~valid[:, 1:]).any()):
            raise ValueError('Only contiguous native left padding is supported')
        if not bool(valid[:, mer_prefix_tokens - 1].all()):
            raise ValueError('Each sample must contain a nonempty valid prefix')
    outputs = []
    for block in plan:
        block_mask = None if attention_mask is None else attention_mask[:, :block.key_end]
        output, weights = original_attention(
            module,
            query[:, :, block.query_begin:block.query_end, :],
            key[:, :, :block.key_end, :],
            value[:, :, :block.key_end, :],
            block_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=None,
            is_causal=block.causal,
            **kwargs,
        )
        expected = (batch, block.query_end - block.query_begin, query_heads, head_dim)
        if output.shape != expected or weights is not None:
            raise ValueError('Native kernel output contract changed')
        outputs.append(output)
    return (outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)), None
