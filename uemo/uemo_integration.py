"""UEmo-D1 inference attention; preserved from the validated final model."""
import inspect
import types
import weakref
import hashlib
from pathlib import Path
from collections.abc import Mapping

from .uemo_attention import factorized_attention


class ScopedAttentionRegistry(Mapping):
    def __init__(self, original, members, attention):
        self.original = original
        self.members = members
        self.attention = attention

    def __getitem__(self, key):
        native = self.original[key]
        if key != 'flash_attention_2':
            return native
        def dispatch(module, *args, **kwargs):
            if module not in self.members:
                return native(module, *args, **kwargs)
            if 'mer_prefix_tokens' not in kwargs:
                raise ValueError('Missing explicit input-prefix boundary')
            return factorized_attention(self.attention, module, *args, **kwargs)
        return dispatch

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)


def install(thinker, *, source_sha256, cpu_test_kernel=None):
    import torch
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as qwen
    from transformers.cache_utils import DynamicCache
    from transformers.integrations.flash_attention import flash_attention_forward

    source = Path(inspect.getsourcefile(qwen)).read_bytes()
    if hashlib.sha256(source).hexdigest() != source_sha256:
        raise ValueError('Installed model source differs from reviewed source')
    if hasattr(thinker, '_mer_input_bidir_handle'):
        raise ValueError('Already installed; no blind reinstallation')
    decoder = thinker if type(thinker) is qwen.Qwen2_5OmniThinkerTextModel else thinker.model
    if type(decoder) is not qwen.Qwen2_5OmniThinkerTextModel:
        raise ValueError('Expected the native Thinker text decoder')
    if isinstance(qwen.ALL_ATTENTION_FUNCTIONS, ScopedAttentionRegistry):
        raise ValueError('An isolated integration already owns this process registry')
    members = weakref.WeakSet()
    for layer in decoder.layers:
        attn = layer.self_attn
        if type(attn) is not qwen.Qwen2_5OmniAttention:
            raise ValueError('Unexpected decoder attention class')
        if attn.config._attn_implementation != 'flash_attention_2' or attn.sliding_window is not None or attn.attention_dropout != 0:
            raise ValueError('Requires original full FA2 zero-dropout attention')
        members.add(attn)
    if not members:
        raise ValueError('Empty decoder')
    original_registry = qwen.ALL_ATTENTION_FUNCTIONS
    if original_registry['flash_attention_2'] is not flash_attention_forward:
        raise ValueError('Unexpected native FA2 registry binding')
    if cpu_test_kernel is not None:
        if any(p.device.type != 'cpu' for p in thinker.parameters()):
            raise ValueError('CPU fixture kernel cannot be installed on a GPU model')
    attention = flash_attention_forward if cpu_test_kernel is None else cpu_test_kernel
    states = weakref.WeakKeyDictionary()
    signature = inspect.signature(decoder.forward)

    def arguments(args, kwargs):
        values = dict(signature.bind_partial(*args, **kwargs).arguments)
        values.update(values.pop('kwargs', {}))
        return values

    def validate(module, args, kwargs):
        x = arguments(args, kwargs)
        prefix = x.get('mer_prefix_tokens')
        if type(prefix) is not int or prefix < 1:
            raise ValueError('Explicit positive integer prefix boundary required')
        inp = x.get('inputs_embeds')
        if inp is None:
            inp = x.get('input_ids')
        if not isinstance(inp, torch.Tensor) or inp.ndim not in (2, 3):
            raise ValueError('Native inputs required')
        batch, count = inp.shape[:2]
        cache = x.get('past_key_values')
        use_cache = x.get('use_cache')
        if use_cache is None:
            use_cache = module.config.use_cache
        if cache is not None and not use_cache:
            raise ValueError('Cache supplied with caching disabled')
        if use_cache and module.training and module.gradient_checkpointing:
            raise ValueError('Cache with gradient checkpointing is unsupported')
        if use_cache and x.get('return_dict', module.config.use_return_dict) is False:
            raise ValueError('Cache requires the native structured return')
        past = 0
        state = None
        if cache is not None:
            if type(cache) is not DynamicCache or cache.is_compileable:
                raise ValueError('Only native DynamicCache is supported')
            past = cache.get_seq_length()
            state = states.get(cache)
            if past and state is None:
                raise ValueError('Unowned cache; bidirectional prefill provenance missing')
            if state is not None and (past, batch, prefix) != state[:3]:
                raise ValueError('Changed cache length, batch, or prefix boundary')
        total = past + count
        if min(batch, count) < 1 or prefix > total or (past and past < prefix):
            raise ValueError('Empty input or partial-prefix cache')
        cp = x.get('cache_position')
        if cp is not None:
            expected = torch.arange(past, total, device=inp.device)
            if not isinstance(cp, torch.Tensor) or cp.shape != (count,) or cp.dtype != torch.long or not torch.equal(cp, expected):
                raise ValueError('Cache positions must be the contiguous native suffix')
        pos = x.get('position_ids')
        if pos is not None and (pos.ndim != 3 or pos.shape != (3,batch,count)):
            raise ValueError('Only native three-axis multimodal positions are supported')
        mask = x.get('attention_mask')
        if mask is None:
            valid = torch.ones(batch,total,dtype=torch.bool,device=inp.device)
        else:
            if not isinstance(mask,torch.Tensor) or mask.shape != (batch,total) or mask.device != inp.device:
                raise ValueError('Native two-dimensional padding mask required')
            if not bool(((mask==0)|(mask==1)).all()):
                raise ValueError('Binary padding mask required')
            valid = mask.bool()
            if bool((valid[:,:-1]&~valid[:,1:]).any()) or not bool(valid[:,prefix-1].all()):
                raise ValueError('Native left padding and nonempty prefix required')
        if state is not None and not torch.equal(valid[:,:past],state[3]):
            raise ValueError('Cached padding/sample layout changed')
        if x.get('output_attentions',False):
            raise ValueError('Attention weights unsupported by original FA2')

    def record(module,args,kwargs,output):
        cache = getattr(output,'past_key_values',None)
        if cache is None:
            return
        x=arguments(args,kwargs)
        batch = (x.get('inputs_embeds') if x.get('inputs_embeds') is not None else x['input_ids']).shape[0]
        length = cache.get_seq_length()
        if any(cache.get_seq_length(i) != length for i in range(len(module.layers))):
            raise ValueError('Unequal per-layer native cache lengths')
        mask=x.get('attention_mask')
        device=next(module.parameters()).device
        valid=torch.ones(batch,length,dtype=torch.bool,device=device) if mask is None else mask.bool().detach().clone()
        states[cache]=(length,batch,x['mer_prefix_tokens'],valid)

    # Hooks read explicit per-call metadata. No global current-mode/current-prefix state.
    pre = decoder.register_forward_pre_hook(validate,with_kwargs=True)
    post = decoder.register_forward_hook(record,with_kwargs=True)
    original_prepare = getattr(thinker,'prepare_inputs_for_generation',None)
    if original_prepare is not None:
        def prepare(self,input_ids,mer_prefix_tokens=None,**kwargs):
            if type(mer_prefix_tokens) is not int or mer_prefix_tokens < 1:
                raise ValueError('Generation requires its explicit fixed prefix boundary')
            values=original_prepare(input_ids,**kwargs)
            values['mer_prefix_tokens']=mer_prefix_tokens
            return values
        thinker.prepare_inputs_for_generation=types.MethodType(prepare,thinker)
    registry=ScopedAttentionRegistry(original_registry,members,attention)
    qwen.ALL_ATTENTION_FUNCTIONS=registry
    handle=dict(decoder=decoder,pre_hook=pre,post_hook=post,original_registry=original_registry,
                original_prepare=original_prepare,cache_states=states,cpu_test_kernel=cpu_test_kernel is not None)
    thinker._mer_input_bidir_handle=handle
    return handle


def uninstall_for_test(thinker):
    """Fixture cleanup only; production process must use one installation."""
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as qwen
    h=thinker._mer_input_bidir_handle
    if not h['cpu_test_kernel']:
        raise ValueError('Production integration is not hot-swappable')
    h['pre_hook'].remove();h['post_hook'].remove()
    qwen.ALL_ATTENTION_FUNCTIONS=h['original_registry']
    if h['original_prepare'] is not None:
        thinker.prepare_inputs_for_generation=h['original_prepare']
    del thinker._mer_input_bidir_handle
