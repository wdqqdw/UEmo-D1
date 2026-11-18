"""UEmo-D1 inference attention; preserved from the validated final model."""
from typing import NamedTuple


class AttentionBlock(NamedTuple):
    query_begin: int
    query_end: int
    key_end: int
    causal: bool


def prefix_attention_plan(query_tokens, key_tokens, prefix_tokens):
    """Physical positions include left padding; native padding masks remain required.

    Queries must be the contiguous suffix of keys. The first call supplies the
    entire prefix. Later cached calls may contain only answer queries.
    """
    if any(type(x) is not int for x in (query_tokens, key_tokens, prefix_tokens)):
        raise ValueError('Token counts must be integers, excluding booleans')
    if not 1 <= query_tokens <= key_tokens or not 1 <= prefix_tokens <= key_tokens:
        raise ValueError('Invalid lengths or incomplete prefix')
    previous_tokens = key_tokens - query_tokens
    if 0 < previous_tokens < prefix_tokens:
        raise ValueError('Partial-prefix caching cannot preserve bidirectional states')
    if previous_tokens:
        return (AttentionBlock(0, query_tokens, key_tokens, True),)
    blocks = [AttentionBlock(0, prefix_tokens, prefix_tokens, False)]
    if prefix_tokens < query_tokens:
        blocks.append(AttentionBlock(prefix_tokens, query_tokens, key_tokens, True))
    return tuple(blocks)
