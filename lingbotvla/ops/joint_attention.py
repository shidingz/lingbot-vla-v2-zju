"""Opt-in joint SDPA attention with legacy empty-row behavior."""

import torch
import torch.nn.functional as F


def our_sdpa_attention_forward(query_states, key_states, value_states, attention_mask):
    """SDPA with native GQA and the existing eager mask/empty-row semantics.

    Q/K/V: [B, L, H, D], bool mask: [B, Lq, Lk], True means visible.
    The caller keeps its FP32 Q/K/V + RoPE for the first controlled experiment.
    No device scalar is read back by Python.
    """
    if attention_mask.dtype != torch.bool or attention_mask.ndim != 3:
        raise ValueError("Joint attention requires bool [B, Lq, Lk] allowed mask")
    batch, length, heads, head_dim = query_states.shape
    kv_heads = key_states.shape[2]
    if heads % kv_heads or value_states.shape[2] != kv_heads:
        raise ValueError("Query heads must be a multiple of matching K/V heads")

    # Old eager replaces every forbidden score with the same finite constant.
    # An entirely forbidden query therefore receives the mean of V, with zero
    # Q/K gradients. First make those SDPA rows safe, then restore that result.
    has_visible_key = attention_mask.any(dim=-1, keepdim=True)
    safe_mask = (attention_mask | ~has_visible_key).unsqueeze(1)
    result = F.scaled_dot_product_attention(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        attn_mask=safe_mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=heads != kv_heads,
    ).transpose(1, 2)
    empty_row_result = value_states.mean(dim=1, keepdim=True)
    empty_row_result = empty_row_result.repeat_interleave(heads // kv_heads, dim=2)
    result = torch.where(has_visible_key.unsqueeze(-1), result, empty_row_result)
    return result.reshape(batch, length, heads * head_dim)
