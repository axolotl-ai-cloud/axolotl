"""Monkeypatches for integrating MuonClip with HuggingFace attention modules."""

from __future__ import annotations

from functools import wraps

import torch

LLAMA_PATCH_FLAG = "_muonclip_llama_patched"
QWEN_PATCH_FLAG = "_muonclip_qwen_patched"
MAX_LOGIT_CHUNK = 512


def _repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    Expand grouped key/value heads to match the number of query heads.
    """

    if num_key_value_groups == 1 or hidden_states.size(1) == 0:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        num_key_value_groups,
        seq_len,
        head_dim,
    )
    return expanded.reshape(batch, num_kv_heads * num_key_value_groups, seq_len, head_dim)


def _headwise_max_logits(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scaling: float,
    *,
    chunk_size: int = MAX_LOGIT_CHUNK,
) -> torch.Tensor:
    """
    Compute per-head max logits without materializing the full attention matrix.
    """

    batch, num_heads, seq_len, _ = query_states.shape
    key_transposed = key_states.transpose(-2, -1)
    chunk = max(1, min(int(chunk_size), seq_len))
    max_vals = torch.full(
        (batch, num_heads),
        float("-inf"),
        dtype=torch.float32,
        device=query_states.device,
    )
    for start in range(0, seq_len, chunk):
        q_chunk = query_states[:, :, start : start + chunk, :]
        logits = torch.matmul(q_chunk, key_transposed) * scaling
        chunk_max = logits.amax(dim=(2, 3)).to(torch.float32)
        max_vals = torch.maximum(max_vals, chunk_max)
    return max_vals


def ensure_llama_attention_instrumentation():
    """Patch `LlamaAttention.forward` to record attention logits for MuonClip."""

    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            apply_rotary_pos_emb,
        )
    except ImportError:  # pragma: no cover - not available in minimal envs
        return

    if getattr(LlamaAttention, LLAMA_PATCH_FLAG, False):
        return

    original_forward = LlamaAttention.forward

    @wraps(original_forward)
    def wrapped(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        tracker = getattr(self, "_muonclip_tracker", None)
        record_logits = (
            tracker is not None
            and getattr(tracker, "active", False)
            and position_embeddings is not None
            and past_key_values is None
        )
        logits = None
        if record_logits:
            logits = _compute_llama_logits(self, hidden_states, position_embeddings, apply_rotary_pos_emb)
        outputs = original_forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if record_logits and logits is not None:
            from axolotl.muonclip.attention import record_attention_logits

            record_attention_logits(self, logits)
        return outputs

    LlamaAttention.forward = wrapped
    setattr(LlamaAttention, LLAMA_PATCH_FLAG, True)


@torch.no_grad()
def _compute_llama_logits(module, hidden_states, position_embeddings, rotary_fn):
    head_dim = module.head_dim
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = rotary_fn(query_states, key_states, cos, sin)
    num_kv_groups = getattr(module, "num_key_value_groups", 1)
    if key_states.size(1) != query_states.size(1):
        key_states = _repeat_kv(key_states, num_kv_groups)
    return _headwise_max_logits(query_states, key_states, module.scaling)


def ensure_qwen_attention_instrumentation():
    """Patch Qwen3 attention layers to record logits for MuonClip."""

    try:  # pragma: no cover
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention,
            apply_rotary_pos_emb as qwen_apply_rotary,
        )
    except ImportError:  # pragma: no cover
        return

    if getattr(Qwen3Attention, QWEN_PATCH_FLAG, False):
        return

    original_forward = Qwen3Attention.forward

    @wraps(original_forward)
    def wrapped(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        tracker = getattr(self, "_muonclip_tracker", None)
        record_logits = (
            tracker is not None
            and getattr(tracker, "active", False)
            and position_embeddings is not None
            and past_key_values is None
        )
        logits = None
        if record_logits:
            logits = _compute_qwen_logits(self, hidden_states, position_embeddings, qwen_apply_rotary)
        outputs = original_forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if record_logits and logits is not None:
            from axolotl.muonclip.attention import record_attention_logits

            record_attention_logits(self, logits)
        return outputs

    Qwen3Attention.forward = wrapped
    setattr(Qwen3Attention, QWEN_PATCH_FLAG, True)


@torch.no_grad()
def _compute_qwen_logits(module, hidden_states, position_embeddings, rotary_fn):
    head_dim = module.head_dim
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = rotary_fn(query_states, key_states, cos, sin)
    num_kv_groups = getattr(module, "num_key_value_groups", 1)
    if key_states.size(1) != query_states.size(1):
        key_states = _repeat_kv(key_states, num_kv_groups)
    return _headwise_max_logits(query_states, key_states, module.scaling)
