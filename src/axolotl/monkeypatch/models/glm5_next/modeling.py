"""Sample packing for GLM-5.3-Flash (glm5_next).

Upstream only masks padding: the KDA short conv and recurrence run unbroken across
packed documents, and the DSA indexer pools and selects keys from earlier documents.
Both are gated on packed `position_ids`, so unpacked runs and cached decoding keep
the stock forward.
"""

import importlib
from functools import wraps

import torch
import torch.nn.functional as F
from transformers.integrations.accelerate import force_accelerate_hooks

from axolotl.monkeypatch.models.qwen4_exp.modeling import (
    CU_SEQLENS_KEY,
    POSITION_IDS_KEY,
    _make_text_model_forward,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MODULE_NAME = "transformers.models.glm5_next.modeling_glm5_next"

_ORIGINALS: dict[str, object] = {}


def _load_fla():
    """(causal_conv1d, chunk_kda), each None when flash-linear-attention is missing."""
    try:
        from fla.modules.convolution import causal_conv1d
    except ImportError:
        causal_conv1d = None
    try:
        from fla.ops.kda import chunk_kda
    except ImportError:
        chunk_kda = None
    return causal_conv1d, chunk_kda


def _make_linear_attention_forward(
    modeling, original, fla_causal_conv1d, fla_chunk_kda
):
    @wraps(original)
    @force_accelerate_hooks("conv1d")
    def patched_forward(
        self, hidden_states, cache_params=None, attention_mask=None, **kwargs
    ):
        kwargs.pop(POSITION_IDS_KEY, None)
        cu_seqlens = kwargs.pop(CU_SEQLENS_KEY, None)
        if cu_seqlens is None:
            return original(self, hidden_states, cache_params, attention_mask, **kwargs)
        if fla_causal_conv1d is None or fla_chunk_kda is None:
            # the torch fallbacks ignore cu_seqlens and would mix packed documents
            raise RuntimeError(
                "Packed sequences require flash-linear-attention (varlen causal_conv1d "
                "and chunk_kda). Install flash-linear-attention or disable sample_packing."
            )

        hidden_states = modeling.apply_mask_to_padding_states(
            hidden_states, attention_mask
        )
        batch_size, seq_len = hidden_states.shape[:2]
        # fla's varlen kernels treat the whole batch as one sequence
        flat_shape = (1, batch_size * seq_len, -1, self.head_dim)

        mixed_qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        )
        mixed_qkv, _ = fla_causal_conv1d(
            x=mixed_qkv.reshape(1, batch_size * seq_len, -1),
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
        )
        query, key, value = torch.split(mixed_qkv, [self.qkv_dim] * 3, dim=-1)

        core_attn_out, _ = fla_chunk_kda(
            query.reshape(flat_shape),
            key.reshape(flat_shape),
            value.reshape(flat_shape),
            g=self.forget_gate(hidden_states).reshape(flat_shape),
            beta=torch.sigmoid(self.b_proj(hidden_states)).reshape(
                1, batch_size * seq_len, -1
            ),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out.reshape(hidden_shape), gate)
        return self.o_proj(output.reshape(batch_size, seq_len, -1))

    return patched_forward


def _make_attention_forward(original):
    """Hand the packed positions to the indexer, which upstream calls without kwargs."""

    @wraps(original)
    def patched_forward(self, *args, **kwargs):
        kwargs.pop(CU_SEQLENS_KEY, None)
        position_ids = kwargs.pop(POSITION_IDS_KEY, None)
        if self.indexer is None or position_ids is None:
            return original(self, *args, **kwargs)
        self.indexer.packed_position_ids = position_ids
        try:
            return original(self, *args, **kwargs)
        finally:
            self.indexer.packed_position_ids = None

    return patched_forward


def _make_indexer_forward(original):
    @wraps(original)
    def patched_forward(self, hidden_states, q_resid, attention_mask, past_key_values):
        position_ids = getattr(self, "packed_position_ids", None)
        if position_ids is None:
            return original(
                self, hidden_states, q_resid, attention_mask, past_key_values
            )
        return packed_indexer_forward(
            self, hidden_states, q_resid, attention_mask, position_ids
        )

    return patched_forward


@torch.no_grad()
def packed_indexer_forward(self, hidden_states, q_resid, attention_mask, position_ids):
    """Upstream top-k selection with k-pools aligned to, and confined within, each document."""
    batch_size, seq_len = hidden_states.shape[:2]
    device = hidden_states.device
    kpool = self.index_kpool
    positions = torch.arange(seq_len, device=device)
    batch_idx = torch.arange(batch_size, device=device)[:, None, None]
    doc_start = positions - position_ids

    q = self.wq_b(q_resid).view(batch_size, seq_len, -1, self.head_dim)
    k = self.k_norm(self.wk(hidden_states))
    gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)

    is_pool_start = position_ids.remainder(kpool) == 0
    num_pools = int(is_pool_start.sum(-1).max())
    pool_start = torch.argsort((~is_pool_start).to(torch.int8), dim=-1, stable=True)
    pool_start = pool_start[:, :num_pools]
    pool_doc = doc_start.gather(-1, pool_start)

    pool_indices = pool_start[..., None] + torch.arange(kpool, device=device)
    safe_indices = pool_indices.clamp(max=seq_len - 1)
    member_valid = (
        (pool_indices < seq_len)
        & is_pool_start.gather(-1, pool_start)[..., None]
        & (doc_start[batch_idx, safe_indices] == pool_doc[..., None])
        & attention_mask[batch_idx, safe_indices]
    )
    pool_valid = member_valid.all(-1)

    logits = (
        gate_scores[batch_idx, safe_indices].float()
        + self.index_kpool_compress_ape.float()
    )
    logits = logits.masked_fill(~member_valid[..., None], float("-inf"))
    probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(k.dtype)
    pool_keys = (probabilities * k[batch_idx, safe_indices]).sum(dim=2)

    scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * self.softmax_scale)
    weights = self.weights_proj(
        hidden_states.to(self.weights_proj.weight.dtype)
    ).float()
    weights = weights * (self.n_heads**-0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    valid_candidates = (
        pool_valid[:, None]
        & (pool_indices[:, None, :, -1] <= positions[None, :, None])
        & (pool_doc[:, None] == doc_start[..., None])
    )
    index_scores = index_scores.masked_fill(
        ~valid_candidates, torch.finfo(index_scores.dtype).min
    )

    selected = index_scores.topk(
        min(self.index_topk // kpool, num_pools), dim=-1
    ).indices
    selected_valid = valid_candidates.gather(-1, selected)
    topk_indices = pool_indices[batch_idx, selected]
    topk_indices = topk_indices.masked_fill(~selected_valid[..., None], -1).flatten(-2)

    output_width = self.index_topk
    if self.index_kpool_always_select_tail and kpool > 1:
        tail_count = (position_ids + 1).remainder(kpool)
        tail_offsets = torch.arange(kpool - 1, device=device)
        tail = (positions + 1 - tail_count)[..., None] + tail_offsets
        tail = tail.masked_fill(tail_offsets >= tail_count[..., None], -1)
        topk_indices = torch.cat([topk_indices, tail], dim=-1)
        output_width += kpool - 1

    topk_indices = F.pad(
        topk_indices, (0, max(0, output_width - topk_indices.shape[-1])), value=-1
    )
    topk_indices = topk_indices[..., :output_width]
    topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
    return topk_indices.to(torch.int32)


def patch_glm5_next_modeling_packing():
    """Route packed-sequence boundaries into the KDA layers and the DSA indexer."""
    try:
        modeling = importlib.import_module(MODULE_NAME)
    except ImportError:
        LOG.warning("glm5_next not found in transformers, skipping patch")
        return None
    if _ORIGINALS:
        return None

    _ORIGINALS.update(
        text_model=modeling.Glm5NextTextModel.forward,
        linear_attn=modeling.Glm5NextTextLinearAttention.forward,
        attn=modeling.Glm5NextTextAttention.forward,
        indexer=modeling.Glm5NextTextIndexer.forward,
    )

    fla_causal_conv1d, fla_chunk_kda = _load_fla()
    modeling.Glm5NextTextModel.forward = _make_text_model_forward(
        _ORIGINALS["text_model"]
    )
    modeling.Glm5NextTextLinearAttention.forward = _make_linear_attention_forward(
        modeling, _ORIGINALS["linear_attn"], fla_causal_conv1d, fla_chunk_kda
    )
    modeling.Glm5NextTextAttention.forward = _make_attention_forward(_ORIGINALS["attn"])
    modeling.Glm5NextTextIndexer.forward = _make_indexer_forward(_ORIGINALS["indexer"])

    LOG.info(
        "Applied Glm5Next packing patch "
        f"(fla={'available' if fla_chunk_kda else 'unavailable'})"
    )

    def unpatch():
        if not _ORIGINALS:
            return
        modeling.Glm5NextTextModel.forward = _ORIGINALS["text_model"]
        modeling.Glm5NextTextLinearAttention.forward = _ORIGINALS["linear_attn"]
        modeling.Glm5NextTextAttention.forward = _ORIGINALS["attn"]
        modeling.Glm5NextTextIndexer.forward = _ORIGINALS["indexer"]
        _ORIGINALS.clear()

    return unpatch
