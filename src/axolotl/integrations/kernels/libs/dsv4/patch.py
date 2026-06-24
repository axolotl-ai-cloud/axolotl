"""Monkeypatch DeepSeek-V4 to use the fused Triton training kernels.

V4 ships eager-only attention (head_dim 512 > FlashAttention's cap, plus a per-head sink
and an in-block compressor KV-concat). This registers a custom attention backend that
routes sliding-attention layers to ``sliding_attn`` and CSA/HCA layers to
``csa_attn`` (splitting the eager-combined ``[sliding | compressed]`` KV and
``[sliding_mask | block_bias]`` mask back apart), and swaps the interleaved partial-RoPE
and the mHC HyperConnection for their fused kernels. o_a_proj stays on cuBLAS (a fused
grouped GEMM doesn't beat it).

Kernels assume ``attention_dropout == 0`` (the V4 default).
"""

import torch

from axolotl.utils.logging import get_logger

from .attention import sliding_attn
from .attention_csa import csa_attn
from .gated_pool import gated_softmax_pool
from .indexer import indexer_scores
from .mhc import hyperconnection_forward
from .rope import apply_rotary_pos_emb_triton

LOG = get_logger(__name__)

_MOD = None  # the transformers deepseek_v4 module; set by patch_deepseek_v4_kernels


def _dsv4_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling=None,
    dropout=0.0,
    sliding_window=None,
    s_aux=None,
    **kwargs,
):
    """Drop-in for ``eager_attention_forward``. ``query`` [B,H,S,D]; ``key``/``value``
    [B,1,KV,D] (shared MQA); ``s_aux`` = per-head sinks; ``attention_mask`` is the
    eager-built [sliding | block_bias] mask. Returns (attn_output [B,S,H,D], None).

    Patched onto the module's ``eager_attention_forward`` (V4's default backend) so the
    mask — and thus the compressor's ``block_bias`` — is still built and handed to us."""
    B, H, S, D = query.shape
    KV = key.shape[2]
    if scaling is None:
        scaling = D**-0.5
    window = sliding_window if sliding_window is not None else module.sliding_window

    if KV == S:  # sliding-attention layer (no compressor entries)
        out = sliding_attn(query, key, value, s_aux, scaling, window)
    else:  # CSA / HCA: keys are [sliding (S) | compressed (KV-S)], mask carries block_bias
        kv_slide = key[:, :, :S]
        kv_comp = key[:, :, S:]
        block_bias = attention_mask[..., S:]
        out = csa_attn(query, kv_slide, kv_comp, block_bias, s_aux, scaling, window)
    return out.transpose(1, 2).contiguous(), None


def _gated_pool_norm(new_kv, new_gate, kv_norm):
    """Fused replacement for ``kv_norm((new_kv * new_gate.softmax(2,fp32)).sum(2))`` —
    the compressor's per-window gated-softmax pool. Returns the compressor dtype (the
    pool is fp32 internally; we cast back so the downstream RoPE/concat stay bf16)."""
    return kv_norm(gated_softmax_pool(new_kv, new_gate).to(new_kv.dtype))


def _indexer_scorer_forward(self, q, compressed_kv, hidden_states):
    """Fused DeepseekV4IndexerScorer: never materializes the [B,S,H,T] fp32 score
    tensor (H=64 index heads). Output feeds only topk().indices, so no grad needed."""
    weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B,S,H]
    return indexer_scores(q, compressed_kv, weights, self.softmax_scale)


def _hca_compressor_forward(
    self, hidden_states, q_residual, position_ids, past_key_values, layer_idx
):
    mod = _MOD
    batch, _, _ = hidden_states.shape
    cache_layer = (
        past_key_values.layers[layer_idx] if past_key_values is not None else None
    )
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = (
            kv[:, :usable],
            gate[:, :usable],
            0,
        )
    else:
        chunk_kv, chunk_gate, first_window_position = (
            cache_layer.store_compression_weights("compressor", kv, gate)
        )

    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
        chunk_gate = (
            chunk_gate.view(batch, n_windows, self.compress_rate, -1)
            + self.position_bias
        )
        compressed = _gated_pool_norm(chunk_kv, chunk_gate, self.kv_norm)
        positions = torch.arange(n_windows, device=compressed.device)
        positions = (
            (positions * self.compress_rate + first_window_position)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        cos, sin = self.rotary_emb(
            compressed, position_ids=positions, layer_type=self.rope_layer_type
        )
        compressed = mod.apply_rotary_pos_emb(
            compressed.unsqueeze(1), cos, sin
        ).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
    compressed_kv = compressed.unsqueeze(1)

    compressed_len = compressed_kv.shape[2]
    seq_len = position_ids.shape[1]
    if seq_len == 1 or compressed_len == 0:
        return compressed_kv, None

    entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
    causal_threshold = (position_ids + 1) // self.compress_rate
    block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
    block_bias = block_bias.masked_fill(
        entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
        float("-inf"),
    )
    return compressed_kv, block_bias


def _indexer_forward(
    self, hidden_states, q_residual, position_ids, past_key_values, layer_idx
):
    mod = _MOD
    batch, seq_len, _ = hidden_states.shape
    cache_layer = (
        past_key_values.layers[layer_idx] if past_key_values is not None else None
    )
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = (
            kv[:, :usable],
            gate[:, :usable],
            0,
        )
    else:
        chunk_kv, chunk_gate, first_window_position = (
            cache_layer.store_compression_weights("indexer", kv, gate)
        )

    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias

        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full(
            (batch, n_windows, 2 * ratio, self.head_dim), float("-inf")
        )
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        if cache_layer is not None:
            prior_kv, prior_gate = cache_layer.update_overlap_state(
                "indexer", chunk_kv, chunk_gate, self.head_dim
            )
            if prior_kv is not None:
                new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

        compressed = _gated_pool_norm(new_kv, new_gate, self.kv_norm)
        positions = torch.arange(n_windows, device=compressed.device)
        positions = positions * self.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(
            compressed, position_ids=positions, layer_type=self.rope_layer_type
        )
        compressed = mod.apply_rotary_pos_emb(
            compressed.unsqueeze(1), cos, sin
        ).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    compressed_kv = (
        compressed
        if cache_layer is None
        else cache_layer.update_compressor_states("indexer", compressed)
    )

    cos_q, sin_q = self.rotary_emb(
        hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type
    )
    q = (
        self.q_b_proj(q_residual)
        .view(batch, seq_len, -1, self.head_dim)
        .transpose(1, 2)
    )
    q = mod.apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

    index_scores = self.scorer(q, compressed_kv, hidden_states)
    compressed_len = compressed_kv.shape[1]
    top_k = min(self.index_topk, compressed_len)

    if compressed_len > 0:
        causal_threshold = (position_ids + 1) // self.compress_rate
        entry_indices = torch.arange(compressed_len, device=index_scores.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        top_k_indices = index_scores.topk(top_k, dim=-1).indices
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
        return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

    return index_scores.topk(top_k, dim=-1).indices


def _csa_compressor_forward(
    self, hidden_states, q_residual, position_ids, past_key_values, layer_idx
):
    mod = _MOD
    batch, seq_len, _ = hidden_states.shape
    cache_layer = (
        past_key_values.layers[layer_idx] if past_key_values is not None else None
    )
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = (
            kv[:, :usable],
            gate[:, :usable],
            0,
        )
    else:
        chunk_kv, chunk_gate, first_window_position = (
            cache_layer.store_compression_weights("compressor", kv, gate)
        )

    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias

        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full(
            (batch, n_windows, 2 * ratio, self.head_dim), float("-inf")
        )
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        if cache_layer is not None:
            prior_kv, prior_gate = cache_layer.update_overlap_state(
                "compressor", chunk_kv, chunk_gate, self.head_dim
            )
            if prior_kv is not None:
                new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

        compressed = _gated_pool_norm(new_kv, new_gate, self.kv_norm)
        positions = torch.arange(n_windows, device=compressed.device)
        positions = positions * self.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(
            compressed, position_ids=positions, layer_type=self.rope_layer_type
        )
        compressed = mod.apply_rotary_pos_emb(
            compressed.unsqueeze(1), cos, sin
        ).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
    compressed_kv = compressed.unsqueeze(1)

    top_k_indices = self.indexer(
        hidden_states, q_residual, position_ids, past_key_values, layer_idx
    )
    compressed_len = compressed_kv.shape[2]
    valid = top_k_indices >= 0
    safe_indices = torch.where(
        valid, top_k_indices, torch.full_like(top_k_indices, compressed_len)
    )
    block_bias = compressed_kv.new_full(
        (batch, 1, seq_len, compressed_len + 1), float("-inf")
    )
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return compressed_kv, block_bias[..., :compressed_len]


def _hyperconnection_forward(self, hidden_streams):
    return hyperconnection_forward(
        hidden_streams,
        self.input_norm,
        self.fn,
        self.base,
        self.scale,
        self.hc_mult,
        self.hc_sinkhorn_iters,
        self.hc_eps,
        post_mult=2.0,
    )


def patch_deepseek_v4_kernels():
    """Patch the eager attention backend + RoPE + HyperConnection. Idempotent.

    V4 stays on ``_attn_implementation='eager'`` (head_dim 512 bans the real flash
    backends), so transformers still builds the mask and cats the compressor's
    ``block_bias`` — we just replace the eager kernel itself."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mod

    global _MOD
    _MOD = mod

    def _rope(x, cos, sin, unsqueeze_dim=1):
        return apply_rotary_pos_emb_triton(x, cos, sin, unsqueeze_dim)

    mod.eager_attention_forward = _dsv4_attention
    mod.apply_rotary_pos_emb = _rope
    mod.DeepseekV4HyperConnection.forward = _hyperconnection_forward
    mod.DeepseekV4IndexerScorer.forward = _indexer_scorer_forward
    mod.DeepseekV4HCACompressor.forward = _hca_compressor_forward
    mod.DeepseekV4Indexer.forward = _indexer_forward
    mod.DeepseekV4CSACompressor.forward = _csa_compressor_forward
    LOG.info(
        "Patched DeepSeek-V4 with fused Triton kernels (attention/rope/mHC/compressor/indexer)"
    )
