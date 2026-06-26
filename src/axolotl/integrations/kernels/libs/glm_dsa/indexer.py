"""Fused GLM-5.2 DSA Lightning-Indexer scorer + top-k (forward-only).

Reference (GlmMoeDsaIndexer.forward, post-RoPE):
    scores       = relu(softmax_scale * q @ k^T)                         # [B, S, H, T]
    index_scores = Σ_h weights[b,s,h] * scores[b,s,h,t]                  # [B, S, T]
    (+ causal mask) -> topk(index_topk).indices                          # [B, S, topk] int32

where q [B,S,H,D] (H=index_n_heads=32, D=index_head_dim=128), k [B,S,D] (shared across heads),
weights[b,s,h] = weights_proj(h)[b,s,h] * H**-0.5.

Eager materializes the full [B,S,H,T] fp32 score tensor (H=32) — the transient hotspot. Fusing
the H-reduction collapses straight to [B,S,T], never holding the H axis. The indexer runs under
``@torch.no_grad`` (its output only feeds ``topk().indices``), so this is FORWARD-ONLY.

The scoring kernel is the DeepSeek-V4 ``indexer_scores`` kernel (identical math: relu(sm·qk) summed
over weighted heads); GLM adds the causal mask + top-k here. ``relu(sm·x) == sm·relu(x)`` for sm>0,
so applying the scale inside or outside relu is equivalent.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BS": bs, "BT": bt}, num_warps=w, num_stages=s)
        for bs in (32, 64, 128)
        for bt in (32, 64, 128)
        for w in (4, 8)
        for s in (2, 3)
    ],
    key=["S", "T", "H"],
)
@triton.jit
def _indexer_score_kernel(
    Q,
    K,
    W,
    OUT,
    softmax_scale,
    S,
    T,
    H,
    sq_b,
    sq_s,
    sq_h,
    sq_d,
    sk_b,
    sk_t,
    sk_d,
    sw_b,
    sw_s,
    sw_h,
    so_b,
    so_s,
    so_t,
    DH: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
):
    b = tl.program_id(0)
    s0 = tl.program_id(1) * BS
    t0 = tl.program_id(2) * BT
    offs_s = s0 + tl.arange(0, BS)
    offs_t = t0 + tl.arange(0, BT)
    offs_d = tl.arange(0, DH)
    smask = offs_s < S
    tmask = offs_t < T

    # k block [DH, BT] (transposed for q @ kᵀ), shared across heads
    k_ptr = K + b * sk_b + offs_d[:, None] * sk_d + offs_t[None, :] * sk_t
    k = tl.load(k_ptr, mask=tmask[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros((BS, BT), dtype=tl.float32)
    for h in range(H):
        q_ptr = (
            Q + b * sq_b + offs_s[:, None] * sq_s + h * sq_h + offs_d[None, :] * sq_d
        )
        q = tl.load(q_ptr, mask=smask[:, None], other=0.0).to(tl.float32)
        qk = tl.dot(q, k, input_precision="ieee")  # [BS, BT]
        qk = tl.maximum(qk, 0.0) * softmax_scale  # relu · scale
        w = tl.load(W + b * sw_b + offs_s * sw_s + h * sw_h, mask=smask, other=0.0).to(
            tl.float32
        )
        acc += qk * w[:, None]

    out_ptr = OUT + b * so_b + offs_s[:, None] * so_s + offs_t[None, :] * so_t
    tl.store(out_ptr, acc, mask=smask[:, None] & tmask[None, :])


def indexer_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """q [B,S,H,D], k [B,S,D] (shared across heads), weights [B,S,H] (already ·H**-0.5).
    Returns index_scores [B,S,T=S] (fp32, no grad). Fuses the H-reduction."""
    B, S, H, DH = q.shape
    T = k.shape[1]
    out = torch.empty(B, S, T, device=q.device, dtype=torch.float32)
    if T == 0 or S == 0:
        return out
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    q, k, w = q.contiguous(), k.contiguous(), weights.contiguous()

    def grid(m):
        return (B, triton.cdiv(S, m["BS"]), triton.cdiv(T, m["BT"]))

    _indexer_score_kernel[grid](
        q,
        k,
        w,
        out,
        float(softmax_scale),
        S,
        T,
        H,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        DH=DH,
    )
    return out


@torch.no_grad()
def indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    index_topk: int,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    seq_q: torch.Tensor | None = None,
    seq_k: torch.Tensor | None = None,
    q_offset: int = 0,
) -> torch.Tensor:
    """Fused indexer scoring + causal mask + top-k. Returns int32 indices [B,S,topk].

    ``attention_mask`` (additive, broadcastable to [B,S,T]) is added if given; otherwise a causal
    mask from ``position_ids`` (or arange) is applied. Mirrors GlmMoeDsaIndexer's masking + topk.

    Under sample packing, ``seq_q`` [B,S] / ``seq_k`` [B,T] give each query/key its document id;
    the mask then forbids cross-document keys (a key in an earlier packed document is causally
    "before" the query by global index but must not be attended). ``q_offset`` shifts local query
    indices to global positions (context parallel). The selected indices still rank same-document
    keys first; the attention kernel re-applies the same document mask (``mla_attn`` forces the
    dense path under packing) so cross-document keys returned when ``topk == T`` are dropped.
    """
    scores = indexer_scores(q, k, weights, softmax_scale)  # [B,S,T]
    B, S, T = scores.shape
    packed = seq_q is not None and seq_k is not None
    if attention_mask is not None and not packed:
        scores = scores + attention_mask.to(scores.dtype)
    else:
        kpos = torch.arange(T, device=scores.device)
        qpos = q_offset + torch.arange(S, device=scores.device)
        causal = kpos[None, None, :] > qpos[None, :, None]
        if seq_q is not None and seq_k is not None:
            causal = causal | (seq_k[:, None, :] != seq_q[:, :, None])
        scores = scores.masked_fill(causal, float("-inf"))
    topk = min(index_topk, T)
    return scores.topk(topk, dim=-1).indices.to(torch.int32)
