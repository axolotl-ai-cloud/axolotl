"""Fused Lightning-Indexer scorer for DeepSeek-V4 CSA (forward-only).

Reference (DeepseekV4IndexerScorer.forward):
    scores  = matmul(q.float(), ck.float().transpose).unsqueeze(1)   # [B, S, H, T]
    scores  = relu(scores) * softmax_scale
    weights = weights_proj(hidden).float() * weights_scaling          # [B, S, H]
    index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)        # [B, S, T]

i.e.  out[b,s,t] = softmax_scale * Σ_h w[b,s,h] · relu(Σ_d q[b,s,h,d]·ck[b,t,d]).

Eager materializes the full [B,S,H,T] fp32 score tensor twice (matmul output + the
``scores*weights`` product) — with H=64 index heads this is the #1 transient memory
hotspot. Fusing the H-reduction collapses it straight to [B,S,T], never holding the
H axis.

FORWARD ONLY: ``index_scores`` is consumed solely by ``topk(...).indices`` (a non-
differentiable LongTensor) to build the gather mask, so no gradient ever flows back
through the scorer. We return a plain (no-grad) tensor — matching eager semantics — and
skip the backward entirely.
"""

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
    CK,
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
    sck_b,
    sck_t,
    sck_d,
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

    # ck is independent of h, so load it once outside the head loop
    ck_ptr = CK + b * sck_b + offs_d[:, None] * sck_d + offs_t[None, :] * sck_t
    ck = tl.load(ck_ptr, mask=tmask[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros((BS, BT), dtype=tl.float32)
    for h in range(H):
        q_ptr = (
            Q + b * sq_b + offs_s[:, None] * sq_s + h * sq_h + offs_d[None, :] * sq_d
        )
        q = tl.load(q_ptr, mask=smask[:, None], other=0.0).to(tl.float32)
        qk = tl.dot(q, ck, input_precision="ieee")
        qk = tl.maximum(qk, 0.0) * softmax_scale
        w = tl.load(W + b * sw_b + offs_s * sw_s + h * sw_h, mask=smask, other=0.0).to(
            tl.float32
        )
        acc += qk * w[:, None]

    out_ptr = OUT + b * so_b + offs_s[:, None] * so_s + offs_t[None, :] * so_t
    tl.store(out_ptr, acc, mask=smask[:, None] & tmask[None, :])


def indexer_scores(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """q [B,S,H,Dh], compressed_kv [B,T,Dh], weights [B,S,H] (already ·weights_scaling).
    Returns index_scores [B,S,T] (fp32, no grad). Drop-in for DeepseekV4IndexerScorer."""
    B, S, H, DH = q.shape
    T = compressed_kv.shape[1]
    out = torch.empty(B, S, T, device=q.device, dtype=torch.float32)
    if T == 0 or S == 0:
        return out
    # compressed_kv may arrive fp32 (keep_in_fp32 compressor) while q is bf16.
    if compressed_kv.dtype != q.dtype:
        compressed_kv = compressed_kv.to(q.dtype)
    q = q.contiguous()
    ck = compressed_kv.contiguous()
    w = weights.contiguous()
    grid = lambda m: (B, triton.cdiv(S, m["BS"]), triton.cdiv(T, m["BT"]))
    _indexer_score_kernel[grid](
        q,
        ck,
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
        ck.stride(0),
        ck.stride(1),
        ck.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        DH=DH,
    )
    return out
