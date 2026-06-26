"""MLA weight-absorption + head-batched sparse gather attention for GLM-5.2 DSA.

In MLA the per-head key/value are produced from a SHARED compressed latent ``c_kv`` [S, kv_lora=512]:
    k_nope[t,h] = c_kv[t] @ W_kb_k[h]        value[t,h] = c_kv[t] @ W_kb_v[h]
so the nope score absorbs kv_b into the query:
    q_nope[h]·k_nope[t,h] = (q_nope[h] @ W_kb_kᵀ[h]) · c_kv[t]
With the rope part (k_rot shared across heads) appended, every head attends the SAME key tensor
    K_shared[t] = cat(c_kv[t] [512], k_rot[t] [64])   (576-wide, shared across heads)
    Q_abs[s,h]  = cat(q_nope[s,h] @ W_kb_kᵀ[h] [512], q_rot[s,h] [64])
    score[s,h,t] = Q_abs[s,h] · K_shared[t] * scale
    out_latent[s,h] = Σ_t softmax(score)[h,t] · c_kv[t]        (512-wide)
    out[s,h] = out_latent[s,h] @ W_kb_v[h]                     (-> v_head_dim 256, then o_proj)

Now all heads share K_shared -> a head-batched MMA (M=H), and the sparse gather loads the 576-wide
shared KV ONCE per query position instead of per head (the win that the per-(b,h,s) GEMV lacked).
The absorption/value GEMMs (q_nope@W_kb_kᵀ, out_latent@W_kb_v) stay in torch and are differentiable,
so gradients flow to kv_b_proj in full-parameter training and to its LoRA adapter in LoRA training;
the kernel differentiates Q_abs / K_shared.

This module currently provides the absorption helpers + the head-batched gather FORWARD kernel,
validated against the eager dense-mask MLA reference. Backward is added next.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._autotune import smem_prune
from .config import KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, V_HEAD_DIM

# Measured: actual SMEM ≈ the single-count tile estimate + ~2KB (tl.trans is handled by the MMA,
# no extra buffer). Use the raw estimate + a small fixed overhead; headroom 0.97 leaves margin
# without pruning borderline-fitting configs (e.g. BH=16,BN=32 which is the fast one on sm120).
_SMEM_OVERHEAD = 3072


def _fwd_smem(cfg, num_stages, **kw):
    # bf16: qn+qr [BH,w] loaded once; c_kv+k_rot [BN,w] pipelined x num_stages.
    w = kw["DL"] + kw["DR"]
    return (cfg["BH"] * w + num_stages * cfg["BN"] * w) * 2 + _SMEM_OVERHEAD


def _bwd_smem(cfg, num_stages, **kw):
    # bwd also holds out_l/dout_l [BH,DL] (loaded once).
    DL, DR = kw["DL"], kw["DR"]
    w = DL + DR
    return (cfg["BH"] * (w + 2 * DL) + num_stages * cfg["BN"] * w) * 2 + _SMEM_OVERHEAD


_ABSORB_CONFIGS = [
    triton.Config({"BH": bh, "BN": bn}, num_warps=warps, num_stages=st)
    for bh in (16, 32)
    for bn in (16, 32, 64, 128)
    for warps in (4, 8)
    for st in (1, 2, 3)
]

DQK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 (compressed score dim)


def split_kv_b(kv_b_weight: torch.Tensor, n_heads: int):
    """Split kv_b_proj.weight [H*(qk_nope+v_head), kv_lora] into per-head
    W_kb_k [H, qk_nope, kv_lora] and W_kb_v [H, v_head, kv_lora]."""
    w = kv_b_weight.view(n_heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK)
    return w[:, :QK_NOPE_HEAD_DIM, :], w[:, QK_NOPE_HEAD_DIM:, :]


def absorb_query(q_nope, q_rot, w_kb_k):
    """q_nope [B,H,S,qk_nope], q_rot [B,H,S,qk_rope], w_kb_k [H,qk_nope,kv_lora] ->
    Q_abs [B,H,S,DQK] = cat(q_nope @ W_kb_kᵀ, q_rot). Differentiable. ``w_kb_k`` is cast to the
    activation dtype (under LoRA the effective kv_b weight promotes to fp32 via the adapters)."""
    q_abs_nope = torch.einsum("bhsn,hnl->bhsl", q_nope, w_kb_k.to(q_nope.dtype))
    return torch.cat([q_abs_nope, q_rot], dim=-1)


def project_value(out_latent, w_kb_v):
    """out_latent [B,H,S,kv_lora] -> out [B,H,S,v_head] = out_latent @ W_kb_vᵀ. Differentiable."""
    return torch.einsum("bhsl,hvl->bhsv", out_latent, w_kb_v.to(out_latent.dtype))


@triton.autotune(
    configs=_ABSORB_CONFIGS,
    key=["TOPK"],
    prune_configs_by={"early_config_prune": smem_prune(_fwd_smem)},
)
@triton.jit
def _absorb_fwd_kernel(
    QABS,
    KSH,
    IDX,
    OUT,
    LSE,
    scale,
    S,
    TOPK,
    H,
    q_offset,
    sqa_b,
    sqa_h,
    sqa_s,
    sqa_d,
    sks_b,
    sks_s,
    sks_d,
    si_b,
    si_s,
    si_t,
    so_b,
    so_h,
    so_s,
    so_d,
    sl_b,
    sl_h,
    sl_s,
    SEQQ,
    SEQK,
    ssq_b,
    ssq_s,
    ssk_b,
    ssk_s,
    DL: tl.constexpr,  # kv_lora_rank (=value latent dim), power of 2
    DR: tl.constexpr,  # qk_rope_head_dim, power of 2
    BH: tl.constexpr,
    BN: tl.constexpr,
    HAS_DOC: tl.constexpr,  # sample-packing: forbid cross-document keys
):
    # S on axis 0 (X, up to 2^31-1); CUDA caps grid Y/Z at 65535 so S must not be on Y/Z.
    s = tl.program_id(0)
    b = tl.program_id(1)
    h0 = tl.program_id(2) * BH
    offs_h = h0 + tl.arange(0, BH)
    hmask = offs_h < H
    offs_l = tl.arange(0, DL)  # compressed-latent dims [0, kv_lora)
    offs_r = tl.arange(0, DR)  # rope dims, stored at [kv_lora, kv_lora+rope)

    # split q_abs into its nope-latent (DL) and rope (DR) parts (avoids padding 576 to a pow2)
    # keep tiles in their native (bf16) dtype for the tensor-core dot (fp32 accumulation)
    base_q = QABS + b * sqa_b + offs_h[:, None] * sqa_h + s * sqa_s
    qn = tl.load(base_q + offs_l[None, :] * sqa_d, mask=hmask[:, None], other=0.0)
    qr = tl.load(
        base_q + (DL + offs_r)[None, :] * sqa_d, mask=hmask[:, None], other=0.0
    )

    m_i = tl.full((BH,), -float("inf"), tl.float32)
    l_i = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, DL), tl.float32)

    s_global = (
        s + q_offset
    )  # CP: query's global position for the causal check (q_offset=0 single-GPU)
    seqq_s = tl.load(SEQQ + b * ssq_b + s * ssq_s) if HAS_DOC else 0
    for t0 in range(0, TOPK, BN):
        offs_t = t0 + tl.arange(0, BN)
        tmask = offs_t < TOPK
        idx = tl.load(
            IDX + b * si_b + s * si_s + offs_t * si_t, mask=tmask, other=0
        ).to(tl.int64)
        valid = tmask & (idx <= s_global)
        if HAS_DOC:
            seqk = tl.load(SEQK + b * ssk_b + idx * ssk_s, mask=tmask, other=-1)
            valid = valid & (seqk == seqq_s)

        # gather the shared KV once: c_kv [BN, DL] (also the value latent) + k_rot [BN, DR]
        base_k = KSH + b * sks_b + idx[:, None] * sks_s
        c_kv = tl.load(base_k + offs_l[None, :] * sks_d, mask=valid[:, None], other=0.0)
        k_rot = tl.load(
            base_k + (DL + offs_r)[None, :] * sks_d, mask=valid[:, None], other=0.0
        )

        scores = tl.dot(qn, tl.trans(c_kv))  # bf16 tensor core, fp32 accum
        scores += tl.dot(qr, tl.trans(k_rot))
        scores = tl.where(valid[None, :], scores * scale, -float("inf"))  # [BH, BN]

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        # A query whose leading topk block is entirely causally/doc-masked has m_new == -inf, so
        # exp(m_i - m_new) = exp(-inf + inf) = NaN. Subtract a finite max (0 there; all scores are
        # -inf so every p is exp(-inf)=0 regardless) — matches the dense path on masked rows.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_safe[:, None])  # [BH, BN]
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(c_kv.dtype), c_kv)
        m_i = m_new

    acc = (
        acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    )  # fully-masked query -> 0 output (loss masked)
    o_ptr = OUT + b * so_b + offs_h[:, None] * so_h + s * so_s + offs_l[None, :] * so_d
    tl.store(o_ptr, acc.to(OUT.dtype.element_ty), mask=hmask[:, None])
    tl.store(LSE + b * sl_b + offs_h * sl_h + s * sl_s, m_i + tl.log(l_i), mask=hmask)


def _bwd_prune_sm90_safe(configs, named_args, **kwargs):
    """sm90 (Hopper) trips a nondeterministic invalid-PC (CUDA 718) while autotune trials the full
    bwd grid (each config is individually clean — it's the trialing that corrupts the context). Force
    a single sanitizer-clean config there; elsewhere fall back to SMEM pruning."""
    import torch

    base = smem_prune(_bwd_smem)
    kept = base(configs, named_args, **kwargs)
    if torch.cuda.get_device_capability() == (9, 0):
        safe = [
            c
            for c in kept
            if c.kwargs.get("BH") == 32
            and c.kwargs.get("BN") == 64
            and c.num_warps == 8
            and c.num_stages == 3
        ]
        if safe:
            return safe[:1]
        return sorted(kept, key=lambda c: (c.kwargs.get("BN", 0), c.num_stages))[:1]
    return kept


@triton.autotune(
    configs=_ABSORB_CONFIGS,
    key=["TOPK"],
    reset_to_zero=[
        "DKSH"
    ],  # dk_shared is atomic-accumulated across positions + head-tiles
    prune_configs_by={"early_config_prune": _bwd_prune_sm90_safe},
)
@triton.jit
def _absorb_bwd_kernel(
    QABS,
    KSH,
    IDX,
    OUTL,
    DOUTL,
    LSE,
    DQABS,
    DKSH,
    scale,
    S,
    TOPK,
    H,
    q_offset,
    sqa_b,
    sqa_h,
    sqa_s,
    sqa_d,
    sks_b,
    sks_s,
    sks_d,
    si_b,
    si_s,
    si_t,
    so_b,
    so_h,
    so_s,
    so_d,
    sl_b,
    sl_h,
    sl_s,
    SEQQ,
    SEQK,
    ssq_b,
    ssq_s,
    ssk_b,
    ssk_s,
    DL: tl.constexpr,
    DR: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    HAS_DOC: tl.constexpr,
):
    # S on axis 0 (X, up to 2^31-1); CUDA caps grid Y/Z at 65535 so S must not be on Y/Z.
    s = tl.program_id(0)
    b = tl.program_id(1)
    h0 = tl.program_id(2) * BH
    offs_h = h0 + tl.arange(0, BH)
    hmask = offs_h < H
    offs_l = tl.arange(0, DL)
    offs_r = tl.arange(0, DR)

    base_q = QABS + b * sqa_b + offs_h[:, None] * sqa_h + s * sqa_s
    qn = tl.load(base_q + offs_l[None, :] * sqa_d, mask=hmask[:, None], other=0.0)
    qr = tl.load(
        base_q + (DL + offs_r)[None, :] * sqa_d, mask=hmask[:, None], other=0.0
    )
    base_o = (
        OUTL + b * so_b + offs_h[:, None] * so_h + s * so_s + offs_l[None, :] * so_d
    )
    out_l = tl.load(base_o, mask=hmask[:, None], other=0.0).to(tl.float32)  # [BH, DL]
    dout_l = tl.load(
        DOUTL + b * so_b + offs_h[:, None] * so_h + s * so_s + offs_l[None, :] * so_d,
        mask=hmask[:, None],
        other=0.0,
    )  # [BH, DL]
    lse = tl.load(LSE + b * sl_b + offs_h * sl_h + s * sl_s, mask=hmask, other=0.0)
    delta = tl.sum(dout_l.to(tl.float32) * out_l, axis=1)  # [BH]

    s_global = s + q_offset
    seqq_s = tl.load(SEQQ + b * ssq_b + s * ssq_s) if HAS_DOC else 0
    dqn = tl.zeros((BH, DL), tl.float32)
    dqr = tl.zeros((BH, DR), tl.float32)
    for t0 in range(0, TOPK, BN):
        offs_t = t0 + tl.arange(0, BN)
        tmask = offs_t < TOPK
        idx = tl.load(
            IDX + b * si_b + s * si_s + offs_t * si_t, mask=tmask, other=0
        ).to(tl.int64)
        valid = tmask & (idx <= s_global)
        if HAS_DOC:
            seqk = tl.load(SEQK + b * ssk_b + idx * ssk_s, mask=tmask, other=-1)
            valid = valid & (seqk == seqq_s)
        base_k = KSH + b * sks_b + idx[:, None] * sks_s
        c_kv = tl.load(base_k + offs_l[None, :] * sks_d, mask=valid[:, None], other=0.0)
        k_rot = tl.load(
            base_k + (DL + offs_r)[None, :] * sks_d, mask=valid[:, None], other=0.0
        )

        score = tl.dot(qn, tl.trans(c_kv)) + tl.dot(qr, tl.trans(k_rot))
        score = tl.where(valid[None, :], score * scale, -float("inf"))
        lse_safe = tl.where(
            lse == -float("inf"), 0.0, lse
        )  # fully-masked query: avoid exp(-inf+inf)=NaN
        p = tl.exp(score - lse_safe[:, None])  # [BH, BN]
        dp = tl.dot(dout_l, tl.trans(c_kv))  # [BH, BN] = dout_latent·c_kv
        ds = (p * (dp - delta[:, None])) * scale  # [BH, BN]  grad wrt scaled score
        ds_b = ds.to(c_kv.dtype)
        p_b = p.to(c_kv.dtype)

        dqn += tl.dot(ds_b, c_kv)  # [BH, DL]
        dqr += tl.dot(ds_b, k_rot)  # [BH, DR]
        # dc_kv = value-path (Σ_h p·dout_latent) + score-path (Σ_h ds·qn); dk_rot = Σ_h ds·qr
        dc = tl.dot(tl.trans(p_b), dout_l.to(c_kv.dtype)) + tl.dot(
            tl.trans(ds_b), qn
        )  # [BN, DL]
        dkr = tl.dot(tl.trans(ds_b), qr)  # [BN, DR]
        dk_ptr = DKSH + b * sks_b + idx[:, None] * sks_s
        tl.atomic_add(
            dk_ptr + offs_l[None, :] * sks_d, tl.where(valid[:, None], dc, 0.0)
        )
        tl.atomic_add(
            dk_ptr + (DL + offs_r)[None, :] * sks_d, tl.where(valid[:, None], dkr, 0.0)
        )

    dq_ptr = DQABS + b * sqa_b + offs_h[:, None] * sqa_h + s * sqa_s
    tl.store(
        dq_ptr + offs_l[None, :] * sqa_d,
        dqn.to(DQABS.dtype.element_ty),
        mask=hmask[:, None],
    )
    tl.store(
        dq_ptr + (DL + offs_r)[None, :] * sqa_d,
        dqr.to(DQABS.dtype.element_ty),
        mask=hmask[:, None],
    )


def _grid_fn(B, S, H):
    def grid(m):
        return (S, B, triton.cdiv(H, m["BH"]))

    return grid


class _MlaAbsorbAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q_abs, k_shared, topk_idx, scale, q_offset, seq_q=None, seq_k=None
    ):
        B, H, S, _ = (
            q_abs.shape
        )  # S = local queries; k_shared may be longer (global, under CP)
        DV = KV_LORA_RANK
        TOPK = topk_idx.shape[-1]
        q_abs, k_shared = q_abs.contiguous(), k_shared.contiguous()
        idx = topk_idx.contiguous()
        has_doc = seq_q is not None and seq_k is not None
        if has_doc:
            seq_q = seq_q.contiguous()
            seq_k = seq_k.contiguous()
            doc_args = (
                seq_q,
                seq_k,
                seq_q.stride(0),
                seq_q.stride(1),
                seq_k.stride(0),
                seq_k.stride(1),
            )
        else:
            doc_args = (idx, idx, 0, 0, 0, 0)
        out = torch.empty(B, H, S, DV, device=q_abs.device, dtype=q_abs.dtype)
        lse = torch.empty(B, H, S, device=q_abs.device, dtype=torch.float32)
        _absorb_fwd_kernel[_grid_fn(B, S, H)](
            q_abs,
            k_shared,
            idx,
            out,
            lse,
            float(scale),
            S,
            TOPK,
            H,
            int(q_offset),
            q_abs.stride(0),
            q_abs.stride(1),
            q_abs.stride(2),
            q_abs.stride(3),
            k_shared.stride(0),
            k_shared.stride(1),
            k_shared.stride(2),
            idx.stride(0),
            idx.stride(1),
            idx.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            *doc_args,
            DL=KV_LORA_RANK,
            DR=QK_ROPE_HEAD_DIM,
            HAS_DOC=has_doc,
        )
        ctx.save_for_backward(q_abs, k_shared, idx, out, lse, doc_args[0], doc_args[1])
        ctx.scale = float(scale)
        ctx.q_offset = int(q_offset)
        ctx.has_doc = has_doc
        return out

    @staticmethod
    def backward(ctx, dout):
        q_abs, k_shared, idx, out, lse, seq_q, seq_k = ctx.saved_tensors
        has_doc = ctx.has_doc
        if has_doc:
            doc_args = (
                seq_q,
                seq_k,
                seq_q.stride(0),
                seq_q.stride(1),
                seq_k.stride(0),
                seq_k.stride(1),
            )
        else:
            doc_args = (idx, idx, 0, 0, 0, 0)
        B, H, S, DQk = q_abs.shape
        Skv = k_shared.shape[1]  # global key count (>= S under CP)
        TOPK = idx.shape[-1]
        dout = dout.contiguous()
        dq_abs = torch.zeros(B, H, S, DQk, device=q_abs.device, dtype=q_abs.dtype)
        # dk_shared scatters at GLOBAL key positions -> size it like k_shared, not the local queries.
        dk_shared = torch.zeros(B, Skv, DQk, device=q_abs.device, dtype=torch.float32)
        _absorb_bwd_kernel[_grid_fn(B, S, H)](
            q_abs,
            k_shared,
            idx,
            out,
            dout,
            lse,
            dq_abs,
            dk_shared,
            ctx.scale,
            S,
            TOPK,
            H,
            ctx.q_offset,
            q_abs.stride(0),
            q_abs.stride(1),
            q_abs.stride(2),
            q_abs.stride(3),
            dk_shared.stride(0),
            dk_shared.stride(1),
            dk_shared.stride(2),
            idx.stride(0),
            idx.stride(1),
            idx.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            *doc_args,
            DL=KV_LORA_RANK,
            DR=QK_ROPE_HEAD_DIM,
            HAS_DOC=has_doc,
        )
        return dq_abs, dk_shared.to(k_shared.dtype), None, None, None, None, None


def mla_absorb_attn(
    q_abs, k_shared, topk_idx, scale, q_offset=0, seq_q=None, seq_k=None
):
    """Differentiable head-batched MLA-absorption sparse attention. q_abs [B,H,S,576] (local
    queries), k_shared [B,Skv,576] (first kv_lora cols = c_kv; Skv>=S under context parallel),
    topk_idx [B,S,T] int32 referencing GLOBAL key positions. ``q_offset`` is the global position of
    local query 0 (for causal masking under CP). Returns out_latent [B,H,S,kv_lora]."""
    return _MlaAbsorbAttn.apply(
        q_abs, k_shared, topk_idx, scale, q_offset, seq_q, seq_k
    )


def mla_absorb_attn_fwd(q_abs, k_shared, topk_idx, scale, q_offset=0):
    """Forward-only (no autograd) — for benchmarking. Returns (out_latent, None)."""
    with torch.no_grad():
        return _MlaAbsorbAttn.apply(q_abs, k_shared, topk_idx, scale, q_offset), None
