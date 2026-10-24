"""Triton FlashAttention for head_dim 512 with segment masks, GQA and Q length != KV length.

``flash_d512.py`` covers square causal varlen attention with repeated KV heads. Packed
encoder/canvas models (DiffusionGemma) need more: the decoder's canvas queries attend to their
document's clean prefix inside the encoder KV cache plus their own canvas segment, so Q and KV
have different lengths and every query row sees up to two disjoint key ranges.

The mask is described per row instead of per element, so nothing quadratic is materialised:

- ``q_ranges [B, M, 4]`` (int32): query row ``i`` may see keys in ``[a0, b0) u [a1, b1)``.
- ``kv_valid [B, N]`` (int8): key ``j`` is usable at all. A row sees exactly
  ``(range0 u range1) n kv_valid``.
- ``kv_q_ranges [B, N, 2]`` (int32): a superset ``[qa, qb)`` of the query rows that see key ``j``.
  The dK/dV kernel only visits those rows; the exact row mask is re-applied inside.

Causal varlen attention is the special case ``range0 = [doc_start, i + 1)``, so the same kernel
serves the encoder's global layers. KV heads are shared natively (GQA): no repeated K/V copies,
and dK/dV for one KV head accumulate over its query-head group inside one program (no atomics).

Rows that see no key produce zeros (FlexAttention's convention for fully masked rows).

Block sizes are chosen for sm120's 99 KB of shared memory per block at head_dim 512.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from axolotl.kernels.op_registry import register_kernel_op
from axolotl.monkeypatch.attention.segment_mask import SegmentAttentionMeta

# Larger than any sequence length; marks "no range" in block reductions.
_EMPTY = tl.constexpr(2**30)
# Finite fill so fully masked blocks never produce inf - inf.
_MASKED = tl.constexpr(-1.0e9)


@triton.jit
def _row_mask(cur_n, a0, b0, a1, b1, kval, n_ctx):
    in0 = (cur_n[None, :] >= a0[:, None]) & (cur_n[None, :] < b0[:, None])
    in1 = (cur_n[None, :] >= a1[:, None]) & (cur_n[None, :] < b1[:, None])
    return (in0 | in1) & (kval[None, :] != 0) & (cur_n[None, :] < n_ctx)


@triton.jit
def _block_span(lo, hi, BLOCK: tl.constexpr):
    """Union [lo, hi) of non-empty row ranges, aligned outward to BLOCK; (0, 0) if none.

    An empty span collapses to a point, so the set difference used by the callers
    ([lo1, min(e1, lo0)) and [max(lo1, e0), e1)) still covers range 1 exactly once."""
    start = (tl.min(tl.where(lo < hi, lo, _EMPTY)) // BLOCK) * BLOCK
    end = tl.cdiv(tl.max(tl.where(lo < hi, hi, 0)), BLOCK) * BLOCK
    return tl.minimum(start, end), end


@triton.jit
def _fwd_blocks(
    acc,
    m_i,
    l_i,
    q,
    kb,
    vb,
    VAL,
    skn,
    skd,
    svn,
    svd,
    svl,
    lo,
    hi,
    a0,
    b0,
    a1,
    b1,
    offs_d,
    sm_scale,
    N_KV,
    BLOCK_N: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(lo, hi, BLOCK_N):
        cur = start_n + offs_n
        inb = cur < N_KV
        k = tl.load(
            kb + cur[None, :] * skn + offs_d[:, None] * skd,
            mask=inb[None, :],
            other=0.0,
        )
        kval = tl.load(VAL + cur * svl, mask=inb, other=0)
        mask = _row_mask(cur, a0, b0, a1, b1, kval, N_KV)
        qk = tl.where(mask, tl.dot(q, k) * sm_scale, _MASKED)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.where(mask, tl.exp(qk - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            vb + cur[:, None] * svn + offs_d[None, :] * svd,
            mask=inb[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, m_i, l_i


def _fwd_smem_bytes(block_n, head_dim, num_stages):
    # K and V tiles, (1 + num_stages)-buffered; same calibration as flash_attn_d512.
    return block_n * head_dim * 2 * (1 + num_stages) + 2048


def _prune_fwd_configs(configs, named_args, **kwargs):
    head_dim = kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM"))
    if head_dim is None or not torch.cuda.is_available():
        return list(configs)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    limit = (
        getattr(props, "shared_memory_per_block_optin", 0)
        or props.shared_memory_per_block
    )
    fit = [
        c
        for c in configs
        if _fwd_smem_bytes(c.kwargs["BLOCK_N"], head_dim, c.num_stages) <= limit
    ]
    return fit or [min(configs, key=lambda c: c.num_stages)]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=s)
        for s in (1, 2)
    ]
    + [triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=1)],
    key=["HEAD_DIM", "GROUPS"],
    prune_configs_by={"early_config_prune": _prune_fwd_configs},
)
@triton.jit
def _seg_fwd(
    Q,
    K,
    V,
    Out,
    L,
    QR,
    VAL,
    sm_scale,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    sob,
    soh,
    som,
    sod,
    srb,
    srm,
    svlb,
    svl,
    H,
    N_Q,
    N_KV,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    off_kh = off_h // GROUPS
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_Q
    q = tl.load(
        Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=row_ok[:, None],
        other=0.0,
    )
    rr = QR + off_b * srb + offs_m * srm
    a0 = tl.load(rr + 0, mask=row_ok, other=0)
    b0 = tl.load(rr + 1, mask=row_ok, other=0)
    a1 = tl.load(rr + 2, mask=row_ok, other=0)
    b1 = tl.load(rr + 3, mask=row_ok, other=0)
    lo0, e0 = _block_span(a0, b0, BLOCK_N)
    lo1, e1 = _block_span(a1, b1, BLOCK_N)
    kb = K + off_b * skb + off_kh * skh
    vb = V + off_b * svb + off_kh * svh
    val = VAL + off_b * svlb
    m_i = tl.full([BLOCK_M], _MASKED, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    # Three disjoint block spans: range 0's span, then range 1's span minus range 0's. Both
    # loops test the full two-range row mask, so a key block is never counted twice.
    acc, m_i, l_i = _fwd_blocks(
        acc, m_i, l_i, q, kb, vb, val, skn, skd, svn, svd, svl,
        lo0, e0, a0, b0, a1, b1, offs_d, sm_scale, N_KV, BLOCK_N,
    )  # fmt: skip
    acc, m_i, l_i = _fwd_blocks(
        acc, m_i, l_i, q, kb, vb, val, skn, skd, svn, svd, svl,
        lo1, tl.minimum(e1, lo0), a0, b0, a1, b1, offs_d, sm_scale, N_KV, BLOCK_N,
    )  # fmt: skip
    acc, m_i, l_i = _fwd_blocks(
        acc, m_i, l_i, q, kb, vb, val, skn, skd, svn, svd, svl,
        tl.maximum(lo1, e0), e1, a0, b0, a1, b1, offs_d, sm_scale, N_KV, BLOCK_N,
    )  # fmt: skip
    empty = l_i == 0.0
    acc = acc / tl.where(empty, 1.0, l_i)[:, None]
    tl.store(
        Out + off_b * sob + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        acc.to(Out.dtype.element_ty),
        mask=row_ok[:, None],
    )
    # Empty rows: any finite value works, their probabilities are masked to 0 in backward.
    lse = tl.where(empty, 0.0, m_i + tl.log(tl.where(empty, 1.0, l_i)))
    tl.store(L + off_bh * N_Q + offs_m, lse, mask=row_ok)


@triton.jit
def _seg_bwd_pre(
    O,  # noqa: E741
    DO,
    Delta,
    sob,
    soh,
    som,
    sod,
    sdb,
    sdh,
    sdm,
    sdd,
    H,
    N_Q,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_Q
    o = tl.load(
        O + off_b * sob + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        mask=row_ok[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + off_b * sdb + off_h * sdh + offs_m[:, None] * sdm + offs_d[None, :] * sdd,
        mask=row_ok[:, None],
        other=0.0,
    )
    tl.store(
        Delta + off_bh * N_Q + offs_m,
        tl.sum(o.to(tl.float32) * do.to(tl.float32), 1),
        mask=row_ok,
    )


@triton.jit
def _seg_bwd_dkdv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    L,
    Delta,
    QR,
    VAL,
    KR,
    sm_scale,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    sdb,
    sdh,
    sdm,
    sdd,
    sgb,
    sgh,
    sgn,
    sgd,
    srb,
    srm,
    svlb,
    svl,
    skrb,
    skrn,
    H,
    HKV,
    N_Q,
    N_KV,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_bkh = tl.program_id(1)
    off_b = off_bkh // HKV
    off_kh = off_bkh % HKV
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    col_ok = offs_n < N_KV
    k = tl.load(
        K + off_b * skb + off_kh * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
        mask=col_ok[:, None],
        other=0.0,
    )
    v = tl.load(
        V + off_b * svb + off_kh * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
        mask=col_ok[:, None],
        other=0.0,
    )
    kval = tl.load(VAL + off_b * svlb + offs_n * svl, mask=col_ok, other=0)
    kr = KR + off_b * skrb + offs_n * skrn
    qa = tl.load(kr + 0, mask=col_ok & (kval != 0), other=0)
    qb = tl.load(kr + 1, mask=col_ok & (kval != 0), other=0)
    m_lo, m_hi = _block_span(qa, qb, BLOCK_M)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    for g in range(GROUPS):
        off_h = off_kh * GROUPS + g
        qbase = Q + off_b * sqb + off_h * sqh
        dobase = DO + off_b * sdb + off_h * sdh
        lbase = (off_b * H + off_h) * N_Q
        for start_m in range(m_lo, m_hi, BLOCK_M):
            cur_m = start_m + offs_m
            row_ok = cur_m < N_Q
            q = tl.load(
                qbase + cur_m[:, None] * sqm + offs_d[None, :] * sqd,
                mask=row_ok[:, None],
                other=0.0,
            )
            do = tl.load(
                dobase + cur_m[:, None] * sdm + offs_d[None, :] * sdd,
                mask=row_ok[:, None],
                other=0.0,
            )
            lse = tl.load(L + lbase + cur_m, mask=row_ok, other=0.0)
            delta = tl.load(Delta + lbase + cur_m, mask=row_ok, other=0.0)
            rr = QR + off_b * srb + cur_m * srm
            a0 = tl.load(rr + 0, mask=row_ok, other=0)
            b0 = tl.load(rr + 1, mask=row_ok, other=0)
            a1 = tl.load(rr + 2, mask=row_ok, other=0)
            b1 = tl.load(rr + 3, mask=row_ok, other=0)
            mask = _row_mask(offs_n, a0, b0, a1, b1, kval, N_KV)
            s = tl.dot(q, tl.trans(k)) * sm_scale
            p = tl.where(mask, tl.exp(s - lse[:, None]), 0.0)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delta[:, None]) * sm_scale).to(q.dtype)
            dk += tl.dot(tl.trans(ds), q)
    gbase = off_b * sgb + off_kh * sgh + offs_n[:, None] * sgn + offs_d[None, :] * sgd
    tl.store(DK + gbase, dk.to(DK.dtype.element_ty), mask=col_ok[:, None])
    tl.store(DV + gbase, dv.to(DV.dtype.element_ty), mask=col_ok[:, None])


@triton.jit
def _dq_blocks(
    dq,
    q,
    do,
    lse,
    delta,
    kb,
    vb,
    VAL,
    skn,
    skd,
    svn,
    svd,
    svl,
    lo,
    hi,
    a0,
    b0,
    a1,
    b1,
    offs_d,
    sm_scale,
    N_KV,
    BLOCK_N: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(lo, hi, BLOCK_N):
        cur = start_n + offs_n
        inb = cur < N_KV
        k = tl.load(
            kb + cur[:, None] * skn + offs_d[None, :] * skd,
            mask=inb[:, None],
            other=0.0,
        )
        v = tl.load(
            vb + cur[:, None] * svn + offs_d[None, :] * svd,
            mask=inb[:, None],
            other=0.0,
        )
        kval = tl.load(VAL + cur * svl, mask=inb, other=0)
        mask = _row_mask(cur, a0, b0, a1, b1, kval, N_KV)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.where(mask, tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None]) * sm_scale).to(k.dtype)
        dq += tl.dot(ds, k)
    return dq


@triton.jit
def _seg_bwd_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    L,
    Delta,
    QR,
    VAL,
    sm_scale,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    sdb,
    sdh,
    sdm,
    sdd,
    sgb,
    sgh,
    sgm,
    sgd,
    srb,
    srm,
    svlb,
    svl,
    H,
    N_Q,
    N_KV,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    off_kh = off_h // GROUPS
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_Q
    q = tl.load(
        Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=row_ok[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + off_b * sdb + off_h * sdh + offs_m[:, None] * sdm + offs_d[None, :] * sdd,
        mask=row_ok[:, None],
        other=0.0,
    )
    lse = tl.load(L + off_bh * N_Q + offs_m, mask=row_ok, other=0.0)
    delta = tl.load(Delta + off_bh * N_Q + offs_m, mask=row_ok, other=0.0)
    rr = QR + off_b * srb + offs_m * srm
    a0 = tl.load(rr + 0, mask=row_ok, other=0)
    b0 = tl.load(rr + 1, mask=row_ok, other=0)
    a1 = tl.load(rr + 2, mask=row_ok, other=0)
    b1 = tl.load(rr + 3, mask=row_ok, other=0)
    lo0, e0 = _block_span(a0, b0, BLOCK_N)
    lo1, e1 = _block_span(a1, b1, BLOCK_N)
    kb = K + off_b * skb + off_kh * skh
    vb = V + off_b * svb + off_kh * svh
    val = VAL + off_b * svlb
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    dq = _dq_blocks(
        dq, q, do, lse, delta, kb, vb, val, skn, skd, svn, svd, svl,
        lo0, e0, a0, b0, a1, b1, offs_d, sm_scale, N_KV, BLOCK_N,
    )  # fmt: skip
    dq = _dq_blocks(
        dq, q, do, lse, delta, kb, vb, val, skn, skd, svn, svd, svl,
        lo1, tl.minimum(e1, lo0), a0, b0, a1, b1, offs_d, sm_scale, N_KV, BLOCK_N,
    )  # fmt: skip
    dq = _dq_blocks(
        dq, q, do, lse, delta, kb, vb, val, skn, skd, svn, svd, svl,
        tl.maximum(lo1, e0), e1, a0, b0, a1, b1, offs_d, sm_scale, N_KV, BLOCK_N,
    )  # fmt: skip
    tl.store(
        DQ + off_b * sgb + off_h * sgh + offs_m[:, None] * sgm + offs_d[None, :] * sgd,
        dq.to(DQ.dtype.element_ty),
        mask=row_ok[:, None],
    )


# Backward tiles: K/V/dK/dV stay resident per program, so BLOCK_N is small at head_dim 512.
_BWD_BLOCK_M = 16
_BWD_BLOCK_N = 32
_BWD_DKDV_WARPS = 8
_BWD_DQ_WARPS = 4


@register_kernel_op("flash_attn_d512_seg_fwd")
def _seg_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    kv_valid: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, M, D = q.shape
    HKV, N = k.shape[1], k.shape[2]
    # [B, M, H, D] storage, returned as a [B, H, M, D] view: the attention interface wants
    # [B, M, H, D] back, so the output needs no transpose copy.
    out = torch.empty((B, M, H, D), device=q.device, dtype=q.dtype).transpose(1, 2)
    lse = torch.empty((B * H, M), device=q.device, dtype=torch.float32)
    _seg_fwd[lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), B * H)](
        q, k, v, out, lse, q_ranges, kv_valid, scale,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        q_ranges.stride(0), q_ranges.stride(1),
        *kv_valid.stride(),
        H, M, N,
        GROUPS=H // HKV, HEAD_DIM=D,
    )  # fmt: skip
    return out, lse


@_seg_fwd_op.register_fake
def _(q, k, v, q_ranges, kv_valid, scale):
    B, H, M, D = q.shape
    out = q.new_empty((B, M, H, D)).transpose(1, 2)
    return out, q.new_empty((B * H, M), dtype=torch.float32)


@register_kernel_op("flash_attn_d512_seg_bwd")
def _seg_bwd_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q_ranges: torch.Tensor,
    kv_valid: torch.Tensor,
    kv_q_ranges: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, M, D = q.shape
    HKV, N = k.shape[1], k.shape[2]
    groups = H // HKV
    dq = torch.empty((B, H, M, D), device=q.device, dtype=q.dtype)
    dk = torch.empty((B, HKV, N, D), device=k.device, dtype=k.dtype)
    dv = torch.empty((B, HKV, N, D), device=v.device, dtype=v.dtype)
    delta = torch.empty((B * H, M), device=q.device, dtype=torch.float32)
    bm, bn = _BWD_BLOCK_M, _BWD_BLOCK_N
    _seg_bwd_pre[(triton.cdiv(M, bm), B * H)](
        out, do, delta, *out.stride(), *do.stride(), H, M,
        HEAD_DIM=D, BLOCK_M=bm, num_warps=4,
    )  # fmt: skip
    _seg_bwd_dkdv[(triton.cdiv(N, bn), B * HKV)](
        q, k, v, do, dk, dv, lse, delta, q_ranges, kv_valid, kv_q_ranges, scale,
        *q.stride(), *k.stride(), *v.stride(), *do.stride(), *dk.stride(),
        q_ranges.stride(0), q_ranges.stride(1),
        *kv_valid.stride(),
        kv_q_ranges.stride(0), kv_q_ranges.stride(1),
        H, HKV, M, N,
        GROUPS=groups, HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn,
        num_warps=_BWD_DKDV_WARPS, num_stages=1,
    )  # fmt: skip
    _seg_bwd_dq[(triton.cdiv(M, bm), B * H)](
        q, k, v, do, dq, lse, delta, q_ranges, kv_valid, scale,
        *q.stride(), *k.stride(), *v.stride(), *do.stride(), *dq.stride(),
        q_ranges.stride(0), q_ranges.stride(1),
        *kv_valid.stride(),
        H, M, N,
        GROUPS=groups, HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn,
        num_warps=_BWD_DQ_WARPS, num_stages=1,
    )  # fmt: skip
    return dq, dk, dv


@_seg_bwd_op.register_fake
def _(do, q, k, v, out, lse, q_ranges, kv_valid, kv_q_ranges, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


class _FlashD512Segments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_ranges, kv_valid, kv_q_ranges, scale):
        out, lse = _seg_fwd_op(q, k, v, q_ranges, kv_valid, scale)
        ctx.save_for_backward(q, k, v, out, lse, q_ranges, kv_valid, kv_q_ranges)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse, q_ranges, kv_valid, kv_q_ranges = ctx.saved_tensors
        dq, dk, dv = _seg_bwd_op(
            do, q, k, v, out, lse, q_ranges, kv_valid, kv_q_ranges, ctx.scale
        )
        return dq, dk, dv, None, None, None, None


def flash_d512_segments(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: SegmentAttentionMeta,
    scale: float | None = None,
) -> torch.Tensor:
    """Segment-masked attention. ``q [B, Hq, M, D]``, ``k``/``v`` ``[B, Hkv, N, D]``.

    Returns ``[B, Hq, M, D]`` (a transposed view of ``[B, M, Hq, D]`` storage)."""
    B, H, M, D = q.shape
    sixteen_bit = q.dtype in (torch.bfloat16, torch.float16)
    if not sixteen_bit or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("flash_d512_segments takes bf16/fp16 q, k, v of one dtype")
    if k.shape[:1] != (B,) or k.shape != v.shape or H % k.shape[1]:
        raise ValueError("k/v must be [B, Hkv, N, D] with Hq divisible by Hkv")
    if meta.q_ranges.shape != (B, M, 4) or meta.kv_valid.shape != (B, k.shape[2]):
        raise ValueError("segment metadata does not match the attention shapes")
    scale = D**-0.5 if scale is None else float(scale)
    return _FlashD512Segments.apply(
        q, k, v, meta.q_ranges, meta.kv_valid, meta.kv_q_ranges, scale
    )
