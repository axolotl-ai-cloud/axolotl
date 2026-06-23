"""Top-k GATHER sparse-MLA for DeepSeek-V4 CSA/HCA core attention (fwd + bwd).

Head-batched, MMA-efficient. An earlier per-query-position layout lost tensor cores
(each position has a distinct top-k set, so gathered keys can't be shared across a
position-block -> scalar dots). DeepSeek's TileLang sparse-MLA avoids this by putting the
**H query heads on the MMA M-axis**: in MLA/MQA all H heads at one position share the same
KV and the same per-position top-k, so a gathered tile of BN keys feeds a full
``q[BH,D] @ Kᵀ[D,BN]`` MMA. We do the same.

One program = (query position s, head-tile of BH heads, batch b). It attends to the
sliding window (≤ ``window`` causal keys, MMA) and the position's ``K`` gathered top-k
compressed keys (gathered in BN-chunks, MMA), plus the per-head sink. Turns the compressed
term from O(S*T) (dense) to O(S*K) while keeping tensor cores busy.

``topk_idx`` is [B, S, K] int32 (-1 = invalid, as ``DeepseekV4Indexer`` returns); kv_comp
is shared K==V (grad scatter-added). int64 offsets so it's correct past 64k context.
"""

import torch
import triton
import triton.language as tl

from .attention import _max_m, _smem_limit

# BH (head tile, MMA M-axis) must be >= 16 (tensor-core min M). bwd also holds do +
# transposes, so it needs smaller tiles; SMEM-pruned per GPU (sm_120 ~99KB vs H100/B200 ~228KB).
_GCFGS = [
    triton.Config({"BH": bh, "BN": bn}, num_warps=w, num_stages=s)
    for bh in (16, 32, 64)
    for bn in (16, 32, 64)
    for w in (4, 8)
    for s in (1, 2)
    if not (bn == 16 and bh >= 64)
]  # N=16 wgmma at large M crashes on Hopper/Blackwell


def _gprune(n_tiles):
    def prune(configs, nargs, **kwargs):
        D = kwargs["D"]
        EL = nargs["EL"]
        dev = torch.cuda.current_device()
        budget = int(_smem_limit(dev) * 0.9)
        max_m = _max_m(dev)  # Blackwell tmem caps the [BH,512] acc at BH<=32
        kept = [
            c
            for c in configs
            if (n_tiles * c.kwargs["BH"] + n_tiles * c.num_stages * c.kwargs["BN"])
            * D
            * EL
            <= budget
            and c.kwargs["BH"] <= max_m
        ]
        return kept or [
            min(configs, key=lambda c: c.kwargs["BH"] * c.kwargs["BN"] * c.num_stages)
        ]

    return prune


@triton.autotune(
    configs=_GCFGS,
    key=["H", "K", "EL"],
    prune_configs_by={"early_config_prune": _gprune(1)},
)
@triton.jit
def _gather_fwd_kernel(
    Q,
    KS,
    KC,
    IDX,
    sinks,
    Out,
    L,
    scale,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skn,
    skd,
    scb,
    scn,
    scd,
    sib,
    sim,
    sik,
    H,
    S,
    K,
    WINDOW,
    EL,
    D: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    ACC: tl.constexpr,
    PREC: tl.constexpr,
):
    NHB = tl.cdiv(H, BH)
    s = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    b = (pid // NHB).to(tl.int64)
    hb = pid % NHB
    offs_h = (hb * BH + tl.arange(0, BH)).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    hmask = offs_h < H

    q = tl.load(
        Q + b * sqb + offs_h[:, None] * sqh + s * sqm + offs_d[None, :] * sqd,
        mask=hmask[:, None],
        other=0.0,
    )
    sink = tl.load(sinks + offs_h, mask=hmask, other=0.0).to(tl.float32)
    m_i = sink
    l_i = tl.zeros([BH], tl.float32) + 1.0
    acc = tl.zeros([BH, D], tl.float32)

    # causal sliding window for position s: j in (s-window, s]
    lo = (tl.maximum(0, s - WINDOW + 1) // BN) * BN
    hi = s + 1
    for start_n in range(lo, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        nmask = (offs_n <= s) & (s - offs_n < WINDOW) & (offs_n < S)
        k = tl.load(
            KS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=nmask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        qk = tl.where(nmask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(ACC), k.to(ACC), input_precision=PREC
        ).to(tl.float32)
        m_i = m_new

    for start_k in range(0, K, BN):
        kidx = start_k + tl.arange(0, BN)
        idx = tl.load(IDX + b * sib + s * sim + kidx * sik, mask=kidx < K, other=-1)
        gv = idx >= 0
        idxs = tl.where(gv, idx, 0).to(tl.int64)
        kc = tl.load(
            KC + b * scb + idxs[:, None] * scn + offs_d[None, :] * scd,
            mask=gv[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(kc), input_precision=PREC) * scale
        qk = tl.where(gv[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(ACC), kc.to(ACC), input_precision=PREC
        ).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        Out + b * sqb + offs_h[:, None] * sqh + s * sqm + offs_d[None, :] * sqd,
        acc.to(Out.dtype.element_ty),
        mask=hmask[:, None],
    )
    tl.store(L + b * (H * S) + offs_h * S + s, m_i + tl.log(l_i), mask=hmask)


@triton.autotune(
    configs=_GCFGS,
    key=["H", "K", "EL"],
    reset_to_zero=["DKS", "DKC", "DSINK"],
    prune_configs_by={"early_config_prune": _gprune(2)},
)
@triton.jit
def _gather_bwd_kernel(
    Q,
    KS,
    KC,
    IDX,
    sinks,
    DO,
    L,
    Delta,
    DQ,
    DKS,
    DKC,
    DSINK,
    scale,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skn,
    skd,
    scb,
    scn,
    scd,
    sib,
    sim,
    sik,
    H,
    S,
    K,
    WINDOW,
    EL,
    D: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    ACC: tl.constexpr,
    PREC: tl.constexpr,
):
    NHB = tl.cdiv(H, BH)
    s = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    b = (pid // NHB).to(tl.int64)
    hb = pid % NHB
    offs_h = (hb * BH + tl.arange(0, BH)).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    hmask = offs_h < H

    q = tl.load(
        Q + b * sqb + offs_h[:, None] * sqh + s * sqm + offs_d[None, :] * sqd,
        mask=hmask[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + b * sqb + offs_h[:, None] * sqh + s * sqm + offs_d[None, :] * sqd,
        mask=hmask[:, None],
        other=0.0,
    )
    l_i = tl.load(L + b * (H * S) + offs_h * S + s, mask=hmask, other=0.0)
    delta = tl.load(Delta + b * (H * S) + offs_h * S + s, mask=hmask, other=0.0)
    sink = tl.load(sinks + offs_h, mask=hmask, other=0.0).to(tl.float32)
    dq = tl.zeros([BH, D], tl.float32)

    lo = (tl.maximum(0, s - WINDOW + 1) // BN) * BN
    hi = s + 1
    for start_n in range(lo, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        nmask = (offs_n <= s) & (s - offs_n < WINDOW) & (offs_n < S)
        k = tl.load(
            KS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=nmask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        p = tl.where(nmask[None, :], tl.exp(qk - l_i[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(k), input_precision=PREC)
        ds = (p * (dp - delta[:, None]) * scale).to(ACC)
        dq += tl.dot(ds, k.to(ACC), input_precision=PREC).to(tl.float32)
        dkv = tl.dot(tl.trans(ds), q.to(ACC), input_precision=PREC).to(
            tl.float32
        ) + tl.dot(tl.trans(p.to(ACC)), do.to(ACC), input_precision=PREC).to(tl.float32)
        tl.atomic_add(
            DKS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
            dkv,
            mask=nmask[:, None],
        )

    for start_k in range(0, K, BN):
        kidx = start_k + tl.arange(0, BN)
        idx = tl.load(IDX + b * sib + s * sim + kidx * sik, mask=kidx < K, other=-1)
        gv = idx >= 0
        idxs = tl.where(gv, idx, 0).to(tl.int64)
        kc = tl.load(
            KC + b * scb + idxs[:, None] * scn + offs_d[None, :] * scd,
            mask=gv[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(kc), input_precision=PREC) * scale
        p = tl.where(gv[None, :], tl.exp(qk - l_i[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(kc), input_precision=PREC)
        ds = (p * (dp - delta[:, None]) * scale).to(ACC)
        dq += tl.dot(ds, kc.to(ACC), input_precision=PREC).to(tl.float32)
        dkv = tl.dot(tl.trans(ds), q.to(ACC), input_precision=PREC).to(
            tl.float32
        ) + tl.dot(tl.trans(p.to(ACC)), do.to(ACC), input_precision=PREC).to(tl.float32)
        tl.atomic_add(
            DKC + b * scb + idxs[:, None] * scn + offs_d[None, :] * scd,
            dkv,
            mask=gv[:, None],
        )

    p_sink = tl.exp(sink - l_i)
    # mask the atomic itself: zeroing the value isn't enough, offs_h is out-of-bounds for
    # padded heads (H % BH != 0).
    tl.atomic_add(DSINK + offs_h, tl.where(hmask, -p_sink * delta, 0.0), mask=hmask)
    tl.store(
        DQ + b * sqb + offs_h[:, None] * sqh + s * sqm + offs_d[None, :] * sqd,
        dq.to(DQ.dtype.element_ty),
        mask=hmask[:, None],
    )


class _CSATopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, ks, kc, idx, sinks, scale, window):
        B, H, S, D = q.shape
        K = idx.shape[2]
        out = torch.empty(B, H, S, D, device=q.device, dtype=q.dtype)
        L = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        ACC = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float32
        PREC = "ieee" if q.element_size() == 4 else "tf32"
        args = (
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            ks.stride(0),
            ks.stride(1),
            ks.stride(2),
            kc.stride(0),
            kc.stride(1),
            kc.stride(2),
            idx.stride(0),
            idx.stride(1),
            idx.stride(2),
        )
        grid = lambda m: (S, B * triton.cdiv(H, m["BH"]))
        _gather_fwd_kernel[grid](
            q,
            ks,
            kc,
            idx,
            sinks,
            out,
            L,
            scale,
            *args,
            H,
            S,
            K,
            window,
            q.element_size(),
            D=D,
            ACC=ACC,
            PREC=PREC,
        )
        ctx.save_for_backward(q, ks, kc, idx, sinks, out, L)
        ctx.scale, ctx.window = scale, window
        return out

    @staticmethod
    def backward(ctx, do):
        q, ks, kc, idx, sinks, out, L = ctx.saved_tensors
        B, H, S, D = q.shape
        K = idx.shape[2]
        do = do.contiguous()
        delta = (do.to(torch.float32) * out.to(torch.float32)).sum(-1)  # [B,H,S]
        dq = torch.empty_like(q)
        dks = torch.zeros_like(ks, dtype=torch.float32)
        dkc = torch.zeros_like(kc, dtype=torch.float32)
        dsink = torch.zeros_like(sinks, dtype=torch.float32)
        ACC = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float32
        PREC = "ieee" if q.element_size() == 4 else "tf32"
        args = (
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            ks.stride(0),
            ks.stride(1),
            ks.stride(2),
            kc.stride(0),
            kc.stride(1),
            kc.stride(2),
            idx.stride(0),
            idx.stride(1),
            idx.stride(2),
        )
        grid = lambda m: (S, B * triton.cdiv(H, m["BH"]))
        _gather_bwd_kernel[grid](
            q,
            ks,
            kc,
            idx,
            sinks,
            do,
            L,
            delta,
            dq,
            dks,
            dkc,
            dsink,
            ctx.scale,
            *args,
            H,
            S,
            K,
            ctx.window,
            q.element_size(),
            D=D,
            ACC=ACC,
            PREC=PREC,
        )
        return (
            dq,
            dks.to(ks.dtype),
            dkc.to(kc.dtype),
            None,
            dsink.to(sinks.dtype),
            None,
            None,
        )


def csa_attn_topk(
    q, kv_slide, kv_comp, topk_idx, sinks, scale=None, sliding_window=128
):
    """Top-k gather Compressed Sparse Attention (head-batched, MMA, linear in compressed dim).
    q: [B,H,S,D]; kv_slide/kv_comp: [B,1,*,D] (K==V); topk_idx: [B,S,K] int32 (-1=invalid);
    sinks: [H]. Returns [B,H,S,D]."""
    B, H, S, D = q.shape
    if scale is None:
        scale = D**-0.5
    ks = kv_slide[:, 0].contiguous()
    kc = kv_comp[:, 0].contiguous()
    idx = topk_idx.contiguous().to(torch.int32)
    return _CSATopK.apply(
        q.contiguous(), ks, kc, idx, sinks.contiguous(), scale, sliding_window
    )
