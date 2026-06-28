"""Core attention for DeepSeek-V4 CSA/HCA layers (sparse-MLA), fwd + bwd.

These layers call ``eager_attention_forward`` with
``kv = cat([sliding_kv (S), compressed_kv (T)], dim=2)`` and
``mask = [sliding_window_causal (S) | block_bias (S, T)]``. ``block_bias`` carries the
per-query causality + Lightning-Indexer top-k selection over the T compressed entries
and is a constant w.r.t. attention (top-k / argmax are non-differentiable).

This kernel does one fused online softmax over: the windowed sliding keys (implicit
mask, O(S*window)), the T compressed keys (dense additive ``block_bias``), and the
per-head sink — never materializing the [B,H,S,S+T] scores eager builds, and keeping
the single MQA KV head in SRAM. fp32 online-softmax accumulation.

Shared-KV MQA: ``kv_slide`` is used as both K and V, ``kv_comp`` likewise; their grads
are dk+dv summed. ``block_bias`` is dense [B,S,T] (smaller than eager's full mask but
still O(S*T) for CSA where T~S/4 — exploiting the top-k to gather only index_topk keys
per query is a further optimization, not done here).
"""

import torch
import triton
import triton.language as tl

from .attention import _CONFIGS, _prune_smem


@triton.autotune(
    configs=_CONFIGS,
    key=["H", "EL"],
    prune_configs_by={"early_config_prune": _prune_smem(1, 2)},
)
@triton.jit
def _csa_fwd_kernel(
    Q,
    KS,
    KC,
    BB,
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
    sbb,
    sbm,
    sbt,
    sob,
    soh,
    som,
    sod,
    H,
    S,
    T,
    WINDOW,
    EL,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_m[:, None] < S,
        other=0.0,
    )
    sink = tl.load(sinks + h).to(tl.float32)
    m_i = tl.zeros([BLOCK_M], tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    lo = (tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N) * BLOCK_N
    hi = tl.minimum(S, pid_m * BLOCK_M + BLOCK_M)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        k = tl.load(
            KS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=nmask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        valid = (
            (offs_n[None, :] <= offs_m[:, None])
            & (offs_m[:, None] - offs_n[None, :] < WINDOW)
            & nmask[None, :]
        )
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            KS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=nmask[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(ACC), v.to(ACC), input_precision=PREC).to(tl.float32)
        m_i = m_new

    for start_t in range(0, T, BLOCK_N):
        offs_t = start_t + tl.arange(0, BLOCK_N)
        tmask = offs_t < T
        kc = tl.load(
            KC + b * scb + offs_t[:, None] * scn + offs_d[None, :] * scd,
            mask=tmask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(kc), input_precision=PREC) * scale
        bb = tl.load(
            BB + b * sbb + offs_m[:, None] * sbm + offs_t[None, :] * sbt,
            mask=(offs_m[:, None] < S) & tmask[None, :],
            other=float("-inf"),
        )
        qk = tl.where(tmask[None, :], qk + bb, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        vc = tl.load(
            KC + b * scb + offs_t[:, None] * scn + offs_d[None, :] * scd,
            mask=tmask[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(ACC), vc.to(ACC), input_precision=PREC).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < S,
    )
    tl.store(L + pid_bh * S + offs_m, m_i + tl.log(l_i), mask=offs_m < S)


def _csa_fwd(q, ks, kc, bb, sinks, scale, window):
    B, H, S, D = q.shape
    T = kc.shape[1]
    out = torch.empty(B, H, S, D, device=q.device, dtype=q.dtype)
    L = torch.empty(B * H, S, device=q.device, dtype=torch.float32)
    ACC = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float32
    PREC = "ieee" if q.element_size() == 4 else "tf32"
    grid = lambda m: (triton.cdiv(S, m["BLOCK_M"]), B * H)
    _csa_fwd_kernel[grid](
        q,
        ks,
        kc,
        bb,
        sinks,
        out,
        L,
        scale,
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
        bb.stride(0),
        bb.stride(1),
        bb.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        H,
        S,
        T,
        window,
        q.element_size(),
        D=D,
        ACC=ACC,
        PREC=PREC,
    )
    return out, L


@triton.autotune(
    configs=_CONFIGS,
    key=["H", "EL"],
    reset_to_zero=["DSINK"],
    prune_configs_by={"early_config_prune": _prune_smem(2, 2)},
)
@triton.jit
def _csa_bwd_dq_kernel(
    Q,
    KS,
    KC,
    BB,
    sinks,
    DO,
    L,
    Delta,
    DQ,
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
    sbb,
    sbm,
    sbt,
    H,
    S,
    T,
    WINDOW,
    EL,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < S
    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=mmask[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=mmask[:, None],
        other=0.0,
    )
    l_i = tl.load(L + pid_bh * S + offs_m, mask=mmask, other=0.0)
    delta = tl.load(Delta + pid_bh * S + offs_m, mask=mmask, other=0.0)
    sink = tl.load(sinks + h).to(tl.float32)
    dq = tl.zeros([BLOCK_M, D], tl.float32)

    lo = (tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N) * BLOCK_N
    hi = tl.minimum(S, pid_m * BLOCK_M + BLOCK_M)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        k = tl.load(
            KS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=nmask[:, None],
            other=0.0,
        )
        v = k
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        valid = (
            (offs_n[None, :] <= offs_m[:, None])
            & (offs_m[:, None] - offs_n[None, :] < WINDOW)
            & nmask[None, :]
        )
        p = tl.where(valid, tl.exp(qk - l_i[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=PREC)
        ds = (p * (dp - delta[:, None]) * scale).to(ACC)
        dq += tl.dot(ds, k.to(ACC), input_precision=PREC).to(tl.float32)

    for start_t in range(0, T, BLOCK_N):
        offs_t = start_t + tl.arange(0, BLOCK_N)
        tmask = offs_t < T
        kc = tl.load(
            KC + b * scb + offs_t[:, None] * scn + offs_d[None, :] * scd,
            mask=tmask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(kc), input_precision=PREC) * scale
        bb = tl.load(
            BB + b * sbb + offs_m[:, None] * sbm + offs_t[None, :] * sbt,
            mask=mmask[:, None] & tmask[None, :],
            other=float("-inf"),
        )
        p = tl.where(tmask[None, :], tl.exp(qk + bb - l_i[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(kc), input_precision=PREC)
        ds = (p * (dp - delta[:, None]) * scale).to(ACC)
        dq += tl.dot(ds, kc.to(ACC), input_precision=PREC).to(tl.float32)

    p_sink = tl.exp(sink - l_i)
    tl.atomic_add(DSINK + h, tl.sum(tl.where(mmask, -p_sink * delta, 0.0)))
    tl.store(
        DQ + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        dq.to(DQ.dtype.element_ty),
        mask=mmask[:, None],
    )


@triton.autotune(
    configs=_CONFIGS,
    key=["H", "EL"],
    reset_to_zero=["DKS", "DKC"],
    prune_configs_by={"early_config_prune": _prune_smem(2, 2)},
)
@triton.jit
def _csa_bwd_dkv_kernel(
    Q,
    KS,
    KC,
    BB,
    DO,
    L,
    Delta,
    DKS,
    DKC,
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
    sbb,
    sbm,
    sbt,
    H,
    S,
    T,
    WINDOW,
    EL,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
    PREC: tl.constexpr,
):
    # pid_n indexes a key block over the combined [sliding (S) ; compressed (T)] axis.
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_d = tl.arange(0, D)
    NSLIDE_BLK = tl.cdiv(S, BLOCK_N)
    is_comp = pid_n >= NSLIDE_BLK

    if is_comp:
        start_t = (pid_n - NSLIDE_BLK) * BLOCK_N
        offs_n = start_t + tl.arange(0, BLOCK_N)
        nmask = offs_n < T
        kc = tl.load(
            KC + b * scb + offs_n[:, None] * scn + offs_d[None, :] * scd,
            mask=nmask[:, None],
            other=0.0,
        )
        dk = tl.zeros([BLOCK_N, D], tl.float32)
        dv = tl.zeros([BLOCK_N, D], tl.float32)
        for start_m in range(0, S, BLOCK_M):  # any query may attend any compressed key
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mmask = offs_m < S
            q = tl.load(
                Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                mask=mmask[:, None],
                other=0.0,
            )
            do = tl.load(
                DO + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                mask=mmask[:, None],
                other=0.0,
            )
            l_i = tl.load(L + pid_bh * S + offs_m, mask=mmask, other=0.0)
            delta = tl.load(Delta + pid_bh * S + offs_m, mask=mmask, other=0.0)
            qk = tl.dot(q, tl.trans(kc), input_precision=PREC) * scale
            bb = tl.load(
                BB + b * sbb + offs_m[:, None] * sbm + offs_n[None, :] * sbt,
                mask=mmask[:, None] & nmask[None, :],
                other=float("-inf"),
            )
            p = tl.where(
                mmask[:, None] & nmask[None, :], tl.exp(qk + bb - l_i[:, None]), 0.0
            )
            dp = tl.dot(do, tl.trans(kc), input_precision=PREC)
            ds = (p * (dp - delta[:, None]) * scale).to(ACC)
            dk += tl.dot(tl.trans(ds), q.to(ACC), input_precision=PREC).to(tl.float32)
            dv += tl.dot(tl.trans(p.to(ACC)), do.to(ACC), input_precision=PREC).to(
                tl.float32
            )
        ptr = DKC + b * scb + offs_n[:, None] * scn + offs_d[None, :] * scd
        tl.atomic_add(ptr, (dk + dv), mask=nmask[:, None])
    else:
        start_n = pid_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        ks = tl.load(
            KS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=nmask[:, None],
            other=0.0,
        )
        dk = tl.zeros([BLOCK_N, D], tl.float32)
        dv = tl.zeros([BLOCK_N, D], tl.float32)
        lo = (start_n // BLOCK_M) * BLOCK_M
        hi = tl.minimum(S, start_n + BLOCK_N + WINDOW - 1)
        for start_m in range(lo, hi, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mmask = offs_m < S
            q = tl.load(
                Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                mask=mmask[:, None],
                other=0.0,
            )
            do = tl.load(
                DO + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                mask=mmask[:, None],
                other=0.0,
            )
            l_i = tl.load(L + pid_bh * S + offs_m, mask=mmask, other=0.0)
            delta = tl.load(Delta + pid_bh * S + offs_m, mask=mmask, other=0.0)
            qk = tl.dot(q, tl.trans(ks), input_precision=PREC) * scale
            valid = (
                (offs_n[None, :] <= offs_m[:, None])
                & (offs_m[:, None] - offs_n[None, :] < WINDOW)
                & nmask[None, :]
            )
            p = tl.where(valid, tl.exp(qk - l_i[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(ks), input_precision=PREC)
            ds = (p * (dp - delta[:, None]) * scale).to(ACC)
            dk += tl.dot(tl.trans(ds), q.to(ACC), input_precision=PREC).to(tl.float32)
            dv += tl.dot(tl.trans(p.to(ACC)), do.to(ACC), input_precision=PREC).to(
                tl.float32
            )
        ptr = DKS + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd
        tl.atomic_add(ptr, (dk + dv), mask=nmask[:, None])


def _csa_bwd(q, ks, kc, bb, sinks, out, do, L, scale, window):
    B, H, S, D = q.shape
    T = kc.shape[1]
    do = do.contiguous()
    delta = (do.to(torch.float32) * out.to(torch.float32)).sum(-1).reshape(B * H, S)
    dq = torch.empty_like(q)
    dks = torch.zeros_like(ks, dtype=torch.float32)
    dkc = torch.zeros_like(kc, dtype=torch.float32)
    dsink = torch.zeros_like(sinks, dtype=torch.float32)
    ACC = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float32
    PREC = "ieee" if q.element_size() == 4 else "tf32"
    EL = q.element_size()
    qs = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    ss = (ks.stride(0), ks.stride(1), ks.stride(2))
    cs = (kc.stride(0), kc.stride(1), kc.stride(2))
    bs = (bb.stride(0), bb.stride(1), bb.stride(2))
    _csa_bwd_dq_kernel[lambda m: (triton.cdiv(S, m["BLOCK_M"]), B * H)](
        q,
        ks,
        kc,
        bb,
        sinks,
        do,
        L,
        delta,
        dq,
        dsink,
        scale,
        *qs,
        *ss,
        *cs,
        *bs,
        H,
        S,
        T,
        window,
        EL,
        D=D,
        ACC=ACC,
        PREC=PREC,
    )
    _csa_bwd_dkv_kernel[
        lambda m: (triton.cdiv(S, m["BLOCK_N"]) + triton.cdiv(T, m["BLOCK_N"]), B * H)
    ](
        q,
        ks,
        kc,
        bb,
        do,
        L,
        delta,
        dks,
        dkc,
        scale,
        *qs,
        *ss,
        *cs,
        *bs,
        H,
        S,
        T,
        window,
        EL,
        D=D,
        ACC=ACC,
        PREC=PREC,
    )
    return dq, dks.to(ks.dtype), dkc.to(kc.dtype), dsink.to(sinks.dtype)


class _CSAAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, ks, kc, bb, sinks, scale, window):
        out, L = _csa_fwd(q, ks, kc, bb, sinks, scale, window)
        ctx.save_for_backward(q, ks, kc, bb, sinks, out, L)
        ctx.scale, ctx.window = scale, window
        return out

    @staticmethod
    def backward(ctx, do):
        q, ks, kc, bb, sinks, out, L = ctx.saved_tensors
        dq, dks, dkc, dsink = _csa_bwd(
            q, ks, kc, bb, sinks, out, do, L, ctx.scale, ctx.window
        )
        return dq, dks, dkc, None, dsink, None, None


def csa_attn(q, kv_slide, kv_comp, block_bias, sinks, scale=None, sliding_window=128):
    """Core attention for CSA/HCA layers.
    q: [B,H,S,D]; kv_slide: [B,1,S,D] (K==V); kv_comp: [B,1,T,D] (K==V);
    block_bias: [B,1,S,T] additive; sinks: [H]. Returns [B,H,S,D].

    NOTE: the dk/dv backward over the compressed block uses ``BLOCK_N=16`` for the grid's
    block count (matches the smallest autotune tile); the kernel masks any tail."""
    B, H, S, D = q.shape
    if scale is None:
        scale = D**-0.5
    # compressed KV can arrive fp32 (keep_in_fp32 compressor); match all to q (no mixed dtypes).
    dt = q.dtype
    kv_slide = kv_slide.to(dt) if kv_slide.dtype != dt else kv_slide
    kv_comp = kv_comp.to(dt) if kv_comp.dtype != dt else kv_comp
    sinks = sinks.to(dt) if sinks.dtype != dt else sinks
    ks = kv_slide[:, 0].contiguous()
    kc = kv_comp[:, 0].contiguous()
    bb = block_bias[:, 0].contiguous()
    return _CSAAttn.apply(
        q.contiguous(), ks, kc, bb, sinks.contiguous(), scale, sliding_window
    )
