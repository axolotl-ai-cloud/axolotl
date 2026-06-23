"""Fused Triton mHC mixer for DeepSeek-V4 HyperConnection (fwd + bwd).

``DeepseekV4HyperConnection`` runs twice per layer. Its FLOPs are tiny but the
forward/backward are dominated by ~40 tiny kernel launches from the 20-iteration
Sinkhorn-Knopp loop over [B,S,4,4] matrices. torch.compile fuses the launches but
mis-compiles ``collapsed`` to NaN on torch 2.11, so we fuse the
sigmoid/softmax/Sinkhorn "mixer" (mixes -> pre, post, comb) into a single Triton
kernel pair instead. The rmsnorm + fn-linear and the ``collapsed`` reduction stay in
eager torch (correct, memory-bound, not the launch bottleneck).

Sinkhorn (matches reference): comb = softmax(cl, -1) + eps; colnorm; then
(iters-1)x [rownorm, colnorm]. Backward recomputes the forward, stores each
normalize's input, and reverse-modes through them (normalize backward:
dx = (dy - sum(dy*y, axis)) / (sum(x,axis)+eps)).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": bm}, num_warps=w) for bm in (8, 16, 32, 64) for w in (2, 4)
    ],
    key=["M"],
)
@triton.jit
def _mhc_fwd_kernel(
    MIX,
    SCALE,
    BASE,
    PRE,
    POST,
    COMB,
    STATES,
    M,
    HC: tl.constexpr,
    MIX_N: tl.constexpr,
    ITERS: tl.constexpr,
    NSTATES: tl.constexpr,
    EPS: tl.constexpr,
    POSTMULT: tl.constexpr,
    SAVE: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BM + tl.arange(0, BM)
    mask = offs < M
    j = tl.arange(0, HC)
    jj = tl.arange(0, HC * HC)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)

    pre_w = tl.load(
        MIX + offs[:, None] * MIX_N + j[None, :], mask=mask[:, None], other=0.0
    )
    post_w = tl.load(
        MIX + offs[:, None] * MIX_N + (HC + j)[None, :], mask=mask[:, None], other=0.0
    )
    comb_w = tl.load(
        MIX + offs[:, None] * MIX_N + (2 * HC + jj)[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    pre_b = tl.load(BASE + j)
    post_b = tl.load(BASE + HC + j)
    comb_b = tl.load(BASE + 2 * HC + jj)

    pre = tl.sigmoid(pre_w * s0 + pre_b[None, :]) + EPS
    post = POSTMULT * tl.sigmoid(post_w * s1 + post_b[None, :])
    cl = tl.reshape(comb_w * s2 + comb_b[None, :], (BM, HC, HC))

    ex = tl.exp(cl - tl.max(cl, axis=2, keep_dims=True))
    comb = ex / tl.sum(ex, axis=2, keep_dims=True) + EPS

    # Store each normalize's INPUT to scratch so backward can reverse-mode the Sinkhorn.
    t = 0
    if SAVE:
        tl.store(
            STATES + offs[:, None] * (NSTATES * HC * HC) + t * (HC * HC) + jj[None, :],
            tl.reshape(comb, (BM, HC * HC)),
            mask=mask[:, None],
        )
    t += 1
    comb = comb / (tl.sum(comb, axis=1, keep_dims=True) + EPS)  # colnorm
    for _ in range(ITERS - 1):
        if SAVE:
            tl.store(
                STATES
                + offs[:, None] * (NSTATES * HC * HC)
                + t * (HC * HC)
                + jj[None, :],
                tl.reshape(comb, (BM, HC * HC)),
                mask=mask[:, None],
            )
        t += 1
        comb = comb / (tl.sum(comb, axis=2, keep_dims=True) + EPS)  # rownorm
        if SAVE:
            tl.store(
                STATES
                + offs[:, None] * (NSTATES * HC * HC)
                + t * (HC * HC)
                + jj[None, :],
                tl.reshape(comb, (BM, HC * HC)),
                mask=mask[:, None],
            )
        t += 1
        comb = comb / (tl.sum(comb, axis=1, keep_dims=True) + EPS)  # colnorm

    tl.store(PRE + offs[:, None] * HC + j[None, :], pre, mask=mask[:, None])
    tl.store(POST + offs[:, None] * HC + j[None, :], post, mask=mask[:, None])
    tl.store(
        COMB + offs[:, None] * (HC * HC) + jj[None, :],
        tl.reshape(comb, (BM, HC * HC)),
        mask=mask[:, None],
    )


@triton.autotune(
    configs=[
        triton.Config({"BM": bm}, num_warps=w) for bm in (8, 16, 32, 64) for w in (2, 4)
    ],
    key=["M"],
    reset_to_zero=["DSCALE", "DBASE"],
)
@triton.jit
def _mhc_bwd_kernel(
    MIX,
    SCALE,
    BASE,
    STATES,
    DPRE,
    DPOST,
    DCOMB,
    DMIX,
    DSCALE,
    DBASE,
    M,
    HC: tl.constexpr,
    MIX_N: tl.constexpr,
    ITERS: tl.constexpr,
    NSTATES: tl.constexpr,
    EPS: tl.constexpr,
    POSTMULT: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BM + tl.arange(0, BM)
    mask = offs < M
    j = tl.arange(0, HC)
    jj = tl.arange(0, HC * HC)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)

    pre_w = tl.load(
        MIX + offs[:, None] * MIX_N + j[None, :], mask=mask[:, None], other=0.0
    )
    post_w = tl.load(
        MIX + offs[:, None] * MIX_N + (HC + j)[None, :], mask=mask[:, None], other=0.0
    )
    comb_w = tl.load(
        MIX + offs[:, None] * MIX_N + (2 * HC + jj)[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    pre_b = tl.load(BASE + j)
    post_b = tl.load(BASE + HC + j)
    _comb_b = tl.load(BASE + 2 * HC + jj)

    pre_sig = tl.sigmoid(pre_w * s0 + pre_b[None, :])
    post_sig = tl.sigmoid(post_w * s1 + post_b[None, :])
    # sm = softmax(cl) = STATES[0] - eps (state 0 saved before colnorm)
    sm = (
        tl.reshape(
            tl.load(
                STATES
                + offs[:, None] * (NSTATES * HC * HC)
                + 0 * (HC * HC)
                + jj[None, :],
                mask=mask[:, None],
                other=0.0,
            ),
            (BM, HC, HC),
        )
        - EPS
    )

    dy = tl.reshape(
        tl.load(
            DCOMB + offs[:, None] * (HC * HC) + jj[None, :],
            mask=mask[:, None],
            other=0.0,
        ),
        (BM, HC, HC),
    )
    # reverse the Sinkhorn normalizes. Forward: colnorm(idx0), then (iters-1)x [rownorm,
    # colnorm]; undo last-first (colnorm=axis1, rownorm=axis2).
    base_st = offs[:, None] * (NSTATES * HC * HC) + jj[None, :]
    t = 2 * (ITERS - 1)  # last (col) state index
    xc = tl.reshape(
        tl.load(STATES + base_st + t * (HC * HC), mask=mask[:, None], other=0.0),
        (BM, HC, HC),
    )
    Sc = tl.sum(xc, axis=1, keep_dims=True) + EPS
    dy = (dy - tl.sum(dy * (xc / Sc), axis=1, keep_dims=True)) / Sc
    t -= 1
    for _ in range(ITERS - 1):
        xr = tl.reshape(
            tl.load(STATES + base_st + t * (HC * HC), mask=mask[:, None], other=0.0),
            (BM, HC, HC),
        )
        Sr = tl.sum(xr, axis=2, keep_dims=True) + EPS
        dy = (dy - tl.sum(dy * (xr / Sr), axis=2, keep_dims=True)) / Sr
        t -= 1
        xc2 = tl.reshape(
            tl.load(STATES + base_st + t * (HC * HC), mask=mask[:, None], other=0.0),
            (BM, HC, HC),
        )
        Sc2 = tl.sum(xc2, axis=1, keep_dims=True) + EPS
        dy = (dy - tl.sum(dy * (xc2 / Sc2), axis=1, keep_dims=True)) / Sc2
        t -= 1
    # dy = dA = d(sm); softmax backward over axis=2
    dcl = sm * (dy - tl.sum(dy * sm, axis=2, keep_dims=True))
    dcl_flat = tl.reshape(dcl, (BM, HC * HC))

    dpre = tl.load(
        DPRE + offs[:, None] * HC + j[None, :], mask=mask[:, None], other=0.0
    )
    dpost = tl.load(
        DPOST + offs[:, None] * HC + j[None, :], mask=mask[:, None], other=0.0
    )
    dpre_z = dpre * pre_sig * (1.0 - pre_sig)
    dpost_z = dpost * POSTMULT * post_sig * (1.0 - post_sig)

    d_pre_w = dpre_z * s0
    d_post_w = dpost_z * s1
    d_comb_w = dcl_flat * s2
    tl.store(DMIX + offs[:, None] * MIX_N + j[None, :], d_pre_w, mask=mask[:, None])
    tl.store(
        DMIX + offs[:, None] * MIX_N + (HC + j)[None, :], d_post_w, mask=mask[:, None]
    )
    tl.store(
        DMIX + offs[:, None] * MIX_N + (2 * HC + jj)[None, :],
        d_comb_w,
        mask=mask[:, None],
    )

    m2 = mask[:, None]
    tl.atomic_add(DSCALE + 0, tl.sum(tl.where(m2, dpre_z * pre_w, 0.0)))
    tl.atomic_add(DSCALE + 1, tl.sum(tl.where(m2, dpost_z * post_w, 0.0)))
    tl.atomic_add(DSCALE + 2, tl.sum(tl.where(m2, dcl_flat * comb_w, 0.0)))
    tl.atomic_add(DBASE + j, tl.sum(tl.where(m2, dpre_z, 0.0), axis=0))
    tl.atomic_add(DBASE + HC + j, tl.sum(tl.where(m2, dpost_z, 0.0), axis=0))
    tl.atomic_add(DBASE + 2 * HC + jj, tl.sum(tl.where(m2, dcl_flat, 0.0), axis=0))


class _SinkhornMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, hc, iters, eps, post_mult):
        M, MIX_N = mixes.shape
        mixes = mixes.contiguous().float()
        scale = scale.contiguous().float()
        base = base.contiguous().float()
        nstates = 1 + 2 * (iters - 1)
        pre = torch.empty(M, hc, device=mixes.device, dtype=torch.float32)
        post = torch.empty(M, hc, device=mixes.device, dtype=torch.float32)
        comb = torch.empty(M, hc * hc, device=mixes.device, dtype=torch.float32)
        # Allocate state history if ANY trainable input needs grad (backward reads it for the
        # scale/base grads too, not just mixes). is_grad_enabled() is False in Function.forward.
        save = any(ctx.needs_input_grad)
        states = (
            torch.empty(M, nstates, hc * hc, device=mixes.device, dtype=torch.float32)
            if save
            else torch.empty(1, device=mixes.device, dtype=torch.float32)
        )
        grid = lambda m: (triton.cdiv(M, m["BM"]),)
        _mhc_fwd_kernel[grid](
            mixes,
            scale,
            base,
            pre,
            post,
            comb,
            states,
            M,
            HC=hc,
            MIX_N=MIX_N,
            ITERS=iters,
            NSTATES=nstates,
            EPS=eps,
            POSTMULT=post_mult,
            SAVE=save,
        )
        ctx.save_for_backward(mixes, scale, base, states)
        ctx.cfg = (hc, MIX_N, iters, nstates, eps, post_mult)
        return pre, post, comb

    @staticmethod
    def backward(ctx, dpre, dpost, dcomb):
        mixes, scale, base, states = ctx.saved_tensors
        hc, MIX_N, iters, nstates, eps, post_mult = ctx.cfg
        M = mixes.shape[0]
        dmix = torch.empty_like(mixes)
        dscale = torch.zeros_like(scale)
        dbase = torch.zeros_like(base)
        grid = lambda m: (triton.cdiv(M, m["BM"]),)
        _mhc_bwd_kernel[grid](
            mixes,
            scale,
            base,
            states,
            dpre.contiguous().float(),
            dpost.contiguous().float(),
            dcomb.contiguous().float(),
            dmix,
            dscale,
            dbase,
            M,
            HC=hc,
            MIX_N=MIX_N,
            ITERS=iters,
            NSTATES=nstates,
            EPS=eps,
            POSTMULT=post_mult,
        )
        return dmix, dscale, dbase, None, None, None, None


def sinkhorn_mix(mixes, scale, base, hc, iters, eps, post_mult=2.0):
    """mixes: [M, hc*(2+hc)]; scale: [3]; base: [hc*(2+hc)]. Returns (pre, post, comb)
    with pre/post [M, hc], comb [M, hc*hc]. Fused sigmoid/softmax/Sinkhorn (fp32)."""
    return _SinkhornMix.apply(mixes, scale, base, hc, iters, eps, post_mult)


# Fused RMSNorm + fn-linear. RMS scale r is per-row, so mixes = r * (streams @ fn^T): one
# pass over the 16384-wide streams covers both streams@fn^T and sum-of-squares (eager does
# ~3 passes + a 536MB fp32 flat).

_RL_CFGS = [
    triton.Config({"BK": bk}, num_warps=w, num_stages=s)
    for bk in (64, 128, 256)
    for w in (4, 8)
    for s in (2, 3)
]


@triton.autotune(configs=_RL_CFGS, key=["M", "K", "N"])
@triton.jit
def _rmsln_fwd_kernel(
    STREAMS,
    FN,
    MIXES,
    R,
    G,
    M,
    K,
    EPS,
    N: tl.constexpr,
    NP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mmask = offs_m < M
    n = tl.arange(0, NP)
    nmask = n < N
    acc = tl.zeros([BM, NP], tl.float32)
    ssq = tl.zeros([BM], tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        kmask = kk < K
        x = tl.load(
            STREAMS + offs_m[:, None] * K + kk[None, :],
            mask=mmask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        ssq += tl.sum(x * x, axis=1)
        w = tl.load(
            FN + n[:, None] * K + kk[None, :],
            mask=nmask[:, None] & kmask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w), input_precision="ieee")
    r = tl.rsqrt(ssq / K + EPS)
    mixes = acc * r[:, None]
    tl.store(
        MIXES + offs_m[:, None] * N + n[None, :],
        mixes,
        mask=mmask[:, None] & nmask[None, :],
    )
    tl.store(R + offs_m, r, mask=mmask)
    tl.store(
        G + offs_m[:, None] * N + n[None, :], acc, mask=mmask[:, None] & nmask[None, :]
    )


@triton.autotune(configs=_RL_CFGS, key=["M", "K", "N"], reset_to_zero=["DFN"])
@triton.jit
def _rmsln_bwd_kernel(
    STREAMS,
    FN,
    R,
    G,
    DMIX,
    DSTREAMS,
    DFN,
    M,
    K,
    N: tl.constexpr,
    NP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mmask = offs_m < M
    n = tl.arange(0, NP)
    nmask = n < N
    dmix = tl.load(
        DMIX + offs_m[:, None] * N + n[None, :],
        mask=mmask[:, None] & nmask[None, :],
        other=0.0,
    )
    g = tl.load(
        G + offs_m[:, None] * N + n[None, :],
        mask=mmask[:, None] & nmask[None, :],
        other=0.0,
    )
    r = tl.load(R + offs_m, mask=mmask, other=0.0)
    c = tl.sum(dmix * g, axis=1)  # dL/dr per row
    dG = dmix * r[:, None]  # [BM, NP]
    coef = (r * r * r) / K * c  # [BM]
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        kmask = kk < K
        w = tl.load(
            FN + n[:, None] * K + kk[None, :],
            mask=nmask[:, None] & kmask[None, :],
            other=0.0,
        )
        x = tl.load(
            STREAMS + offs_m[:, None] * K + kk[None, :],
            mask=mmask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        dmf = tl.dot(dmix, w, input_precision="ieee")  # (d_mixes @ fn) [BM, BK]
        dx = r[:, None] * dmf - x * coef[:, None]
        tl.store(
            DSTREAMS + offs_m[:, None] * K + kk[None, :],
            dx,
            mask=mmask[:, None] & kmask[None, :],
        )
        dfn = tl.dot(tl.trans(dG), x, input_precision="ieee")  # [NP, BK]
        tl.atomic_add(
            DFN + n[:, None] * K + kk[None, :],
            dfn,
            mask=nmask[:, None] & kmask[None, :],
        )


class _RMSNormLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, streams_flat, fn, eps, BM):
        M, K = streams_flat.shape
        N = fn.shape[0]
        NP = triton.next_power_of_2(N)
        sf = streams_flat.contiguous()
        fn = fn.contiguous().float()
        mixes = torch.empty(M, N, device=sf.device, dtype=torch.float32)
        R = torch.empty(M, device=sf.device, dtype=torch.float32)
        G = torch.empty(M, N, device=sf.device, dtype=torch.float32)
        grid = lambda m: (triton.cdiv(M, m["BM"]),)
        _rmsln_fwd_kernel[grid](sf, fn, mixes, R, G, M, K, eps, N=N, NP=NP, BM=BM)
        ctx.save_for_backward(sf, fn, R, G)
        ctx.dims = (M, K, N, NP, BM)
        return mixes

    @staticmethod
    def backward(ctx, dmix):
        sf, fn, R, G = ctx.saved_tensors
        M, K, N, NP, BM = ctx.dims
        dstreams = torch.empty_like(sf, dtype=torch.float32)
        dfn = torch.zeros_like(fn)
        grid = lambda m: (triton.cdiv(M, m["BM"]),)
        _rmsln_bwd_kernel[grid](
            sf,
            fn,
            R,
            G,
            dmix.contiguous().float(),
            dstreams,
            dfn,
            M,
            K,
            N=N,
            NP=NP,
            BM=BM,
        )
        return dstreams.to(sf.dtype), dfn, None, None


def _rmsnorm_linear(streams_flat, fn, eps, BM=16):
    return _RMSNormLinear.apply(streams_flat, fn, eps, BM)


# fused collapse: collapsed[m,d] = sum_h pre[m,h] * streams[m,h,d]
@triton.autotune(
    configs=[
        triton.Config({"BD": bd}, num_warps=w)
        for bd in (256, 512, 1024)
        for w in (4, 8)
    ],
    key=["M", "D"],
)
@triton.jit
def _collapse_fwd_kernel(PRE, STREAMS, OUT, M, D, HC: tl.constexpr, BD: tl.constexpr):
    m = tl.program_id(0)
    dblk = tl.program_id(1) * BD + tl.arange(0, BD)
    dmask = dblk < D
    acc = tl.zeros([BD], tl.float32)
    for h in range(HC):
        p = tl.load(PRE + m * HC + h)
        s = tl.load(STREAMS + m * (HC * D) + h * D + dblk, mask=dmask, other=0.0).to(
            tl.float32
        )
        acc += p * s
    tl.store(OUT + m * D + dblk, acc, mask=dmask)


@triton.autotune(
    configs=[
        triton.Config({"BD": bd}, num_warps=w)
        for bd in (256, 512, 1024)
        for w in (4, 8)
    ],
    key=["M", "D"],
    reset_to_zero=["DPRE"],
)
@triton.jit
def _collapse_bwd_kernel(
    PRE, STREAMS, GOUT, DPRE, DSTREAMS, M, D, HC: tl.constexpr, BD: tl.constexpr
):
    m = tl.program_id(0)
    dblk = tl.program_id(1) * BD + tl.arange(0, BD)
    dmask = dblk < D
    g = tl.load(GOUT + m * D + dblk, mask=dmask, other=0.0).to(tl.float32)
    for h in range(HC):
        p = tl.load(PRE + m * HC + h)
        s = tl.load(STREAMS + m * (HC * D) + h * D + dblk, mask=dmask, other=0.0).to(
            tl.float32
        )
        tl.store(DSTREAMS + m * (HC * D) + h * D + dblk, p * g, mask=dmask)
        tl.atomic_add(DPRE + m * HC + h, tl.sum(tl.where(dmask, g * s, 0.0)))


class _Collapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre, streams):
        M, HC, D = streams.shape
        pre = pre.contiguous()
        streams = streams.contiguous()
        out = torch.empty(M, D, device=streams.device, dtype=torch.float32)
        grid = lambda m: (M, triton.cdiv(D, m["BD"]))
        _collapse_fwd_kernel[grid](pre, streams, out, M, D, HC=HC)
        ctx.save_for_backward(pre, streams)
        ctx.dims = (M, HC, D)
        return out

    @staticmethod
    def backward(ctx, gout):
        pre, streams = ctx.saved_tensors
        M, HC, D = ctx.dims
        dpre = torch.zeros_like(pre)
        dstreams = torch.empty_like(streams, dtype=torch.float32)
        grid = lambda m: (M, triton.cdiv(D, m["BD"]))
        _collapse_bwd_kernel[grid](
            pre, streams, gout.contiguous().float(), dpre, dstreams, M, D, HC=HC
        )
        return dpre, dstreams.to(streams.dtype)


def hyperconnection_forward(
    hidden_streams, input_norm, fn, base, scale, hc, iters, eps, post_mult=2.0
):
    """Drop-in for DeepseekV4HyperConnection.forward. Returns (post, comb, collapsed).
    Fully fused Triton: rmsnorm+fn-linear (one pass over streams), Sinkhorn mixer, and the
    weighted collapse — eager does these in ~3 passes over the 16384-wide streams + a 536MB
    fp32 intermediate."""
    dtype = hidden_streams.dtype
    # match the fused-linear / Sinkhorn params to the stream compute dtype (they may be fp32).
    fn = fn.to(dtype) if fn.dtype != dtype else fn
    scale = scale.to(dtype) if scale.dtype != dtype else scale
    base = base.to(dtype) if base.dtype != dtype else base
    *lead, _, hidden = hidden_streams.shape
    streams_flat = hidden_streams.reshape(-1, hc * hidden)
    mixes = _rmsnorm_linear(streams_flat, fn, input_norm.eps)
    pre, post, comb = sinkhorn_mix(mixes, scale, base, hc, iters, eps, post_mult)
    collapsed = (
        _Collapse.apply(pre, hidden_streams.reshape(-1, hc, hidden))
        .reshape(*lead, hidden)
        .to(dtype)
    )
    post = post.reshape(*lead, hc)
    comb = comb.reshape(*lead, hc, hc)
    return post, comb, collapsed
