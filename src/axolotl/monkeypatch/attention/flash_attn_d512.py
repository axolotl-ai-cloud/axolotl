"""Triton FlashAttention for head_dim=512 (forward + backward, dense + varlen packing).

Flash/cuDNN SDPA backends cap at head_dim 256; at 512 PyTorch falls back to the memory-efficient or
math backend (O(S^2), and math materializes the full scores). This kernel fills that gap: a tiled
flash attention that fits head_dim 512 by using small M/N blocks (the d=512 register/SMEM wall) and
runs fwd+bwd. Varlen packing is done in-kernel from position_ids (doc_start = row - position_id), so
no block-diagonal mask tensor is built. Validated bit-accurate (cosine 1.0) vs per-document SDPA and
~2x faster than the SDPA-efficient fallback at head_dim 512 on sm_120.

Primary use: the Gemma-4 global (full_attention, head_dim=512) layers under sample packing.
"""

import torch
import triton
import triton.language as tl

from axolotl.kernels.op_registry import register_kernel_op

DEV = "cuda"


def _fwd_smem_bytes(BLOCK_N, head_dim, num_stages):
    """Exact SMEM the forward kernel needs: the q-tile lives in registers; the K and V tiles are
    (1+num_stages)-buffered for software pipelining (+~2KB scratch for L/reductions). Calibrated
    against compiled kernels: (BN,D)=(32,512) -> 67584 B @ns1, 100352 B @ns2 (clean +32768/stage)."""
    return BLOCK_N * head_dim * 2 * (1 + num_stages) + 2048


def _device_smem_limit():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (
        getattr(props, "shared_memory_per_block_optin", 0)
        or props.shared_memory_per_block
    )


def _prune_fwd_configs(configs, named_args, **kwargs):
    """Drop configs whose estimated SMEM exceeds the device opt-in cap before they ever compile, so
    autotuning doesn't pay the compile cost of doomed variants (Triton would also reject them at
    compile via OutOfResources; this just skips the attempt). Always keep at least one."""
    # constexpr meta (HEAD_DIM/CAUSAL/VARLEN) arrives via kwargs, runtime args via named_args.
    head_dim = kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM"))
    if (
        head_dim is None
    ):  # can't estimate -> let Triton's compile-time OOM check prune instead
        return list(configs)
    limit = _device_smem_limit()
    fit = [
        c
        for c in configs
        if _fwd_smem_bytes(c.kwargs["BLOCK_N"], head_dim, c.num_stages) <= limit
    ]
    return fit or [min(configs, key=lambda c: c.num_stages)]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=s)
        for s in (1, 2, 3)
    ],
    key=["N_CTX", "HEAD_DIM", "CAUSAL", "VARLEN"],
    prune_configs_by={"early_config_prune": _prune_fwd_configs},
)
@triton.jit
def _fwd(
    Q,
    K,
    V,
    sm_scale,
    Out,
    L,
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
    POS,
    spb,
    spn,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    qb = Q + off_b * sqb + off_h * sqh
    kb = K + off_b * skb + off_h * skh
    vb = V + off_b * svb + off_h * svh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        qb + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )
    ds_m = (
        offs_m - tl.load(POS + off_b * spb + offs_m * spn, mask=offs_m < N_CTX, other=0)
        if VARLEN
        else offs_m * 0
    )
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    n_end = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    n_start = (
        tl.min(ds_m) // BLOCK_N
    ) * BLOCK_N  # varlen: skip kv blocks before this q-block's doc
    for start_n in range(n_start, n_end, BLOCK_N):
        cur = start_n + offs_n
        k = tl.load(
            kb + cur[None, :] * skn + offs_d[:, None] * skd,
            mask=cur[None, :] < N_CTX,
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale
        mask = cur[None, :] < N_CTX
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= cur[None, :])
        if VARLEN:
            mask = mask & (cur[None, :] >= ds_m[:, None])
        qk = tl.where(mask, qk, -1.0e9)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            vb + cur[:, None] * svn + offs_d[None, :] * svd,
            mask=cur[:, None] < N_CTX,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(
        Out + off_b * sob + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < N_CTX,
    )
    tl.store(L + off_bh * N_CTX + offs_m, m_i + tl.log(l_i), mask=offs_m < N_CTX)


@triton.jit
def _bwd_pre(
    O,  # noqa: E741
    DO,
    Delta,
    sob,
    soh,
    som,
    sod,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    base = off_b * sob + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    o = tl.load(O + base, mask=offs_m[:, None] < N_CTX, other=0.0)
    do = tl.load(DO + base, mask=offs_m[:, None] < N_CTX, other=0.0)
    tl.store(
        Delta + off_bh * N_CTX + offs_m,
        tl.sum(o.to(tl.float32) * do.to(tl.float32), 1),
        mask=offs_m < N_CTX,
    )


@triton.jit
def _bwd_dkdv(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    L,
    D,
    POS,
    DOC_END,
    spb,
    spn,
    sqb,
    sqh,
    sqm,
    sqd,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    base = off_b * sqb + off_h * sqh
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)
    k = tl.load(
        K + base + offs_n[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_n[:, None] < N_CTX,
        other=0.0,
    )
    v = tl.load(
        V + base + offs_n[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_n[:, None] < N_CTX,
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    m_start = start_n * BLOCK_N if CAUSAL else 0
    # varlen: queries beyond this kv-block's document don't attend -> stop at its doc end
    last_k = tl.minimum(start_n * BLOCK_N + BLOCK_N - 1, N_CTX - 1)
    m_end = tl.load(DOC_END + off_b * spb + last_k * spn) if VARLEN else N_CTX
    for start_m in range(m_start, m_end, BLOCK_M):
        cur_m = start_m + offs_m
        q = tl.load(
            Q + base + cur_m[:, None] * sqm + offs_d[None, :] * sqd,
            mask=cur_m[:, None] < N_CTX,
            other=0.0,
        )
        do = tl.load(
            DO + base + cur_m[:, None] * sqm + offs_d[None, :] * sqd,
            mask=cur_m[:, None] < N_CTX,
            other=0.0,
        )
        l = tl.load(L + off_bh * N_CTX + cur_m, mask=cur_m < N_CTX, other=0.0)  # noqa: E741
        delta = tl.load(D + off_bh * N_CTX + cur_m, mask=cur_m < N_CTX, other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        mask = (cur_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        if CAUSAL:
            mask = mask & (cur_m[:, None] >= offs_n[None, :])
        if VARLEN:
            ds_m = cur_m - tl.load(
                POS + off_b * spb + cur_m * spn, mask=cur_m < N_CTX, other=0
            )
            mask = mask & (offs_n[None, :] >= ds_m[:, None])
        p = tl.where(mask, tl.exp(s - l[:, None]), 0.0)
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None]) * sm_scale).to(q.dtype)
        dk += tl.dot(tl.trans(ds), q)
    tl.store(
        DK + base + offs_n[:, None] * sqm + offs_d[None, :] * sqd,
        dk.to(DK.dtype.element_ty),
        mask=offs_n[:, None] < N_CTX,
    )
    tl.store(
        DV + base + offs_n[:, None] * sqm + offs_d[None, :] * sqd,
        dv.to(DV.dtype.element_ty),
        mask=offs_n[:, None] < N_CTX,
    )


@triton.jit
def _bwd_dq(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    L,
    D,
    POS,
    spb,
    spn,
    sqb,
    sqh,
    sqm,
    sqd,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    base = off_b * sqb + off_h * sqh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )
    do = tl.load(
        DO + base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )
    l = tl.load(L + off_bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)  # noqa: E741
    delta = tl.load(D + off_bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    ds_m = (
        offs_m - tl.load(POS + off_b * spb + offs_m * spn, mask=offs_m < N_CTX, other=0)
        if VARLEN
        else offs_m * 0
    )
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    n_end = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    n_start = (tl.min(ds_m) // BLOCK_N) * BLOCK_N
    for start_n in range(n_start, n_end, BLOCK_N):
        cur = start_n + offs_n
        k = tl.load(
            K + base + cur[:, None] * sqm + offs_d[None, :] * sqd,
            mask=cur[:, None] < N_CTX,
            other=0.0,
        )
        v = tl.load(
            V + base + cur[:, None] * sqm + offs_d[None, :] * sqd,
            mask=cur[:, None] < N_CTX,
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * sm_scale
        mask = (offs_m[:, None] < N_CTX) & (cur[None, :] < N_CTX)
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= cur[None, :])
        if VARLEN:
            mask = mask & (cur[None, :] >= ds_m[:, None])
        p = tl.where(mask, tl.exp(s - l[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None]) * sm_scale).to(k.dtype)
        dq += tl.dot(ds, k)
    tl.store(
        DQ + base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        dq.to(DQ.dtype.element_ty),
        mask=offs_m[:, None] < N_CTX,
    )


@register_kernel_op("flash_attn_d512_fwd")
def _flash_d512_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos: torch.Tensor,
    causal: bool,
    varlen: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = q.shape
    spb, spn = pos.stride()
    o = torch.empty_like(q)
    L = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    # BLOCK_M/BLOCK_N, num_warps, num_stages come from the autotuner (SMEM-pruned per device).
    _fwd[lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)](
        q,
        k,
        v,
        scale,
        o,
        L,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        pos,
        spb,
        spn,
        H,
        N,
        HEAD_DIM=D,
        CAUSAL=causal,
        VARLEN=varlen,
    )
    return o, L


@_flash_d512_fwd_op.register_fake
def _(q, k, v, pos, causal, varlen, scale):
    B, H, N, _ = q.shape
    return torch.empty_like(q), torch.empty(
        (B * H, N), device=q.device, dtype=torch.float32
    )


@register_kernel_op("flash_attn_d512_bwd")
def _flash_d512_bwd_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    L: torch.Tensor,
    pos: torch.Tensor,
    doc_end: torch.Tensor,
    causal: bool,
    varlen: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, D = q.shape
    spb, spn = pos.stride()
    do = do.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    BM, BN = 16, 32
    _bwd_pre[(triton.cdiv(N, BM), B * H)](
        o, do, delta, *o.stride(), H, N, HEAD_DIM=D, BLOCK_M=BM, num_warps=4
    )
    _bwd_dkdv[(triton.cdiv(N, BN), B * H)](
        q,
        k,
        v,
        scale,
        do,
        dk,
        dv,
        L,
        delta,
        pos,
        doc_end,
        spb,
        spn,
        *q.stride(),
        H,
        N,
        HEAD_DIM=D,
        BLOCK_M=BM,
        BLOCK_N=BN,
        CAUSAL=causal,
        VARLEN=varlen,
        num_warps=4,
        num_stages=1,
    )
    _bwd_dq[(triton.cdiv(N, BM), B * H)](
        q,
        k,
        v,
        scale,
        do,
        dq,
        L,
        delta,
        pos,
        spb,
        spn,
        *q.stride(),
        H,
        N,
        HEAD_DIM=D,
        BLOCK_M=BM,
        BLOCK_N=BN,
        CAUSAL=causal,
        VARLEN=varlen,
        num_warps=4,
        num_stages=1,
    )
    return dq, dk, dv


@_flash_d512_bwd_op.register_fake
def _(do, q, k, v, o, L, pos, doc_end, causal, varlen, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _compute_doc_end(pos: torch.Tensor) -> torch.Tensor:
    """Per-row index of the next document start after each position (or N).

    Suffix-min of "j if pos[j]==0 else N" shifted left by one — fully tensorized so
    torch.compile can trace it (the per-row ``.nonzero()`` loop graph-breaks).
    """
    B, N = pos.shape
    idx = torch.arange(N, device=pos.device, dtype=pos.dtype)
    starts_or_n = torch.where(pos == 0, idx.expand_as(pos), pos.new_full((), N))
    shifted = torch.cat([starts_or_n[:, 1:], pos.new_full((B, 1), N)], dim=1)
    return torch.flip(
        torch.cummin(torch.flip(shifted, dims=[1]), dim=1).values, dims=[1]
    )


class _FlashD512(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, position_ids, scale=None):
        # The backward kernels launch with q's strides only and reuse them for k/v/do/dk/dv. In real
        # attention q is non-contiguous ([B,S,H,D].transpose(1,2)) while k/v are contiguous (e.g. GQA
        # repeat_interleave); the mismatched strides make the backward read wrong memory and explode
        # the gradient (forward is unaffected -- it passes each tensor's own strides). Force one layout.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, N = q.shape[0], q.shape[2]
        D = q.shape[3]
        VARLEN = position_ids is not None
        if position_ids is None:
            pos = torch.zeros((B, N), device=q.device, dtype=torch.int32)
        else:
            pos = (
                (position_ids if position_ids.dim() > 1 else position_ids[None])
                .to(torch.int32)
                .contiguous()
            )
        if VARLEN:
            doc_end = _compute_doc_end(pos)
        else:
            doc_end = pos
        scale = D**-0.5 if scale is None else float(scale)
        o, L = _flash_d512_fwd_op(q, k, v, pos, causal, VARLEN, scale)
        ctx.save_for_backward(q, k, v, o, L, pos, doc_end)
        ctx.causal = causal
        ctx.scale = scale
        ctx.varlen = VARLEN
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L, pos, doc_end = ctx.saved_tensors
        dq, dk, dv = _flash_d512_bwd_op(
            do, q, k, v, o, L, pos, doc_end, ctx.causal, ctx.varlen, ctx.scale
        )
        return dq, dk, dv, None, None, None


def flash_d512(q, k, v, causal=True, position_ids=None, scale=None):
    return _FlashD512.apply(q, k, v, causal, position_ids, scale)
