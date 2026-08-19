"""Fused flash attention for DeepSeek-V4 sliding-window layers (fwd + bwd).

Replaces the forced-eager path in
``transformers.models.deepseek_v4.modeling_deepseek_v4.eager_attention_forward``
for ``sliding_attention`` layers. V4 cannot use FA2/3/4 (head_dim 512 > 256 cap),
SDPA (no per-head sink), or flex (compressor KV-concat), so eager — which
materializes ``[B, H, S, KV]`` scores and broadcasts the single MQA KV head ×64 via
``repeat_kv`` — is the only baseline. This kernel:

  * keeps the single MQA KV head in SRAM (no ``repeat_kv`` blow-up),
  * never materializes the score matrix,
  * masks the sliding window *implicitly* from indices, so work is O(S * window)
    instead of O(S * KV),
  * folds the gpt-oss-style per-head learnable **sink** into the online softmax by
    seeding ``m = sink`` / ``l = 1`` (the sink is a logit-only column: it enters the
    denominator, contributes nothing to the output),
  * exposes an ``acc_dtype`` knob (fp32 vs bf16 online-softmax/PV accumulation) so the
    speed/memory vs error trade-off can be measured per path.

Sliding-window semantics (verified against ``create_sliding_window_causal_mask``):
query ``i`` attends to key ``j`` iff ``j <= i`` and ``i - j < sliding_window``.
"""

import functools

import torch
import triton
import triton.language as tl

from axolotl.kernels.op_registry import register_kernel_op

# Autotuner prunes configs that overflow SMEM per dtype/head_dim. ``EL`` (element size) is
# in the autotune key so fp32 and bf16 are tuned/cached separately.
_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for bm in (16, 32, 64, 128)
    for bn in (16, 32, 64)
    for w in (4, 8)
    for s in (1, 2)
    # N=16 wgmma at large M illegal-accesses on Hopper/Blackwell; keep BN=16 only for small
    # BM (the fp32 D=512 path needs the small tile).
    if not (bn == 16 and bm >= 64)
]


@functools.lru_cache(maxsize=None)
def _smem_limit(dev):
    # Real per-block SMEM of this GPU (portable: sm_120 ~99KB, H100/B200 ~228KB).
    try:
        return triton.runtime.driver.active.utils.get_device_properties(dev)[
            "max_shared_mem"
        ]
    except Exception:
        return 101376


@functools.lru_cache(maxsize=None)
def _max_m(dev):
    # Blackwell (cc >= 10, tcgen05) keeps the MMA accumulator in tensor memory (512 cols).
    # With D=512 the [M,512] acc nearly fills tmem, so M>32 misaligns (sm_100) / OOMs tmem;
    # cap M<=32 there. Hopper (cc 9, no tmem) handles up to 128.
    try:
        return 32 if torch.cuda.get_device_capability(dev)[0] >= 10 else 128
    except Exception:
        return 128


def _prune_smem(n_qtiles, n_kvtiles):
    """Drop configs that would overflow SMEM (D-wide tiles ×num_stages) or, on Blackwell,
    exceed the tensor-memory accumulator limit (M>32). Portable; the autotuner's own
    OutOfResources catch is the backstop."""

    def prune(configs, nargs, **kwargs):
        D, EL = (
            kwargs["D"],
            nargs["EL"],
        )  # D is constexpr (kwargs); EL is runtime (nargs)
        dev = torch.cuda.current_device()
        budget = int(_smem_limit(dev) * 0.9)
        max_m = _max_m(dev)
        kept = []
        for c in configs:
            bm, bn, st = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.num_stages
            est = (n_qtiles * bm + n_kvtiles * st * bn) * D * EL
            if est <= budget and bm <= max_m:
                kept.append(c)
        return kept or [
            min(
                configs,
                key=lambda c: c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"] * c.num_stages,
            )
        ]

    return prune


@triton.autotune(
    configs=_CONFIGS,
    key=["H", "EL"],
    prune_configs_by={"early_config_prune": _prune_smem(1, 2)},
)  # q + (k,v)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sinks,
    Out,
    L,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    H,
    S,
    KV,
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

    q_ptr = (
        Q
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptr, mask=offs_m[:, None] < S, other=0.0)

    sink = tl.load(sinks + h).to(tl.float32)
    m_i = tl.zeros([BLOCK_M], tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], tl.float32) + 1.0  # exp(sink - sink) seed
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # sliding window: keys in (m - window, m]
    lo = tl.maximum(0, (pid_m * BLOCK_M - WINDOW + 1))
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(KV, pid_m * BLOCK_M + BLOCK_M)

    kbase = K + b * stride_kb
    vbase = V + b * stride_vb
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < KV
        k = tl.load(
            kbase + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
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
            vbase + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=nmask[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(ACC), v.to(ACC), input_precision=PREC).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptr = (
        Out
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < S)
    tl.store(L + pid_bh * S + offs_m, m_i + tl.log(l_i), mask=offs_m < S)


@register_kernel_op("dsv4_sliding_attn_fwd")
def _sliding_attn_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    window: int,
    acc_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = q.shape
    KV = k.shape[1]
    out = torch.empty(B, H, S, D, device=q.device, dtype=q.dtype)
    L = torch.empty(B * H, S, device=q.device, dtype=torch.float32)
    ACC = tl.float32 if acc_fp32 else tl.bfloat16
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)
    _fwd_kernel[grid](
        q,
        k,
        v,
        sinks,
        out,
        L,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        H,
        S,
        KV,
        window,
        q.element_size(),
        D=D,
        ACC=ACC,
        PREC=("ieee" if q.element_size() == 4 else "tf32"),
    )
    return out, L


@_sliding_attn_fwd_op.register_fake
def _(q, k, v, sinks, scale, window, acc_fp32):
    B, H, S, _ = q.shape
    return torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty(
        B * H, S, device=q.device, dtype=torch.float32
    )


# Two-kernel backward (dq pass, dk/dv pass): splitting halves the resident D-wide dot
# operands per kernel so 16x16 tiles fit ~99KB SMEM.
@triton.autotune(
    configs=_CONFIGS,
    key=["H", "EL"],
    reset_to_zero=["DSINK"],
    prune_configs_by={"early_config_prune": _prune_smem(2, 2)},
)
@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    sinks,
    DO,
    L,
    Delta,
    DQ,
    DSINK,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    H,
    S,
    KV,
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
        Q
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=mmask[:, None],
        other=0.0,
    )
    do = tl.load(
        DO
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=mmask[:, None],
        other=0.0,
    )
    l_i = tl.load(L + pid_bh * S + offs_m, mask=mmask, other=0.0)
    delta = tl.load(Delta + pid_bh * S + offs_m, mask=mmask, other=0.0)
    sink = tl.load(sinks + h).to(tl.float32)

    dq = tl.zeros([BLOCK_M, D], tl.float32)
    lo = (tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N) * BLOCK_N
    hi = tl.minimum(KV, pid_m * BLOCK_M + BLOCK_M)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < KV
        k = tl.load(
            K
            + b * stride_kb
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=nmask[:, None],
            other=0.0,
        )
        v = tl.load(
            V
            + b * stride_kb
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=nmask[:, None],
            other=0.0,
        )
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

    p_sink = tl.exp(sink - l_i)
    tl.atomic_add(DSINK + h, tl.sum(tl.where(mmask, -p_sink * delta, 0.0)))
    tl.store(
        DQ
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        dq.to(DQ.dtype.element_ty),
        mask=mmask[:, None],
    )


@triton.autotune(
    configs=_CONFIGS,
    key=["H", "EL"],
    reset_to_zero=["DK", "DV"],
    prune_configs_by={"early_config_prune": _prune_smem(2, 2)},
)
@triton.jit
def _bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    L,
    Delta,
    DK,
    DV,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    H,
    S,
    KV,
    WINDOW,
    EL,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    nmask = offs_n < KV

    k = tl.load(
        K + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=nmask[:, None],
        other=0.0,
    )
    v = tl.load(
        V + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=nmask[:, None],
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, D], tl.float32)
    dv = tl.zeros([BLOCK_N, D], tl.float32)

    # queries that attend to this kv block: i in [n0, n0 + BLOCK_N - 1 + WINDOW - 1]
    lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    hi = tl.minimum(S, pid_n * BLOCK_N + BLOCK_N + WINDOW - 1)
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mmask = offs_m < S
        q = tl.load(
            Q
            + b * stride_qb
            + h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
            mask=mmask[:, None],
            other=0.0,
        )
        do = tl.load(
            DO
            + b * stride_qb
            + h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
            mask=mmask[:, None],
            other=0.0,
        )
        l_i = tl.load(L + pid_bh * S + offs_m, mask=mmask, other=0.0)
        delta = tl.load(Delta + pid_bh * S + offs_m, mask=mmask, other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        valid = (
            (offs_n[None, :] <= offs_m[:, None])
            & (offs_m[:, None] - offs_n[None, :] < WINDOW)
            & nmask[None, :]
        )
        p = tl.where(valid, tl.exp(qk - l_i[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=PREC)
        ds = (p * (dp - delta[:, None]) * scale).to(ACC)
        dk += tl.dot(tl.trans(ds), q.to(ACC), input_precision=PREC).to(tl.float32)
        dv += tl.dot(tl.trans(p.to(ACC)), do.to(ACC), input_precision=PREC).to(
            tl.float32
        )

    # shared MQA head: sum across q heads via atomics
    dk_ptr = (
        DK + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    )
    dv_ptr = (
        DV + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    )
    tl.atomic_add(dk_ptr, dk, mask=nmask[:, None])
    tl.atomic_add(dv_ptr, dv, mask=nmask[:, None])


@register_kernel_op("dsv4_sliding_attn_bwd")
def _sliding_attn_bwd_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    L: torch.Tensor,
    scale: float,
    window: int,
    acc_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, D = q.shape
    KV = k.shape[1]
    do = do.contiguous()
    delta = (do.to(torch.float32) * out.to(torch.float32)).sum(-1).reshape(B * H, S)
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dsink = torch.zeros_like(sinks, dtype=torch.float32)
    ACC = tl.float32 if acc_fp32 else tl.bfloat16
    PREC = "ieee" if q.element_size() == 4 else "tf32"
    EL = q.element_size()
    args = (
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        H,
        S,
        KV,
        window,
        EL,
    )
    _bwd_dq_kernel[lambda m: (triton.cdiv(S, m["BLOCK_M"]), B * H)](
        q, k, v, sinks, do, L, delta, dq, dsink, scale, *args, D=D, ACC=ACC, PREC=PREC
    )
    _bwd_dkdv_kernel[lambda m: (triton.cdiv(KV, m["BLOCK_N"]), B * H)](
        q, k, v, do, L, delta, dk, dv, scale, *args, D=D, ACC=ACC, PREC=PREC
    )
    return dq, dk.to(k.dtype), dv.to(v.dtype), dsink.to(sinks.dtype)


@_sliding_attn_bwd_op.register_fake
def _(do, q, k, v, sinks, out, L, scale, window, acc_fp32):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(sinks),
    )


class _SlidingAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, scale, window, acc_dtype):
        out, L = _sliding_attn_fwd_op(
            q, k, v, sinks, scale, window, acc_dtype == torch.float32
        )
        ctx.save_for_backward(q, k, v, sinks, out, L)
        ctx.scale, ctx.window, ctx.acc_dtype = scale, window, acc_dtype
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, out, L = ctx.saved_tensors
        dq, dk, dv, dsink = _sliding_attn_bwd_op(
            do,
            q,
            k,
            v,
            sinks,
            out,
            L,
            ctx.scale,
            ctx.window,
            ctx.acc_dtype == torch.float32,
        )
        return dq, dk, dv, dsink, None, None, None


def sliding_attn(
    q, k, v, sinks, scale=None, sliding_window=128, acc_dtype=torch.float32
):
    """q: [B, H, S, D]; k, v: [B, 1, KV, D] (single MQA head); sinks: [H].
    Returns attn output [B, H, S, D]. ``acc_dtype`` controls PV/dq/dk/dv accumulation."""
    B, H, S, D = q.shape
    if scale is None:
        scale = D**-0.5
    # match k/v/sinks to q so the kernel never sees mixed dtypes
    dt = q.dtype
    k = k.to(dt) if k.dtype != dt else k
    v = v.to(dt) if v.dtype != dt else v
    sinks = sinks.to(dt) if sinks.dtype != dt else sinks
    k2 = k[:, 0].contiguous()
    v2 = v[:, 0].contiguous()
    return _SlidingAttn.apply(
        q.contiguous(), k2, v2, sinks.contiguous(), scale, sliding_window, acc_dtype
    )
