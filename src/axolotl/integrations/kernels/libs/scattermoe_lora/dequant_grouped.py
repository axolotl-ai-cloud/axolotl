"""Dequant + grouped-GEMM MoE experts path — the fast alternative to the fully-fused
``scatter2scatter_lora_mx`` Triton kernel.

Benchmarks (DSV4-Flash expert shapes) showed the fused in-kernel-decode Triton kernel is
~4x slower than dequantizing NVFP4->bf16 and running a cuBLAS-grade grouped GEMM, because a
hand-written Triton MoE GEMM can't match cuBLAS/tensor-core throughput. This module provides:

  * primary (SM90/SM100 + DeepGEMM): native fused-decode fp4 grouped GEMM (no bf16 weight
    materialization, tensor-core speed) — ``deepgemm_grouped_available()`` gates it;
  * fallback (any other GPU, e.g. SM120): a fast custom-Triton NVFP4->bf16 dequant (~35x
    faster than torchao's eager ``.dequantize()``) + ``torch._grouped_mm``, tiled per
    expert-chunk + checkpointed so the bf16 weight transient stays bounded.

``torch._grouped_mm`` has no working autograd (stride bug), so ``_GroupedMM`` supplies the
backward manually (``dX = grouped_mm(grad, Wᵀ)``; experts are frozen → no weight grad).

STATUS: the chunked dequant+grouped BASE path is implemented + parity-validated here. Wiring
into ``parallel_linear_lora`` (LoRA-on-experts + the DeepGEMM primary call) is the remaining
integration step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint

from .mx_weights import fp4_codebook

_DG = None


def _dg():
    """Cached DeepGEMM kernel module. ``get_kernel`` is not idempotent — calling it twice
    re-runs the build's ``register_fake`` and raises, so resolve it once and reuse."""
    global _DG
    if _DG is None:
        from kernels import get_kernel

        _DG = get_kernel("kernels-community/deep-gemm")
    return _DG


def deepgemm_grouped_available() -> bool:
    """True iff DeepGEMM's grouped fp8xfp4 MoE kernel can run here.

    SM100 ONLY (Blackwell datacenter, e.g. B200). The kernel symbol resolves on SM90 (Hopper) too,
    but ``m_grouped_fp8_fp4_gemm_nt_contiguous`` asserts ``arch_major == 10`` at call time, so SM90
    must report unavailable and fall back to Marlin (which supports SM80-90,120). Gating on the
    capability up front avoids a hard CUDA assert mid-forward."""
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    if major != 10:  # fp8xfp4 grouped kernel is sm100-only
        return False
    try:
        return getattr(_dg(), "m_grouped_fp8_fp4_gemm_nt_contiguous", None) is not None
    except Exception:
        return False


@triton.jit
def _nvfp4_deq_kernel(
    Qp, Sc, Pt, Out, Kd, ROWS_PER_E, CB, sq0, sq1, ss0, ss1, so0, so1, BK: tl.constexpr
):
    rid = tl.program_id(0).to(tl.int64)  # c*nrows*stride overflows int32 at E=256
    kk = tl.program_id(1) * BK + tl.arange(0, BK)
    km = kk < Kd
    pt = tl.load(Pt + rid // ROWS_PER_E).to(tl.float32)
    packed = tl.load(Qp + rid * sq0 + (kk // 2) * sq1, mask=km, other=0).to(tl.int32)
    nib = tl.where((kk % 2) == 1, (packed >> 4) & 0xF, packed & 0xF)
    sc = tl.load(Sc + rid * ss0 + (kk // 16) * ss1, mask=km, other=0.0).to(tl.float32)
    tl.store(
        Out + rid * so0 + kk * so1,
        (tl.load(CB + nib) * sc * pt).to(tl.bfloat16),
        mask=km,
    )


def nvfp4_dequant_bf16(
    qdata: torch.Tensor, scale: torch.Tensor, per_tensor: torch.Tensor
) -> torch.Tensor:
    """NVFP4 -> bf16 via a memory-bound Triton kernel (~35x faster than torchao eager).
    ``qdata`` [c,N,K/2] uint8, ``scale`` [c,N,K/16] e4m3, ``per_tensor`` [c] fp32 → [c,N,K] bf16."""
    c, nrows, kh = qdata.shape
    kd = kh * 2
    out = torch.empty(c, nrows, kd, device=qdata.device, dtype=torch.bfloat16)
    q2, s2, o2 = (
        qdata.reshape(c * nrows, kh),
        scale.reshape(c * nrows, kd // 16),
        out.reshape(c * nrows, kd),
    )
    BK = 1024  # larger blocks lift HBM utilization (memory-bound kernel)
    _nvfp4_deq_kernel[(c * nrows, triton.cdiv(kd, BK))](
        q2,
        s2,
        per_tensor.contiguous(),
        o2,
        kd,
        nrows,
        fp4_codebook(qdata.device),
        q2.stride(0),
        q2.stride(1),
        s2.stride(0),
        s2.stride(1),
        o2.stride(0),
        o2.stride(1),
        BK=BK,
    )
    return out


@triton.jit
def _fp8_e8m0_q_kernel(X, O, SF, sx0, sx1, so0, so1, ss0, ss1, G: tl.constexpr):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = kb * G + tl.arange(0, G)
    x = tl.load(X + row * sx0 + offs * sx1).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    sf = tl.exp2(
        tl.ceil(tl.log2(amax / 448.0))
    )  # amax/448 rounded up to E8M0 (power of 2)
    q = (x / sf).to(tl.float8e4nv)
    tl.store(O + row * so0 + offs * so1, q)
    tl.store(SF + row * ss0 + kb * ss1, sf)


def fp8_e8m0_cast_128(x: torch.Tensor):
    """Fast Triton replica of DeepGEMM's ``per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128)``
    (exact bit-match, ~4x faster). x[M,K] bf16 -> (fp8_e4m3 [M,K], scale fp32 [M,K/128])."""
    M, K = x.shape
    assert K % 128 == 0
    o = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty(M, K // 128, device=x.device, dtype=torch.float32)
    _fp8_e8m0_q_kernel[(M, K // 128)](
        x,
        o,
        sf,
        x.stride(0),
        x.stride(1),
        o.stride(0),
        o.stride(1),
        sf.stride(0),
        sf.stride(1),
        G=128,
    )
    return o, sf


def nvfp4_to_mxfp4_weight(
    qdata: torch.Tensor, scale: torch.Tensor, per_tensor: torch.Tensor, chunk: int = 16
):
    """One-time per-expert weight requant NVFP4 (E4M3/16) -> MXFP4 (E8M0/128) for DeepGEMM's
    grouped fp8xfp4 kernel (which only accepts E8M0/128). qdata [E,N,K/2], scale [E,N,K/16],
    per_tensor [E] -> (wq [E,N,K/2] uint8, ws [E,N,K/128] fp32).

    Dequant + cast in expert CHUNKS into preallocated outputs so the bf16 intermediate is bounded to
    ``chunk`` experts (the full [E,N,K] bf16 was ~E/chunk x larger and OOMed at E=256). The output is
    full-size by construction (the grouped GEMM needs all experts present)."""
    cast = _dg().utils.per_token_cast_to_fp4
    E = int(qdata.size(0))
    q0, s0 = cast(
        nvfp4_dequant_bf16(qdata[:1], scale[:1], per_tensor[:1])[0].contiguous(),
        True,
        128,
    )
    wq = q0.new_empty((E, *q0.shape))
    ws = s0.new_empty((E, *s0.shape))
    for c0 in range(0, E, chunk):
        c1 = min(c0 + chunk, E)
        Wb = nvfp4_dequant_bf16(qdata[c0:c1], scale[c0:c1], per_tensor[c0:c1])
        for i in range(c1 - c0):
            q, s = cast(Wb[i].contiguous(), True, 128)
            wq[c0 + i] = q
            ws[c0 + i] = s
        del Wb
    return wq, ws


def deepgemm_grouped_fp8_fp4(
    a_bf16: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, m_indices: torch.Tensor
) -> torch.Tensor:
    """Contiguous-grouped fp8(act) x mxfp4(weight) base GEMM via DeepGEMM (SM90/SM100).
    a_bf16 [Mt,K], wq [E,N,K/2] uint8, ws [E,N,K/128] fp32, m_indices [Mt] int32 -> [Mt,N] bf16.
    Acts quantized to fp8 (E8M0/128) here; the kernel transforms the raw scales internally."""
    dg = _dg()
    a = fp8_e8m0_cast_128(a_bf16)
    d = torch.empty(
        a_bf16.size(0), wq.size(1), device=a_bf16.device, dtype=torch.bfloat16
    )
    dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a,
        (wq, ws),
        d,
        m_indices,
        recipe=None,
        recipe_a=(1, 128),
        recipe_b=(1, 128),
        disable_ue8m0_cast=False,
    )
    return d


def _cached_mxfp4(w_nv, per_tensor, cache=None, key=None):
    """Return (wq, ws) MXFP4 weight for a (frozen) NVFP4Tensor, requantized once and cached.
    Prefer a persistent ``cache`` dict (e.g. on the owning module) keyed by ``key`` — under
    FSDP2 the gathered param is a fresh tensor each step, so a module-level cache (not a
    per-tensor attribute) is what avoids recomputing the requant every forward. Falls back to a
    per-tensor attribute, then to a one-shot recompute."""
    from .runtime import RUNTIME

    if not RUNTIME.mxfp4_cache_persist:
        # FSDP: any attached cache accumulates a full-model mxfp4 copy across layers -> OOM.
        # Recompute uncached so resident mxfp4 stays bounded to one layer.
        return nvfp4_to_mxfp4_weight(w_nv.qdata, w_nv.scale, per_tensor)
    if cache is not None and key is not None and key in cache:
        return cache[key]
    cached = getattr(w_nv, "_dg_mxfp4", None)
    if cached is None:
        cached = nvfp4_to_mxfp4_weight(w_nv.qdata, w_nv.scale, per_tensor)
        try:
            w_nv._dg_mxfp4 = cached
        except (AttributeError, RuntimeError):
            pass
    if cache is not None and key is not None:
        cache[key] = cached
    return cached


@triton.jit
def _nvfp4_deq_fp8_kernel(
    Qp, Sc, Pt, Out, Kd, ROWS_PER_E, CB, sq0, sq1, ss0, ss1, so0, so1, BK: tl.constexpr
):
    rid = tl.program_id(0).to(tl.int64)
    kk = tl.program_id(1) * BK + tl.arange(0, BK)
    km = kk < Kd
    pt = tl.load(Pt + rid // ROWS_PER_E).to(tl.float32)
    packed = tl.load(Qp + rid * sq0 + (kk // 2) * sq1, mask=km, other=0).to(tl.int32)
    nib = tl.where((kk % 2) == 1, (packed >> 4) & 0xF, packed & 0xF)
    sc = tl.load(Sc + rid * ss0 + (kk // 16) * ss1, mask=km, other=0.0).to(tl.float32)
    tl.store(
        Out + rid * so0 + kk * so1,
        (tl.load(CB + nib) * sc * pt).to(tl.float8e4nv),
        mask=km,
    )


def nvfp4_dequant_fp8(
    qdata: torch.Tensor, scale: torch.Tensor, per_tensor: torch.Tensor
) -> torch.Tensor:
    """NVFP4 -> fp8 (e4m3) — half the bytes of the bf16 dequant. For the fp8-read backward dX
    (#3744): the grouped GEMM reads fp8 and upcasts to bf16 in-register, halving the weight's
    write+read bandwidth (a win on bandwidth-bound sm120; ~neutral speed but still half-memory
    on sm100)."""
    c, nrows, kh = qdata.shape
    kd = kh * 2
    out = torch.empty(c, nrows, kd, device=qdata.device, dtype=torch.float8_e4m3fn)
    q2, s2, o2 = (
        qdata.reshape(c * nrows, kh),
        scale.reshape(c * nrows, kd // 16),
        out.reshape(c * nrows, kd),
    )
    BK = 1024
    _nvfp4_deq_fp8_kernel[(c * nrows, triton.cdiv(kd, BK))](
        q2,
        s2,
        per_tensor.contiguous(),
        o2,
        kd,
        nrows,
        fp4_codebook(qdata.device),
        q2.stride(0),
        q2.stride(1),
        s2.stride(0),
        s2.stride(1),
        o2.stride(0),
        o2.stride(1),
        BK=BK,
    )
    return out


# BM is the routing pad TILE (128 cutlass, 64 marlin); a constexpr autotune key so BN/BK/warps
# still tune per (N,K,BM).
@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn in (64, 128)
        for bk in (128, 256)
        for w in (4, 8)
        for st in (3, 4)
    ],
    key=["N", "K", "BM"],
)
@triton.jit
def _grouped_dx_fp8_kernel(
    GRAD,
    W,
    MIDX,
    OUT,
    M,
    N,
    K,
    sg0,
    sg1,
    sw0,
    sw1,
    sw2,
    so0,
    so1,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    e = tl.load(MIDX + pid_m).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    rk = pid_k * BK + tl.arange(0, BK)
    mk = rk < K
    acc = tl.zeros((BM, BK), tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        a = tl.load(
            GRAD + rm[:, None] * sg0 + rn[None, :] * sg1,
            mask=rm[:, None] < M,
            other=0.0,
        )
        # mask K dim to handle non-BK-aligned K (e.g. I=704 which is not a multiple of 128)
        w = tl.load(
            W + e * sw0 + rn[:, None] * sw1 + rk[None, :] * sw2,
            mask=mk[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(a, w)
    tl.store(
        OUT + rm[:, None] * so0 + rk[None, :] * so1,
        acc.to(tl.bfloat16),
        mask=(rm[:, None] < M) & mk[None, :],
    )


def grouped_dx_fp8(
    grad: torch.Tensor, w_fp8: torch.Tensor, m_indices: torch.Tensor, block_m: int = 128
) -> torch.Tensor:
    """dX = grad @ W (contract N) for contiguous-grouped experts, reading the fp8 weight and
    upcasting in-register. grad[Mt,N] bf16, w_fp8[E,N,K] e4m3, m_indices[Mt/block_m] (one expert id
    per block_m-row tile; block_m = the routing pad TILE: 128 cutlass, 64 marlin) -> [Mt,K] bf16."""
    Mt, N = grad.shape
    K = w_fp8.size(2)
    out = torch.empty(Mt, K, device=grad.device, dtype=torch.bfloat16)
    grid = lambda meta: (Mt // block_m, triton.cdiv(K, meta["BK"]))  # noqa: E731
    _grouped_dx_fp8_kernel[grid](
        grad,
        w_fp8,
        m_indices,
        out,
        Mt,
        N,
        K,
        grad.stride(0),
        grad.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        w_fp8.stride(2),
        out.stride(0),
        out.stride(1),
        BM=block_m,
    )
    return out


class _GroupedMM(torch.autograd.Function):
    """``torch._grouped_mm`` with a manual backward (its autograd is stride-broken). Frozen
    weight ``bT`` (the dequantized experts) → only the input grad ``dX`` is returned."""

    @staticmethod
    def forward(ctx, a, bT, offs):
        ctx.save_for_backward(bT, offs)
        return torch._grouped_mm(a, bT, offs=offs)

    @staticmethod
    def backward(ctx, g):
        bT, offs = ctx.saved_tensors
        dA = torch._grouped_mm(
            g.contiguous(), bT.transpose(1, 2).contiguous(), offs=offs
        )
        return dA, None, None


def _chunk_forward(sorted_tok, gqd, gsc, dqd, dsc, pt, coff, limit):
    """One expert-chunk: dequant -> grouped gate_up -> clamped-SwiGLU -> grouped down."""
    gub = nvfp4_dequant_bf16(gqd, gsc, pt).transpose(1, 2)
    dnb = nvfp4_dequant_bf16(dqd, dsc, pt).transpose(1, 2)
    x = _GroupedMM.apply(sorted_tok, gub, coff)
    g, u = x.chunk(2, dim=-1)
    h = (
        torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
    ).contiguous()
    return _GroupedMM.apply(h, dnb, coff)


def chunked_dequant_grouped_base(
    hidden, idx, wts, gate_up_nv, down_nv, per_tensor, limit, chunk_size=8
):
    """Base clamped-SwiGLU MoE on NVFP4 experts via chunked dequant + grouped_mm.

    hidden [N,H]; idx [N,topk]; wts [N,topk]; gate_up_nv/down_nv: NVFP4Tensor
    ([E,2I,H]/[E,H,I]); per_tensor: scalar or [E] fp32. Returns [N,H]. Differentiable to
    ``hidden``. Per-chunk ``checkpoint`` keeps the bf16 weight transient bounded (re-dequant
    in backward).
    """
    E = gate_up_nv.qdata.size(0)
    Hdim = hidden.size(1)
    # raw-row-major scale indexing only; deswizzling is not implemented here
    assert not getattr(gate_up_nv, "is_swizzled_scales", False), (
        "swizzled NVFP4 scales unsupported"
    )
    assert not getattr(down_nv, "is_swizzled_scales", False), (
        "swizzled NVFP4 scales unsupported"
    )
    # per_tensor_scale may be a global scalar or per-expert; normalize to [E]
    per_tensor = per_tensor.reshape(-1).to(torch.float32)
    if per_tensor.numel() == 1:
        per_tensor = per_tensor.expand(E)
    flat_exp = idx.reshape(-1)
    order = flat_exp.argsort()
    rep = torch.arange(hidden.size(0), device=hidden.device).repeat_interleave(
        idx.size(1)
    )[order]
    wflat = wts.reshape(-1)[order]
    counts = torch.bincount(flat_exp, minlength=E)
    tok_off = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).tolist()
    flat = hidden[rep]

    out = hidden.new_zeros(hidden.size(0), Hdim)
    pieces = []
    gq, gs = gate_up_nv.qdata, gate_up_nv.scale
    dq, ds = down_nv.qdata, down_nv.scale
    for c0 in range(0, E, chunk_size):
        c1 = min(c0 + chunk_size, E)
        t0, t1 = int(tok_off[c0]), int(tok_off[c1])
        if t1 == t0:
            continue
        coff = counts[c0:c1].cumsum(0).to(torch.int32)
        o = checkpoint(
            _chunk_forward,
            flat[t0:t1],
            gq[c0:c1],
            gs[c0:c1],
            dq[c0:c1],
            ds[c0:c1],
            per_tensor[c0:c1],
            coff,
            limit,
            use_reentrant=False,
        )
        pieces.append(o * wflat[t0:t1, None].to(o.dtype))
    if not pieces:  # no routed tokens (e.g. empty batch / expert-parallel shard)
        return out
    return out.index_add(0, rep, torch.cat(pieces, 0))
