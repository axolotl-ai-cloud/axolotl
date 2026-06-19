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


def deepgemm_grouped_available() -> bool:
    """True iff DeepGEMM's grouped fp8/fp4 MoE kernel can run here (SM90/SM100 + CUDA
    runtime new enough + the ``kernels`` package resolves the build)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major not in (9, 10):
        return False
    try:
        from kernels import get_kernel

        dg = get_kernel("kernels-community/deep-gemm")
        return getattr(dg, "m_grouped_fp8_fp4_gemm_nt_contiguous", None) is not None
    except Exception:
        return False


@triton.jit
def _nvfp4_deq_kernel(Qp, Sc, Pt, Out, Kd, ROWS_PER_E, CB,
                      sq0, sq1, ss0, ss1, so0, so1, BK: tl.constexpr):
    rid = tl.program_id(0)
    kk = tl.program_id(1) * BK + tl.arange(0, BK)
    km = kk < Kd
    pt = tl.load(Pt + rid // ROWS_PER_E).to(tl.float32)
    packed = tl.load(Qp + rid * sq0 + (kk // 2) * sq1, mask=km, other=0).to(tl.int32)
    nib = tl.where((kk % 2) == 1, (packed >> 4) & 0xF, packed & 0xF)
    sc = tl.load(Sc + rid * ss0 + (kk // 16) * ss1, mask=km, other=0.0).to(tl.float32)
    tl.store(Out + rid * so0 + kk * so1, (tl.load(CB + nib) * sc * pt).to(tl.bfloat16), mask=km)


def nvfp4_dequant_bf16(qdata: torch.Tensor, scale: torch.Tensor, per_tensor: torch.Tensor) -> torch.Tensor:
    """NVFP4 -> bf16 via a memory-bound Triton kernel (~35x faster than torchao eager).
    ``qdata`` [c,N,K/2] uint8, ``scale`` [c,N,K/16] e4m3, ``per_tensor`` [c] fp32 → [c,N,K] bf16."""
    c, nrows, kh = qdata.shape
    kd = kh * 2
    out = torch.empty(c, nrows, kd, device=qdata.device, dtype=torch.bfloat16)
    q2, s2, o2 = qdata.reshape(c * nrows, kh), scale.reshape(c * nrows, kd // 16), out.reshape(c * nrows, kd)
    _nvfp4_deq_kernel[(c * nrows, triton.cdiv(kd, 256))](
        q2, s2, per_tensor.contiguous(), o2, kd, nrows, fp4_codebook(qdata.device),
        q2.stride(0), q2.stride(1), s2.stride(0), s2.stride(1), o2.stride(0), o2.stride(1), BK=256,
    )
    return out


class _GroupedMM(torch.autograd.Function):
    """``torch._grouped_mm`` with a manual backward (its autograd is stride-broken). Frozen
    weight ``bT`` (the dequantized experts) → only the input grad ``dX`` is returned."""

    @staticmethod
    def forward(ctx, a, bT, offs):  # a[M,K], bT[E,K,N] -> [M,N]
        ctx.save_for_backward(bT, offs)
        return torch._grouped_mm(a, bT, offs=offs)

    @staticmethod
    def backward(ctx, g):
        bT, offs = ctx.saved_tensors
        dA = torch._grouped_mm(g.contiguous(), bT.transpose(1, 2).contiguous(), offs=offs)
        return dA, None, None


def _chunk_forward(sorted_tok, gqd, gsc, dqd, dsc, pt, coff, limit):
    """One expert-chunk: dequant -> grouped gate_up -> clamped-SwiGLU -> grouped down."""
    gub = nvfp4_dequant_bf16(gqd, gsc, pt).transpose(1, 2)  # [c,H,2I]
    dnb = nvfp4_dequant_bf16(dqd, dsc, pt).transpose(1, 2)  # [c,I,H]
    x = _GroupedMM.apply(sorted_tok, gub, coff)
    g, u = x.chunk(2, dim=-1)
    h = (torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)).contiguous()
    return _GroupedMM.apply(h, dnb, coff)


def chunked_dequant_grouped_base(hidden, idx, wts, gate_up_nv, down_nv, per_tensor,
                                 limit, chunk_size=8):
    """Base clamped-SwiGLU MoE on NVFP4 experts via chunked dequant + grouped_mm.

    hidden [N,H]; idx [N,topk]; wts [N,topk]; gate_up_nv/down_nv: NVFP4Tensor
    ([E,2I,H]/[E,H,I]); per_tensor: scalar or [E] fp32. Returns [N,H]. Differentiable to
    ``hidden``. Per-chunk ``checkpoint`` keeps the bf16 weight transient bounded (re-dequant
    in backward).
    """
    E = gate_up_nv.qdata.size(0)
    Hdim = hidden.size(1)
    # raw-row-major scale indexing only; deswizzling is not implemented here
    assert not getattr(gate_up_nv, "is_swizzled_scales", False), "swizzled NVFP4 scales unsupported"
    assert not getattr(down_nv, "is_swizzled_scales", False), "swizzled NVFP4 scales unsupported"
    # torchao per_tensor_scale is either a global scalar or per-expert; normalize to [E]
    per_tensor = per_tensor.reshape(-1).to(torch.float32)
    if per_tensor.numel() == 1:
        per_tensor = per_tensor.expand(E)
    flat_exp = idx.reshape(-1)
    order = flat_exp.argsort()
    rep = torch.arange(hidden.size(0), device=hidden.device).repeat_interleave(idx.size(1))[order]
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
        o = checkpoint(_chunk_forward, flat[t0:t1], gq[c0:c1], gs[c0:c1], dq[c0:c1], ds[c0:c1],
                       per_tensor[c0:c1], coff, limit, use_reentrant=False)
        pieces.append(o * wflat[t0:t1, None].to(o.dtype))
    if not pieces:  # no routed tokens (e.g. empty batch / expert-parallel shard)
        return out
    return out.index_add(0, rep, torch.cat(pieces, 0))
