# mypy: disable-error-code="arg-type"
"""Fused RMSNorm -> NVFP4 quantization for consumer Blackwell (sm_120).

MSLK ships a fused rms+quant kernel (`triton_scale_nvfp4_quant_rms`) but it uses a
1D persistent thread-grid sized from the sm_100 SM count and faults with an illegal
memory access on sm_120. The plain quant kernel (`triton_quantize_nvfp4`) uses a 2D
M/N tile grid and runs fine on sm_120, so we rebuild the fusion on that grid instead.

RMS is a full-row reduction (needs all K columns) which a 64-col tile can't see, so we
take the reciprocal-norm factor in torch (one cheap read of x) and fuse the per-row
normalize + per-column gamma + NVFP4 quant into the tile kernel. The kernel emits the
normalized bf16 activation (for the LoRA adapter / next op) alongside the packed FP4
qdata and swizzled e4m3 scales consumed directly by ``torch._scaled_mm``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from mslk.quantize.triton.fp4_quantize import (
    convert_fp32_to_fp4_packed,
    nvfp4_scale_swizzle,
)
from torch import nn
from torch.utils.weak import WeakTensorKeyDictionary

# Maps a fused-norm output tensor `y` to its already-computed (fp4 qdata, e4m3
# scales). The norm output flows unchanged into the consuming NVFP4 base linear
# (no op between norm and projection), so identity lookup hits and the linear
# skips re-quantizing. Weak keys so entries clear when activations are freed.
_PREQUANT_CACHE = WeakTensorKeyDictionary()


def get_prequant(x: torch.Tensor):
    """Return cached (fp4, scales) for a fused-norm output, or None."""
    return _PREQUANT_CACHE.get(x)


@triton.jit
def _fused_rmsnorm_nvfp4_kernel(
    x_ptr,
    rms_ptr,  # [M] reciprocal rms factor, fp32
    w_ptr,  # [N] rmsnorm gamma
    y_ptr,  # [M, N] normalized activation out, bf16
    q_ptr,  # [M, N//2] packed fp4
    s_ptr,  # swizzled e4m3 scales
    stride_xm,
    stride_xn,
    M,
    N,
    M_PER_BLOCK: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    E4M3_EPS = 1.5258789e-05
    FP8_E4M3_MAX = 448.0
    FP4_E2M1_MAX = 6.0

    NUM_ELEM_PER_LAYOUT = 128 * 4
    NUM_N_BLOCKS = tl.cdiv(N, 64)

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # Tail block that only zeros the padded scales when M < 128.
    if M_PER_BLOCK != 128 and pid_m * M_PER_BLOCK >= M:
        tl.device_assert(pid_m == 1, "pid_m != 1 when M_PER_BLOCK != 128")
        layout_off = pid_n * NUM_ELEM_PER_LAYOUT
        offs_m = tl.arange(0, 128)[:, None]
        scale_offs = layout_off + nvfp4_scale_swizzle(offs_m)
        oob_mask = (offs_m >= M) & tl.full((4,), True, dtype=tl.int1)[None, :]
        zero_scales = tl.full([128, 4], 0, dtype=tl.float8e4nv)
        tl.store(s_ptr + scale_offs, zero_scales, mask=oob_mask)
        return

    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]

    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None

    load_offsets = offs_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptr + load_offsets, mask=mask, other=other).to(tl.float32)

    # Fused RMSNorm: per-row reciprocal factor * per-column gamma.
    rms = tl.load(rms_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    y = x * rms * w  # [M_PER_BLOCK, 64] normalized activation

    # Emit normalized activation for the adapter / residual-consuming op.
    tl.store(y_ptr + (offs_m * N + offs_n), y.to(y_ptr.dtype.element_ty), mask=mask)

    y_blocks = y.reshape(M_PER_BLOCK, 4, 16)
    block_amax = tl.max(tl.abs(y_blocks), axis=2)  # [M_PER_BLOCK, 4]

    scales = tl.div_rn(block_amax, FP4_E2M1_MAX)
    scales = tl.clamp(scales, E4M3_EPS, FP8_E4M3_MAX)
    scales = scales.to(tl.float8e4nv)

    total_scale = tl.div_rn(1.0, scales.to(tl.float32)[:, :, None])
    y_blocks = y_blocks * total_scale

    if USE_MASK:
        scale_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
        scale_mask = (offs_m < M) & (scale_offs_n < (N // 16))
        scales = tl.where(scale_mask, scales, 0.0)

    swz_m = (pid_m * M_PER_BLOCK % 128) + tl.arange(0, M_PER_BLOCK)[:, None]
    layout_off = (
        (pid_m * M_PER_BLOCK) // 128
    ) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    scale_offs = layout_off + nvfp4_scale_swizzle(swz_m)
    tl.store(s_ptr + scale_offs, scales)

    x_fp4x2 = convert_fp32_to_fp4_packed(y_blocks.reshape(M_PER_BLOCK, 32, 2).split())
    q_offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    q_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    if USE_MASK:
        q_mask = (q_offs_m < M) & (q_offs_n < N // 2)
    else:
        q_mask = None
    tl.store(q_ptr + (q_offs_m * (N // 2) + q_offs_n), x_fp4x2, mask=q_mask)


# See kernels/swiglu.py: run eager under torch.compile so the raw triton launch
# isn't traced into the compiled graph (decompose_triton_kernel_wrapper_functional).
@torch.compiler.disable
def fused_rmsnorm_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rms: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm(x) * weight, returning (normalized bf16, packed fp4, swizzled scales).

    Single-level (activation) NVFP4: global_scale is implicitly 1.0. The fp4/scale
    outputs feed ``torch._scaled_mm`` directly (TN layout, contraction %32).

    ``rms`` (the per-row ``rsqrt(mean(x^2)+eps)``) may be passed in when the caller
    already computed it (e.g. the autograd Function saves it for backward), to avoid
    a redundant full-row reduction over x.
    """
    orig_dims, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape
    assert N % 16 == 0, "K must be divisible by 16 for NVFP4 quantization"

    if rms is None:
        rms = torch.rsqrt(x2.float().pow(2).mean(-1, keepdim=True) + eps).reshape(M)
    else:
        rms = rms.reshape(M)

    num_scales = N // 16
    n_row_blocks = triton.cdiv(M, 128)
    n_col_blocks = triton.cdiv(num_scales, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    xq = x2.new_empty(M, N // 2, dtype=torch.uint8)
    scales = x2.new_empty(padded_rows, padded_cols, dtype=torch.float8_e4m3fn)
    y = x2.new_empty(M, N, dtype=x.dtype)

    M_PER_BLOCK = min(triton.next_power_of_2(M), 128)
    USE_MASK = M % M_PER_BLOCK != 0 or N % 64 != 0

    grid = (triton.cdiv(N, 64), triton.cdiv(M, M_PER_BLOCK))
    if M_PER_BLOCK != 128:
        grid = (grid[0], grid[1] + 1)

    _fused_rmsnorm_nvfp4_kernel[grid](
        x2,
        rms,
        weight,
        y,
        xq,
        scales,
        x2.stride(0),
        x2.stride(1),
        M,
        N,
        M_PER_BLOCK=M_PER_BLOCK,
        USE_MASK=USE_MASK,
    )

    return (
        y.view(*orig_dims, N),
        xq.view(torch.float4_e2m1fn_x2).view(*orig_dims, N // 2),
        scales,
    )


class _FusedRMSNormNVFP4Function(torch.autograd.Function):
    """RMSNorm forward via the fused kernel; standard RMSNorm backward.

    Forward emits the normalized bf16 activation `y` plus its NVFP4 quant; only
    `y` is differentiable. Backward is the textbook RMSNorm gradient (the quant
    artifacts are forward-only — the base linear's own backward handles dgrad).
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        x2 = x.reshape(-1, x.shape[-1])
        r = torch.rsqrt(x2.float().pow(2).mean(-1, keepdim=True) + eps)
        y, xq, xsc = fused_rmsnorm_nvfp4(x, weight, eps, rms=r.reshape(-1))
        ctx.save_for_backward(x2, weight, r)
        ctx.eps = eps
        # Stash the fused quant keyed by the output the consuming linear sees.
        # Returning FP4/e4m3 tensors from the Function trips autograd's zero-grad
        # fill (unimplemented for those dtypes), so cache instead of returning them.
        _PREQUANT_CACHE[y] = (xq.reshape(-1, xq.shape[-1]), xsc)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x2, weight, r = ctx.saved_tensors
        K = x2.shape[-1]
        g = grad_y.reshape(-1, K).float()
        xf = x2.float()
        wf = weight.float()
        gg = g * wf
        c = (gg * xf).sum(-1, keepdim=True)
        grad_x = (r * (gg - (r * r) * xf * c / K)).reshape(x2.shape)
        grad_w = None
        if ctx.needs_input_grad[1]:
            grad_w = (g * (xf * r)).sum(0).to(weight.dtype)
        return grad_x.to(grad_y.dtype).reshape(grad_y.shape), grad_w, None


class NVFP4FusedRMSNorm(nn.Module):
    """Drop-in RMSNorm that fuses the NVFP4 activation quant into the norm.

    Returns the normalized bf16 activation (autograd-tracked) and caches the fp4
    quant of that activation so a downstream NVFP4 base linear skips re-quantizing.
    Falls back to a plain RMSNorm when the feature dim isn't %16 (unquantizable).
    """

    def __init__(self, weight: nn.Parameter, eps: float, zero_centered: bool = False):
        super().__init__()
        # Reuse the original Parameter as-is to preserve PEFT's frozen/trainable
        # state (freezing is via requires_grad, not by un-Parametering).
        self.weight = weight
        self.eps = eps
        # Zero-centered gamma (Gemma / Qwen3.x: ``y = normed * (1 + weight)``) vs
        # plain (Llama: ``normed * weight``). Detected in from_norm; the effective
        # gamma is built in the autograd graph so the weight grad is correct either way.
        self.zero_centered = zero_centered
        self.quantizable = weight.shape[-1] % 16 == 0

    def _gamma(self):
        return 1.0 + self.weight if self.zero_centered else self.weight

    def forward(self, x):
        gamma = self._gamma()
        if not self.quantizable:
            xf = x.float()
            r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            return (xf * r).to(x.dtype) * gamma
        return _FusedRMSNormNVFP4Function.apply(x, gamma, self.eps)

    @classmethod
    def from_norm(cls, norm: nn.Module, eps_attr: str = "variance_epsilon"):
        eps = getattr(norm, eps_attr, None)
        if eps is None:
            eps = getattr(norm, "eps", 1e-6)
        eps = float(eps)
        # Detect the gamma convention empirically (robust to arch naming): compare
        # the real norm's output to both candidate formulas on a probe.
        w = norm.weight
        with torch.no_grad():
            x = torch.randn(8, w.shape[-1], device=w.device, dtype=w.dtype)
            normed = x.float() * torch.rsqrt(
                x.float().pow(2).mean(-1, keepdim=True) + eps
            )
            y = norm(x).float()
            e_plain = (y - normed * w.float()).abs().mean()
            e_zc = (y - normed * (1.0 + w.float())).abs().mean()
            zero_centered = bool(e_zc < e_plain)
        return cls(w, eps, zero_centered)
