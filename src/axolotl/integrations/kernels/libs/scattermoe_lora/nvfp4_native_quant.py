"""Arch-gated NVIDIA-native Blackwell hardware-cvt NVFP4 quantizer; byte-identical to torchao, falls back off-Blackwell."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - triton optional
    _TRITON_OK = False


NVFP4_BLOCK_SIZE = 16
_MIN_CUDA = (12, 8)  # sm_120a/sm_100a e2m1x2 cvt needs CUDA >= 12.8
_BLACKWELL_MAJORS = (10, 12)  # sm100a (B100/B200) + sm120a (RTX 50xx)
_F4_E2M1_MAX = 6.0
_F8E4M3_MAX = 448.0
_E4M3_EPS = float(
    torch.finfo(torch.float8_e4m3fn).tiny
)  # 2**-6 floor; MUST equal torchao E4M3_EPS for byte-parity


def is_blackwell_native_nvfp4_available(
    device: torch.device | int | None = None,
) -> bool:
    """True iff this machine can run the native encode: CUDA Blackwell (sm120a/sm100a), CUDA>=12.8, Triton."""
    if not _TRITON_OK:
        return False
    if not torch.cuda.is_available():
        return False

    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:  # pragma: no cover - defensive
        return False
    if major not in _BLACKWELL_MAJORS:
        return False

    cuda_ver = getattr(torch.version, "cuda", None)
    if cuda_ver is None:
        return False
    try:
        parts = tuple(int(p) for p in cuda_ver.split(".")[:2])
    except ValueError:  # pragma: no cover - defensive
        return False
    if parts < _MIN_CUDA:
        return False

    return True


def _native_k_supported(k: int) -> bool:
    """K must be a power-of-two multiple of 16 (the kernel's tl.arange(0,K) extent); D192 falls back."""
    return k % NVFP4_BLOCK_SIZE == 0 and k > 0 and (k & (k - 1)) == 0


# Triton kernels vendored from AtlasAttention nvfp4_quant.py, floor set to torchao E4M3_EPS=2**-6 for byte-parity.
if _TRITON_OK:
    _F4_MAX_C = tl.constexpr(_F4_E2M1_MAX)
    _F8E4M3_MAX_C = tl.constexpr(_F8E4M3_MAX)
    _E4M3_FLOOR_C = tl.constexpr(_E4M3_EPS)

    @triton.jit
    def _cvt_fp32_to_fp4x2_packed(x_pairs):
        """Pack 4 fp32 pairs -> uint32 via hardware cvt; first-of-pair in the low nibble (torchao convention)."""
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 byte0, byte1, byte2, byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            constraints=("=r,r,r,r,r,r,r,r,r"),
            args=x_pairs,
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    @triton.jit
    def _quant_per_block_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        R,
        K: tl.constexpr,
        BR: tl.constexpr,
    ):
        """Fused single-level encode: amax -> e4m3 block scale -> reciprocal-normalize -> hardware-cvt pack.

        Loads the even (low-nibble) and odd (high-nibble) columns of every block as two separate
        strided tiles rather than reshaping the [BR,K] load into [BR,NG,16]. ``tl.reshape`` on the
        loaded tile miscompiles at certain tile widths in this Triton (e.g. exactly 1024 bytes/row:
        K=256 fp32 / K=512 bf16), silently reducing the wrong 16-element group and corrupting both
        scale and qdata. The reshape-free strided form is correct at every shape and dtype.
        """
        pid = tl.program_id(0)
        rows = pid * BR + tl.arange(0, BR)
        rmask = rows < R
        NG: tl.constexpr = K // 16
        g = tl.arange(0, NG)
        e = tl.arange(0, 8)
        # [BR,NG,8] offsets of the even (low-nibble) columns; +1 yields the odd (high-nibble) ones.
        base = rows[:, None, None] * K + g[None, :, None] * 16 + 2 * e[None, None, :]
        m3 = rmask[:, None, None]
        lo = tl.load(x_ptr + base, mask=m3, other=0.0).to(tl.float32)
        hi = tl.load(x_ptr + base + 1, mask=m3, other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(lo), axis=2), tl.max(tl.abs(hi), axis=2))
        sc = tl.clamp(amax / _F4_MAX_C, _E4M3_FLOOR_C, _F8E4M3_MAX_C).to(tl.float8e4nv)
        # x * (1.0 / block_scale_fp32): match torchao reciprocal-multiply numerics.
        rn = (1.0 / sc.to(tl.float32))[:, :, None]
        nlo = tl.clamp(lo * rn, -_F4_MAX_C, _F4_MAX_C)
        nhi = tl.clamp(hi * rn, -_F4_MAX_C, _F4_MAX_C)
        q = _cvt_fp32_to_fp4x2_packed((nlo, nhi)).reshape(BR, K // 2)
        pcols = tl.arange(0, K // 2)
        tl.store(
            packed_ptr + rows[:, None] * (K // 2) + pcols[None, :],
            q,
            mask=rmask[:, None],
        )
        gcols = tl.arange(0, NG)
        tl.store(
            scale_ptr + rows[:, None] * NG + gcols[None, :],
            sc,
            mask=rmask[:, None],
        )

    @triton.jit
    def _pack_only_kernel(
        x_ptr,
        packed_ptr,
        R,
        K: tl.constexpr,
        BR: tl.constexpr,
    ):
        """Pack already-scaled+clamped fp32 in [-6,6] via hardware cvt (the two-level path's accelerated step).

        Reshape-free strided even/odd loads — see ``_quant_per_block_kernel`` for why ``tl.reshape``
        on the loaded tile is avoided.
        """
        pid = tl.program_id(0)
        rows = pid * BR + tl.arange(0, BR)
        rmask = rows < R
        NG: tl.constexpr = K // 16
        g = tl.arange(0, NG)
        e = tl.arange(0, 8)
        base = rows[:, None, None] * K + g[None, :, None] * 16 + 2 * e[None, None, :]
        m3 = rmask[:, None, None]
        lo = tl.load(x_ptr + base, mask=m3, other=0.0).to(tl.float32)
        hi = tl.load(x_ptr + base + 1, mask=m3, other=0.0).to(tl.float32)
        q = _cvt_fp32_to_fp4x2_packed((lo, hi)).reshape(BR, K // 2)
        pcols = tl.arange(0, K // 2)
        tl.store(
            packed_ptr + rows[:, None] * (K // 2) + pcols[None, :],
            q,
            mask=rmask[:, None],
        )


_DEFAULT_BR = (
    8  # rows/program; amortizes launch and keeps the per-block reduction in registers
)


def _native_block_quant(
    data_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused single-level encode of 2D (R,K) -> (packed uint8 (R,K//2), linear e4m3 scale (R,K//16)).

    Accepts bf16 or fp32: the kernel widens to fp32 in-register (bf16->fp32 is exact), so feeding
    bf16 straight in is byte-identical to pre-casting while skipping an fp32 [R,K] transient — that
    extra read+write otherwise ~halves throughput on this bandwidth-bound cast.
    """
    assert _TRITON_OK, "triton unavailable; cannot use the fused NVFP4 quant"
    assert data_2d.dtype in (torch.float32, torch.bfloat16) and data_2d.dim() == 2
    R, K = data_2d.shape
    assert _native_k_supported(K), f"K={K} not supported by the native kernel"
    xf = data_2d.contiguous()
    packed = torch.empty(R, K // 2, dtype=torch.uint8, device=xf.device)
    scale = torch.empty(
        R, K // NVFP4_BLOCK_SIZE, dtype=torch.float8_e4m3fn, device=xf.device
    )
    BR = _DEFAULT_BR
    grid = (triton.cdiv(R, BR),)
    _quant_per_block_kernel[grid](xf, packed, scale, R, K=K, BR=BR)
    return packed, scale


def _native_pack(data_scaled_fp32_2d: torch.Tensor) -> torch.Tensor:
    """Pack already-scaled+clamped fp32 data (R, K) -> uint8 (R, K//2)."""
    assert _TRITON_OK, "triton unavailable"
    assert data_scaled_fp32_2d.dtype == torch.float32 and data_scaled_fp32_2d.dim() == 2
    R, K = data_scaled_fp32_2d.shape
    assert _native_k_supported(K), f"K={K} not supported by the native kernel"
    xf = data_scaled_fp32_2d.contiguous()
    packed = torch.empty(R, K // 2, dtype=torch.uint8, device=xf.device)
    BR = _DEFAULT_BR
    grid = (triton.cdiv(R, BR),)
    _pack_only_kernel[grid](xf, packed, R, K=K, BR=BR)
    return packed


def _native_nvfp4_quantize(
    data_hp: torch.Tensor,
    block_size: int,
    per_tensor_scale: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-identical replacement for torchao nvfp4_quantize; returns (linear e4m3 scales, packed FP4)."""
    assert data_hp.dtype in (torch.bfloat16, torch.float), (
        f"{data_hp.dtype} not supported"
    )
    assert data_hp.size(-1) % block_size == 0, "K dim must be divisible by block_size"
    assert data_hp.is_contiguous(), "Only support contiguous data for now"
    assert block_size == NVFP4_BLOCK_SIZE, "NVFP4 requires block_size=16"

    orig_shape = data_hp.shape
    K = orig_shape[-1]
    R = 1
    for d in orig_shape[:-1]:
        R *= int(d)

    if per_tensor_scale is None:
        # Single level: the fully-fused kernel does everything (byte-identical). Feed the native
        # dtype straight in — the kernel widens to fp32 in-register, so a prior .float() only burns
        # bandwidth materializing an fp32 transient (~2.4x slower end-to-end on bf16 weights).
        data_2d = data_hp.reshape(R, K).contiguous()
        qdata_2d, scale_2d = _native_block_quant(data_2d)
        out_scales = scale_2d.reshape(R, K // block_size)
        data_lp = qdata_2d.reshape(R, K // 2)
        return out_scales, data_lp

    # Two-level: mirror torchao's scale math exactly; reshape (E,-1,block) lets a per-expert [E,1,1] pts broadcast.
    data_hp_blk = data_hp.float().reshape(orig_shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(data_hp_blk), dim=-1)
    block_scale = max_abs / _F4_E2M1_MAX
    block_scale_fp32 = block_scale.to(torch.float32)

    pts = per_tensor_scale
    if len(pts.shape) == 3:
        pts = pts.squeeze(-1)

    scaled_block_scales = block_scale_fp32 / pts
    scaled_block_scales_fp8 = torch.clamp(
        scaled_block_scales, min=_E4M3_EPS, max=_F8E4M3_MAX
    ).to(torch.float8_e4m3fn)
    scaled_block_scales_fp32 = scaled_block_scales_fp8.to(torch.float32)
    reciprocal_scale = (1.0 / pts) / scaled_block_scales_fp32
    data_scaled = data_hp_blk * reciprocal_scale.unsqueeze(-1)
    data_scaled = torch.clamp(data_scaled, -_F4_E2M1_MAX, _F4_E2M1_MAX)
    data_scaled = data_scaled.reshape(
        R, K
    ).contiguous()  # row-major: torchao's pack element order

    data_lp = _native_pack(data_scaled).reshape(R, K // 2)
    out_scales = scaled_block_scales_fp8.reshape(R, K // block_size)
    return out_scales, data_lp


_ORIG_TO_NVFP4: Callable[..., Any] | None = None


def _native_to_nvfp4(
    data_hp: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    per_tensor_scale: Optional[torch.Tensor] = None,
    act_per_tensor_scale: Optional[torch.Tensor] = None,
    is_swizzled_scales: bool = False,
    use_triton_kernel: bool = False,
    act_quant_kwargs=None,
):
    """Byte-identical native replacement for NVFP4Tensor.to_nvfp4; delegates to torchao when the gate fails."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
    )
    from torchao.prototype.mx_formats.utils import to_blocked

    K = data_hp.shape[-1]
    can_native = (
        is_blackwell_native_nvfp4_available(data_hp.device)
        and block_size == NVFP4_BLOCK_SIZE
        and len(data_hp.shape) in (2, 3)
        and data_hp.dtype in (torch.bfloat16, torch.float32)
        and K % NVFP4_BLOCK_SIZE == 0
        and _native_k_supported(K)
        and not use_triton_kernel  # torchao mslk triton path is separate
        and data_hp.is_cuda
        and data_hp.is_contiguous()  # _native_nvfp4_quantize asserts contiguity
    )

    if not can_native:
        orig_to_nvfp4 = _ORIG_TO_NVFP4
        if orig_to_nvfp4 is None:
            raise RuntimeError("native NVFP4 shim called before install_native_nvfp4()")
        return orig_to_nvfp4(
            data_hp,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            use_triton_kernel=use_triton_kernel,
            act_quant_kwargs=act_quant_kwargs,
        )

    leading_dims, M = data_hp.shape[:-2], data_hp.shape[-2]

    blockwise_scales, data_lp = _native_nvfp4_quantize(
        data_hp, block_size, per_tensor_scale
    )

    if is_swizzled_scales:
        scale_shape = (math.prod(leading_dims) * M, K // block_size)
        blockwise_scales = to_blocked(blockwise_scales.view(scale_shape)).flatten()
        scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
    else:
        scale_M, scale_K = M, K // block_size
    blockwise_scales = blockwise_scales.view(*leading_dims, scale_M, scale_K)
    data_lp = data_lp.view(*leading_dims, M, K // 2)

    return NVFP4Tensor(
        data_lp,
        blockwise_scales,
        block_size,
        data_hp.dtype,
        per_tensor_scale,
        act_per_tensor_scale,
        is_swizzled_scales,
        use_triton_kernel,
        act_quant_kwargs,
    )


def install_native_nvfp4() -> bool:
    """Idempotently monkeypatch NVFP4Tensor.to_nvfp4 with the native path (safe: delegates to torchao off-gate)."""
    global _ORIG_TO_NVFP4
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except Exception:  # pragma: no cover
        return False
    if _ORIG_TO_NVFP4 is not None:
        return True
    _ORIG_TO_NVFP4 = (
        NVFP4Tensor.to_nvfp4.__func__
        if hasattr(NVFP4Tensor.to_nvfp4, "__func__")
        else NVFP4Tensor.to_nvfp4
    )
    NVFP4Tensor.to_nvfp4 = staticmethod(_native_to_nvfp4)
    return True


def uninstall_native_nvfp4() -> bool:
    """Restore the original torchao to_nvfp4. Idempotent."""
    global _ORIG_TO_NVFP4
    if _ORIG_TO_NVFP4 is None:
        return False
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except Exception:  # pragma: no cover
        return False
    NVFP4Tensor.to_nvfp4 = staticmethod(_ORIG_TO_NVFP4)
    _ORIG_TO_NVFP4 = None
    return True
