"""Pure-torch NVFP4 quantize/dequantize reference.

NVFP4: E2M1 4-bit codes (two per byte, low nibble first) + one E4M3 block scale
per 16 contiguous K elements + an optional fp32 per-tensor scale.
``value = code_value * block_scale * per_tensor_scale``.

Used as the numeric oracle for the fp4_cute kernel path (dequantize the exact
operands the kernel consumes, matmul in fp32) and as a checkpoint-free
quantizer for tests. Encoder rounding does not need to be
bit-identical to torchao/quack: the oracle always dequantizes the operands
actually fed to the kernel, so correctness checks are encoder-independent.
"""

from __future__ import annotations

import torch

SF_VEC_SIZE = 16
E4M3_MAX = 448.0

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between consecutive magnitudes, for nearest-value bucketing.
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Full 16-entry table indexed by the 4-bit code (bit 3 = sign).
FP4_CODE_VALUES = tuple(E2M1_MAGNITUDES) + tuple(-m for m in E2M1_MAGNITUDES)


def fp4_code_to_value(codes: torch.Tensor) -> torch.Tensor:
    """Map 4-bit E2M1 codes in [0, 16) to their float32 values."""
    table = torch.tensor(FP4_CODE_VALUES, dtype=torch.float32, device=codes.device)
    return table[codes.long()]


def encode_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round float values to the nearest E2M1 code (uint8 in [0, 16))."""
    bounds = torch.tensor(_E2M1_BOUNDS, dtype=torch.float32, device=x.device)
    mag_idx = torch.bucketize(x.abs().float(), bounds)
    sign = (x < 0).to(torch.uint8) << 3
    return mag_idx.to(torch.uint8) | sign


def pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes into bytes along the last dim, low nibble first."""
    assert codes.dtype == torch.uint8 and codes.shape[-1] % 2 == 0
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return lo | (hi << 4)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_fp4_codes`."""
    assert packed.dtype == torch.uint8
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


def quantize_nvfp4_ref(
    x: torch.Tensor,
    per_tensor_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``x [..., K]`` (K % 16 == 0) to NVFP4.

    per_tensor_scale: None -> 1.0; "auto" is intentionally not supported here,
    pass an explicit value when exercising the two-level scheme.

    Returns ``(packed u8 [..., K/2], block_scale e4m3 [..., K/16], pts f32 scalar)``.
    """
    assert x.shape[-1] % SF_VEC_SIZE == 0, (
        f"K={x.shape[-1]} not a multiple of {SF_VEC_SIZE}"
    )
    xf = x.float()
    if per_tensor_scale is None:
        pts = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    else:
        pts = torch.as_tensor(per_tensor_scale, dtype=torch.float32, device=x.device)

    blocks = xf.reshape(*xf.shape[:-1], -1, SF_VEC_SIZE)
    amax = blocks.abs().amax(dim=-1)
    scale_f32 = (amax / (E2M1_MAGNITUDES[-1] * pts)).clamp(max=E4M3_MAX)
    scale_e4m3 = scale_f32.to(torch.float8_e4m3fn)
    # Encode against the STORED (rounded) scale, like real quantizers.
    scale_dec = scale_e4m3.float() * pts
    denom = torch.where(scale_dec == 0, torch.ones_like(scale_dec), scale_dec)
    q = blocks / denom.unsqueeze(-1)
    codes = encode_e2m1(q).reshape(xf.shape)
    return pack_fp4_codes(codes), scale_e4m3, pts


def dequantize_nvfp4_ref(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    per_tensor_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize to float32. ``packed [..., K/2]``, ``block_scale [..., K/16]``."""
    codes = unpack_fp4_codes(packed)
    vals = fp4_code_to_value(codes)
    scale = block_scale.float().repeat_interleave(SF_VEC_SIZE, dim=-1)
    out = vals * scale
    if per_tensor_scale is not None:
        out = out * torch.as_tensor(
            per_tensor_scale, dtype=torch.float32, device=out.device
        )
    return out


def simulate_nvfp4_quant(
    x: torch.Tensor, per_tensor_scale: float | torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize-dequantize roundtrip: the exact value grid the tensor core sees."""
    packed, scale, pts = quantize_nvfp4_ref(x, per_tensor_scale)
    return dequantize_nvfp4_ref(packed, scale, pts)
