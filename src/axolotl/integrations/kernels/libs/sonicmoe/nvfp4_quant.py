"""Pure-torch NVFP4 quantize/dequantize reference.

NVFP4: E2M1 4-bit codes (two per byte, low nibble first) + one E4M3 block scale
per 16 contiguous K elements + an optional fp32 per-tensor scale.
``value = code_value * block_scale * per_tensor_scale``.

Used as the numeric oracle for the fp4_cute kernel path (dequantize the exact
operands the kernel consumes, matmul in fp32) and as a checkpoint-free
quantizer for tests. Encoder rounding does not need to be
bit-identical to torchao/quack: the oracle always dequantizes the operands
actually fed to the kernel, so correctness checks are encoder-independent.

``quantize_nvfp4_merge`` / ``fake_quant_nvfp4`` are different: they are the
single merge-identity quantizer shared by merge-aware LoRA training and the
``merge-lora`` writer, and delegate to torchao so both sides are bitwise
identical to each other and to the ecosystem encoder.
"""

from __future__ import annotations

import os

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


def _normalize_pts(
    per_tensor_scale: float | torch.Tensor | None, x: torch.Tensor
) -> torch.Tensor | None:
    """fp32 pts broadcastable inside torchao's ``[d0, nblocks]`` block layout.

    Scalars stay 0-dim; a per-expert vector (3D ``x`` only) becomes ``[E, 1]``.
    """
    if per_tensor_scale is None:
        return None
    pts = torch.as_tensor(per_tensor_scale, dtype=torch.float32, device=x.device)
    if pts.numel() == 1:
        return pts.reshape(())
    assert x.dim() == 3 and pts.numel() == x.shape[0], (
        f"per_tensor_scale numel {pts.numel()} must be 1 or match dim 0 of {tuple(x.shape)}"
    )
    return pts.reshape(-1, 1)


def quantize_nvfp4_merge(
    x: torch.Tensor,
    per_tensor_scale: float | torch.Tensor | None = None,
    *,
    scale_mode: str = "fresh",
    base_block_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The merge-identity NVFP4 quantizer. ``x [..., K]`` (2D or 3D, K % 16 == 0).

    Merge-aware LoRA training fake-quant and the ``merge-lora`` writer MUST both
    go through this function so the grid trained against and the grid written to
    disk agree bitwise. Slicing dim 0/1 commutes with quantization (blocks live
    on the last dim), so a fused ``[E, 2I, H]`` call and per-projection row
    slices with the same pts produce identical bytes.

    scale_mode:
      - ``fresh``: block scales recomputed from ``x`` via torchao's
        ``nvfp4_quantize`` (bit-identical to ``NVFP4Tensor.to_nvfp4``). For
        merge-aware adapters, whose deltas the training grid already snapped.
      - ``reuse``: keep ``base_block_scale`` (e4m3), bumping only blocks whose
        amax outgrew the grid, nearest-code encode. For unprepared adapters,
        where recomputing scales would re-round every element and bury a
        sub-grid-step delta under uncorrelated noise.

    per_tensor_scale: None | scalar | per-expert ``[E]`` (3D ``x`` only).
    Returns ``(packed u8 [..., K/2], block_scale e4m3 [..., K/16])``.
    """
    assert x.dim() in (2, 3), f"expected 2D/3D weight, got {tuple(x.shape)}"
    assert x.shape[-1] % SF_VEC_SIZE == 0, (
        f"K={x.shape[-1]} not a multiple of {SF_VEC_SIZE}"
    )
    if scale_mode == "fresh":
        from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

        xc = x.contiguous()
        if xc.dtype not in (torch.bfloat16, torch.float32):
            xc = xc.float()
        pts = _normalize_pts(per_tensor_scale, xc)
        scale, packed = nvfp4_quantize(xc, SF_VEC_SIZE, pts)
        return packed, scale.view(*x.shape[:-1], x.shape[-1] // SF_VEC_SIZE)
    if scale_mode == "reuse":
        assert base_block_scale is not None, "reuse mode needs base_block_scale"
        return _quantize_nvfp4_reuse_grid(
            x, base_block_scale, 1.0 if per_tensor_scale is None else per_tensor_scale
        )
    raise ValueError(f"unknown scale_mode {scale_mode!r}")


def _quantize_nvfp4_reuse_grid(
    x: torch.Tensor,
    base_block_scale: torch.Tensor,
    pts: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original grid, fresh codes; bump only the block scales the value outgrew."""
    xf = x.float()
    if isinstance(pts, torch.Tensor):
        pts = pts.float()
    sc_f = base_block_scale.to(x.device).float()
    amax = xf.unflatten(-1, (sc_f.shape[-1], SF_VEC_SIZE)).abs().amax(-1)
    need = amax > 6.0 * sc_f * pts
    if need.any():
        sc_f = torch.where(need, (amax / (6.0 * pts)).clamp(max=E4M3_MAX), sc_f)
    sc_out = sc_f.to(base_block_scale.dtype)
    denom = (
        (sc_out.float() * pts).repeat_interleave(SF_VEC_SIZE, dim=-1).clamp_min(1e-30)
    )
    lut = torch.tensor(FP4_CODE_VALUES, dtype=torch.float32, device=x.device)
    idx = ((xf / denom).unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)
    packed = idx[..., 0::2] | (idx[..., 1::2] << 4)
    return packed, sc_out


def fake_quant_nvfp4_dispatch(
    x: torch.Tensor,
    per_tensor_scale: float | torch.Tensor | None = None,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """``fake_quant_nvfp4`` via the fused triton kernel on CUDA (bitwise-equal),
    torchao reference otherwise. ``AXOLOTL_SONICMOE_MERGE_AWARE_KERNEL=0`` is
    the kill switch."""
    if x.is_cuda and os.environ.get("AXOLOTL_SONICMOE_MERGE_AWARE_KERNEL") != "0":
        from .triton_nvfp4 import fake_quant_nvfp4_triton, triton_available

        if triton_available():
            pts = per_tensor_scale
            if pts is not None and not isinstance(pts, torch.Tensor):
                pts = torch.as_tensor(pts, dtype=torch.float32, device=x.device)
            return fake_quant_nvfp4_triton(x, pts, inplace=inplace)
    return fake_quant_nvfp4(x, per_tensor_scale)


def fake_quant_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Roundtrip through the merge quantizer (fresh scales) and torchao's own
    dequantize: the exact weight values the merged checkpoint will load as.
    Differentiate via STE: ``x + (fake_quant_nvfp4(x, pts) - x).detach()``."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    packed, scale = quantize_nvfp4_merge(x, per_tensor_scale, scale_mode="fresh")
    pts = _normalize_pts(per_tensor_scale, x)
    if pts is not None:
        # dimensioned (never 0-dim), like the loader's fused [E,1,1] pts: a 0-dim pts
        # loses the promotion to fp32 in get_hp_scales (bf16 * 0-dim f32 stays bf16),
        # rounding the scale product differently than the fused-load dequant
        pts = (
            pts.reshape([-1] + [1] * (x.dim() - 1)) if pts.dim() else pts.reshape(1, 1)
        )
    # orig_dtype bf16 = the loader's construction, so scale math matches it bitwise
    nv = NVFP4Tensor(packed, scale, SF_VEC_SIZE, torch.bfloat16, per_tensor_scale=pts)
    return nv.dequantize(x.dtype)
