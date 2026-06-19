# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""dequant_nvfp4_fp8_triton produces the NVFP4 weights as fp8 (e4m3) for the fp8-read path.

The NVFP4 + LoRA fp8-read fast path (workstation Blackwell / sm_120) materializes the frozen
experts as fp8 instead of bf16, halving the transient weight, and the base grouped GEMM upcasts
to bf16 in-register. The fp8 dequant runs the same one-pass NVFP4 kernel with an fp8 store, so it
must equal the validated linear fold computed in fp32 and rounded once to fp8 (a single
fp32->fp8 round, not fp32->bf16->fp8). ``scale_mode="none"`` (the shipped path, no per-expert
scale) carries no reciprocal; ``scale_mode="perexpert"`` returns a reciprocal that reconstructs
the weight after the kernel's per-expert pow2 scale.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor

from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant_kernel import (  # noqa: E402
    _NVFP4_E2M1_LUT,
    dequant_nvfp4_fp8_triton,
    dequant_nvfp4_full_triton,
)

DEV = "cuda"
E, ROWS, COLS = 8, 256, 512  # [E, N, K]; K power-of-2 (kernel requirement)
F8 = torch.float8_e4m3fn


def _validated_fold_fp32(nv):
    """The fused path's exact linear fold in fp32: LUT[nibble] * (scale_e4m3 * per_tensor)."""
    qd = nv.qdata
    scale_f = nv.scale.to(torch.float32)
    pts = getattr(nv, "per_tensor_scale", None)
    if pts is not None:
        scale_f = scale_f * pts.to(torch.float32)
    lut = torch.tensor(_NVFP4_E2M1_LUT, device=qd.device, dtype=torch.float32)
    lo = lut[(qd & 0xF).long()]
    hi = lut[((qd >> 4) & 0xF).long()]
    val = torch.stack([lo, hi], dim=-1).reshape(*qd.shape[:-1], qd.shape[-1] * 2)
    return val * scale_f.repeat_interleave(16, dim=-1)


def _build(pts_mode):
    g = torch.Generator(device=DEV).manual_seed(0)
    t = torch.randn(E, ROWS, COLS, device=DEV, dtype=torch.bfloat16, generator=g) * 0.1
    if pts_mode == "none":
        return NVFP4Tensor.to_nvfp4(t, block_size=16)
    pts = (t.abs().amax() / (6.0 * 448.0)).to(torch.float32)  # mirrors the loader's gpts
    return NVFP4Tensor.to_nvfp4(t, block_size=16, per_tensor_scale=pts)


@pytest.mark.parametrize("pts_mode", ["none", "scalar"])
def test_fp8_noscale_matches_fold_cast(pts_mode):
    """scale_mode="none" == the validated fp32 fold rounded once to fp8 (no reciprocal)."""
    nv = _build(pts_mode)
    out, inv_s = dequant_nvfp4_fp8_triton(nv, scale_mode="none")
    assert out.dtype == F8 and out.shape == (E, ROWS, COLS)
    assert inv_s is None
    ref = _validated_fold_fp32(nv).to(F8)
    assert torch.equal(out, ref), (
        f"{pts_mode}: fp8 dequant != fold-cast-to-fp8 "
        f"(max |Δ| {(out.float() - ref.float()).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("pts_mode", ["none", "scalar"])
def test_fp8_perexpert_reconstructs(pts_mode):
    """scale_mode="perexpert": fp8(S·W) · (1/S) reconstructs the weight; pow2 S keeps 1/S exact,
    and using the e4m3 normal range is no worse than the no-scale fp8 vs the bf16 reference."""
    nv = _build(pts_mode)
    out, inv_s = dequant_nvfp4_fp8_triton(nv, scale_mode="perexpert")
    assert out.dtype == F8 and inv_s is not None and inv_s.numel() == E
    recon = out.float() * inv_s.reshape(E, 1, 1).float()
    ref = dequant_nvfp4_full_triton(nv, torch.bfloat16).float()  # the bf16-read weight
    rel = (recon - ref).norm() / ref.norm()
    assert torch.isfinite(recon).all() and rel < 0.05, f"{pts_mode}: relL2 {rel:.3e}"
