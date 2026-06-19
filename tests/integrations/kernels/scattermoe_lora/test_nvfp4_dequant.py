# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""dequant_nvfp4_full_triton matches the validated NVFP4 linear fold, bit-exactly.

The NVFP4 + LoRA fast path dequantizes the frozen experts to bf16 once via a custom Triton
kernel (the eager ``NVFP4Tensor.dequantize()`` materializes ~145 GB of unfused intermediates
and ``torch.compile`` miscompiles the subclass). Correctness bar = the SAME linear fold the
fused path uses (``selective_nvfp4_weights_fwd``: ``LUT[nibble] * (scale_e4m3.float() *
per_tensor.float())``), which the model's ``test_nvfp4_experts_forward`` already validates.

Covers the three ``per_tensor_scale`` shapes the loader / FSDP carry: None, a shared scalar
(what the loader emits), and a per-expert ``[E, 1, 1]``. The in-kernel fp32 fold must be
bit-exact against the reference in every case (a post-store bf16 multiply double-rounds, and a
bare ``[E]`` per-expert scale previously crashed on a broadcast mismatch).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor

from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant_kernel import (  # noqa: E402
    _NVFP4_E2M1_LUT,
    dequant_nvfp4_full_triton,
)

DEV = "cuda"
E, ROWS, COLS = 8, 256, 512  # [E, N, K]; K power-of-2 (kernel requirement)


def _validated_fold(nv, dtype=torch.float32):
    """The fused path's exact linear fold (mx_weights.selective_nvfp4_weights_fwd):
    value = LUT[nibble] * (scale_e4m3.float() * per_tensor.float())."""
    qd = nv.qdata
    scale_f = nv.scale.to(torch.float32)
    pts = getattr(nv, "per_tensor_scale", None)
    if pts is not None:
        scale_f = scale_f * pts.to(torch.float32)
    lut = torch.tensor(_NVFP4_E2M1_LUT, device=qd.device, dtype=torch.float32)
    lo = lut[(qd & 0xF).long()]
    hi = lut[((qd >> 4) & 0xF).long()]
    val = torch.stack([lo, hi], dim=-1).reshape(*qd.shape[:-1], qd.shape[-1] * 2)
    return (val * scale_f.repeat_interleave(16, dim=-1)).to(dtype)


def _build(pts_mode):
    g = torch.Generator(device=DEV).manual_seed(0)
    t = torch.randn(E, ROWS, COLS, device=DEV, dtype=torch.bfloat16, generator=g) * 0.1
    if pts_mode == "none":
        return NVFP4Tensor.to_nvfp4(t, block_size=16)
    if pts_mode == "scalar":
        pts = (t.abs().amax() / (6.0 * 448.0)).to(torch.float32)  # mirrors the loader's gpts
        return NVFP4Tensor.to_nvfp4(t, block_size=16, per_tensor_scale=pts)
    # per-expert [E,1,1]: a valid NVFP4 tensor carrying a per-expert scale (FSDP carry shape)
    nv = NVFP4Tensor.to_nvfp4(t, block_size=16)
    nv.per_tensor_scale = (
        torch.rand(E, 1, 1, device=DEV, generator=g).to(torch.float32) + 0.5
    )
    return nv


@pytest.mark.parametrize("pts_mode", ["none", "scalar", "perexpert"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dequant_matches_validated_fold(pts_mode, dtype):
    nv = _build(pts_mode)
    ours = dequant_nvfp4_full_triton(nv, dtype)
    ref = _validated_fold(nv, dtype)
    assert ours.shape == ref.shape == (E, ROWS, COLS)
    # The fp32 fold is computed identically (LUT * scale * pts), so the kernel must match the
    # reference exactly at fp32 and after the single bf16 round.
    assert torch.equal(ours, ref), (
        f"{pts_mode}/{dtype}: max rel "
        f"{(ours - ref).float().abs().max().item() / max(ref.float().abs().max().item(), 1e-9):.2e}"
    )
