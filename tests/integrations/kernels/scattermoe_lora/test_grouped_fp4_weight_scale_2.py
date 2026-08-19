# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Non-unit per-expert ``weight_scale_2`` (per_tensor_scale) fwd+bwd parity (C1 / A5 / D-A1).

The real DSV4 NVFP4 checkpoint carries a non-unit per-expert ``per_tensor_scale`` (weight_scale_2),
a second scale applied on top of the E4M3 per-16-block scale. The CUTLASS forward cannot fold it
(set_weights drops it while the backward applies it -> wrong forward + grad mismatch), so the
backend selection routes a non-unit per_tensor_scale to Marlin or DeepGEMM (both fold it into the
weight) and hard-errors if neither is available. This test pins that the Marlin/DeepGEMM forward +
the bf16-dequant backward both honor a per-expert weight_scale_2, matching a bf16 oracle whose
weights are dequantized WITH the same per-expert per_tensor_scale.

Why this is NOT pure-dequant-fallback testable: the chunked-dequant fallback (``moe_grouped_backend
= 'dequant'``) still uses the CUTLASS *forward* engine (set_weights), which drops weight_scale_2 -
so a fused backend (Marlin sm80+ / DeepGEMM sm90+) that folds it is REQUIRED for the forward. On the
sm89 L40S CI that is Marlin, so this test EXECUTES there. On a GPU with neither, it skips.

Gate: CUDA + a weight_scale_2-folding forward backend (Marlin sm80+ / DeepGEMM sm90+). NOT
Blackwell-gated (Marlin covers sm89), but it IS hardware-only - there is no CPU path.
"""

from __future__ import annotations

import pytest
import torch

from .test_grouped_fp4_train import (
    DEV,
    SHAPE_IDS,
    SHAPES,
    _bf16_oracle,
    _cos,
    _fp4_codebook,
)

_IS_CUDA = torch.cuda.is_available()


def _ws2_backend_available() -> bool:
    """True iff a forward backend that FOLDS weight_scale_2 (Marlin sm80+ / DeepGEMM sm90+) is
    available. CUTLASS does NOT fold it, so its presence alone does not qualify."""
    if not _IS_CUDA:
        return False
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
            _backend_available,
        )

        return _backend_available("marlin") or _backend_available("deepgemm")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ws2_backend_available(),
    reason="non-unit weight_scale_2 needs a folding forward backend (Marlin sm80+ / DeepGEMM "
    "sm90+); CUTLASS drops it and there is no CPU path - hardware-only",
)


def _quantize_nvfp4_ws2(W: torch.Tensor, pt_e: torch.Tensor):
    """Quantize bf16 W[E,N,K] -> NVFP4Tensor carrying a per-expert per_tensor_scale ``pt_e`` [E].

    Factors ``pt_e`` out of the weight before the E4M3-block quantization and stores it as
    per_tensor_scale, so ``nvfp4_dequant_bf16(qdata, scale, pt_e)`` reconstructs the original W:
    quantize W/pt_e into the block path, then scale back by pt_e at dequant. This reproduces the
    real DSV4 layout (a non-unit weight_scale_2) rather than the unit-scale synthetic in the
    sibling correctness module.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    dev = W.device
    cb = _fp4_codebook(dev)
    E, N, K = W.shape
    Wf = (W.float() / pt_e.view(E, 1, 1)).reshape(E, N, K // 16, 16)
    amax = Wf.abs().amax(-1).clamp_min(1e-6)
    scale_e4m3 = (amax / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale_e4m3.float().clamp_min(1e-9)
    nib = (
        ((Wf / scale_f.unsqueeze(-1)).unsqueeze(-1) - cb.view(1, 1, 1, 1, 16))
        .abs()
        .argmin(-1)
    )
    nib = nib.to(torch.uint8).reshape(E, N, K)
    packed = (nib[..., 0::2] & 0xF) | ((nib[..., 1::2] & 0xF) << 4)
    return NVFP4Tensor(
        packed.contiguous(),
        scale_e4m3.contiguous(),
        block_size=16,
        orig_dtype=torch.bfloat16,
        per_tensor_scale=pt_e.contiguous(),
    )


@pytest.mark.parametrize("cfg", SHAPES, ids=SHAPE_IDS)
def test_grouped_fp4_ws2_fwd_bwd_matches_oracle(cfg):
    """A non-unit per-expert weight_scale_2 forward (cosine >= 0.97) and backward (cosine >= 0.95)
    match a bf16 oracle dequantized WITH the same per-expert per_tensor_scale.

    Exercises the Marlin/DeepGEMM weight_scale_2 fold in the forward and the per-expert ``_pt``
    application in the bf16-dequant backward (prefer_fp8_dx=False). Skips on a GPU that has neither
    a Marlin nor a DeepGEMM backend (CUTLASS-only / no fused backend) since CUTLASS cannot fold it.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        nvfp4_dequant_bf16,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        _has_nonunit_pt,
        grouped_fp4_moe_train,
    )

    E, H, I, topk, N, r = cfg["E"], cfg["H"], cfg["I"], cfg["topk"], cfg["N"], cfg["r"]
    s = cfg["scaling"]
    twoI = 2 * I
    torch.manual_seed(11)

    # Non-unit per-expert weight_scale_2 (clearly != 1 to make the fold load-bearing).
    pt_gu = (0.5 + torch.rand(E, device=DEV)).to(torch.float32)
    pt_dn = (0.5 + torch.rand(E, device=DEV)).to(torch.float32)

    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    gu_nv = _quantize_nvfp4_ws2(Wgu, pt_gu)
    dn_nv = _quantize_nvfp4_ws2(Wdn, pt_dn)

    # Sanity: the construction really carries a non-unit weight_scale_2 (else the test is vacuous).
    assert _has_nonunit_pt(gu_nv, dn_nv), (
        "weight_scale_2 construction produced a unit pt"
    )

    # Oracle weights: dequantize WITH the per-expert per_tensor_scale.
    Wgu_b = nvfp4_dequant_bf16(gu_nv.qdata, gu_nv.scale, pt_gu)
    Wdn_b = nvfp4_dequant_bf16(dn_nv.qdata, dn_nv.scale, pt_dn)

    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.stack([torch.randperm(E, device=DEV)[:topk] for _ in range(N)])
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1).to(torch.bfloat16)

    def mk(*shape):
        return torch.randn(*shape, device=DEV, dtype=torch.bfloat16) * 0.02

    Agu, Bgu, Adn, Bdn = mk(r * E, H), mk(twoI, r * E), mk(r * E, I), mk(H, r * E)

    hk = hidden.detach().clone().requires_grad_(True)
    Agk = Agu.clone().requires_grad_(True)
    Bgk = Bgu.clone().requires_grad_(True)
    Adk = Adn.clone().requires_grad_(True)
    Bdk = Bdn.clone().requires_grad_(True)

    out = grouped_fp4_moe_train(
        hk,
        idx,
        wts,
        gu_nv,
        dn_nv,
        (Agk, Bgk, s),
        (Adk, Bdk, s),
        cfg["limit"],
        "nvfp4",
        act_type=cfg["act_type"],
        mxfp4_cache={},
        prefer_fp8_dx=False,
    )
    assert torch.isfinite(out).all(), f"{cfg['name']}: ws2 fwd has NaN/inf"
    out.float().pow(2).mean().backward()

    hr = hidden.detach().clone().requires_grad_(True)
    Agr = Agu.clone().requires_grad_(True)
    Bgr = Bgu.clone().requires_grad_(True)
    Adr = Adn.clone().requires_grad_(True)
    Bdr = Bdn.clone().requires_grad_(True)
    ref = _bf16_oracle(
        hr, idx, wts, Wgu_b, Wdn_b, Agr, Bgr, Adr, Bdr, cfg["act_type"], cfg["limit"], s
    )
    cos_fwd = _cos(out, ref)
    assert cos_fwd > 0.97, f"{cfg['name']}: ws2 fwd cosine {cos_fwd:.4f} < 0.97"

    ref.pow(2).mean().backward()
    for name, gk, gr in [
        ("d_hidden", hk.grad, hr.grad),
        ("d_Agu", Agk.grad, Agr.grad),
        ("d_Adn", Adk.grad, Adr.grad),
    ]:
        assert gk is not None, f"{cfg['name']}: ws2 {name} grad is None"
        assert torch.isfinite(gk).all(), f"{cfg['name']}: ws2 {name} has NaN/inf"
        cos = _cos(gk, gr)
        assert cos > 0.95, f"{cfg['name']}: ws2 {name} cosine {cos:.4f} < 0.95"
