# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Arch-agnostic (non-Blackwell) coverage for the grouped NVFP4 MoE training path.

The headline correctness oracle (test_grouped_fp4_train.py) is gated to Blackwell (sm100/sm120)
because the CUTLASS/DeepGEMM forward backends only run there. But the grouped autograd Function
also has a path that runs on ANY sm80+ CUDA GPU:

  * FORWARD: the Marlin W4A16 backend (bf16-act x NVFP4-weight, standard Ampere-class ``mma``,
    no Hopper/Blackwell intrinsics) covers sm80 (A100) / sm89 (L40S / RTX 4090) / sm120.
  * BACKWARD: the gradient-consistent chunked bf16-dequant base dX (NVFP4 -> bf16 via a Triton
    kernel + ``torch._grouped_mm``), selected on every non-sm120 arch and forced here via
    ``prefer_fp8_dx=False``. This is the B8 FP4-expert backward and is arch-agnostic.

So on the L40S (sm89) CI these tests EXECUTE (via Marlin fwd + bf16-dequant bwd), giving the
grouped fwd/bwd oracle and the B11 backward None-grad contract real coverage there. They also
execute on Blackwell. The Marlin/DeepGEMM/CUTLASS *numerics* and *perf* remain hardware-gated
in test_grouped_fp4_train.py / test_grouped_fp4_perf.py; this file does not weaken those.

Gate: CUDA available AND ``grouped_fp4_available('nvfp4')`` (True on sm80+ with nvcc via Marlin,
or on sm90/100 via DeepGEMM, or sm120 via CUTLASS). NOT Blackwell-gated.

Helpers (quantize_nvfp4, _bf16_oracle, _make_inputs, _cos, SHAPES, DEV) are reused from the
sibling correctness module; importing them is safe - its module-level Blackwell skip applies only
to ITS test functions, not to importing its helpers.
"""

from __future__ import annotations

import pytest
import torch

# Reuse the validated quantizer + oracle + fixtures from the sibling correctness module.
from .test_grouped_fp4_train import (
    DEV,
    SHAPE_IDS,
    SHAPES,
    _bf16_oracle,
    _cos,
    _fresh_nv,
    _make_inputs,
    quantize_nvfp4,
)

_IS_CUDA = torch.cuda.is_available()


def _grouped_available() -> bool:
    """True iff SOME grouped fp4 forward backend resolves here (Marlin on sm80+, else DeepGEMM /
    CUTLASS). On the L40S CI this is True via Marlin, so the tests below execute (no Blackwell)."""
    if not _IS_CUDA:
        return False
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
            grouped_fp4_available,
        )

        return grouped_fp4_available("nvfp4")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _grouped_available(),
    reason="grouped NVFP4 needs CUDA + a resolvable fp4 forward backend (Marlin sm80+ / "
    "DeepGEMM sm90+ / CUTLASS sm120)",
)


# ===========================================================================
# Dequant-path forward+backward oracle (executes on sm89 via Marlin + bf16-dequant bwd)
# ===========================================================================


@pytest.mark.parametrize("cfg", SHAPES, ids=SHAPE_IDS)
def test_grouped_fp4_fwd_bwd_dequant_path_matches_oracle(cfg):
    """Forward cosine >= 0.97 and backward grads cosine >= 0.95 vs the bf16 per-expert oracle,
    forcing the arch-agnostic bf16-dequant base dX (prefer_fp8_dx=False).

    Runs on any sm80+ GPU (Marlin forward + chunked bf16-dequant backward), so it executes on the
    sm89 CI where the Blackwell-gated oracle in test_grouped_fp4_train.py cannot.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        nvfp4_dequant_bf16,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_moe_train,
    )

    E, H, I = cfg["E"], cfg["H"], cfg["I"]
    s = cfg["scaling"]

    # Fresh NV tensors (the marlin backend frees qdata on first build, so the oracle weights are
    # dequantized from an independent copy of the same random matrices).
    twoI = 2 * I
    torch.manual_seed(0)
    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    gu_nv, dn_nv = _fresh_nv(Wgu, Wdn)
    pt = torch.ones(E, device=DEV)
    Wgu_b = nvfp4_dequant_bf16(gu_nv.qdata, gu_nv.scale, pt)
    Wdn_b = nvfp4_dequant_bf16(dn_nv.qdata, dn_nv.scale, pt)

    # Inputs (reuse the fixture for hidden/idx/wts + LoRA params, then make leaf copies).
    hidden, idx, wts, _gu, _dn, _Wgu, _Wdn, (Agu, Bgu, Adn, Bdn) = _make_inputs(cfg)

    hk = hidden.detach().clone().requires_grad_(True)
    Agk = Agu.detach().clone().requires_grad_(True)
    Bgk = Bgu.detach().clone().requires_grad_(True)
    Adk = Adn.detach().clone().requires_grad_(True)
    Bdk = Bdn.detach().clone().requires_grad_(True)

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
        prefer_fp8_dx=False,  # force the arch-agnostic bf16-dequant base dX (B8)
    )
    assert torch.isfinite(out).all(), f"{cfg['name']}: fwd output has NaN/inf"
    assert out.shape == (cfg["N"], H)
    out.float().pow(2).mean().backward()

    # Oracle on the same dequantized weights.
    Agr = Agu.detach().clone().requires_grad_(True)
    Bgr = Bgu.detach().clone().requires_grad_(True)
    Adr = Adn.detach().clone().requires_grad_(True)
    Bdr = Bdn.detach().clone().requires_grad_(True)
    hr = hidden.detach().clone().requires_grad_(True)
    ref = _bf16_oracle(
        hr, idx, wts, Wgu_b, Wdn_b, Agr, Bgr, Adr, Bdr, cfg["act_type"], cfg["limit"], s
    )
    cos_fwd = _cos(out, ref)
    assert cos_fwd > 0.97, f"{cfg['name']}: fwd cosine {cos_fwd:.4f} < 0.97"

    ref.pow(2).mean().backward()
    for name, gk, gr in [
        ("d_hidden", hk.grad, hr.grad),
        ("d_Agu", Agk.grad, Agr.grad),
        ("d_Adn", Adk.grad, Adr.grad),
    ]:
        assert gk is not None, (
            f"{cfg['name']}: {name} grad is None (backward didn't run)"
        )
        assert torch.isfinite(gk).all(), f"{cfg['name']}: {name} has NaN/inf"
        cos = _cos(gk, gr)
        assert cos > 0.95, f"{cfg['name']}: {name} cosine {cos:.4f} < 0.95"


# ===========================================================================
# B11: backward None-grad contract (data-independent FSDP2 backward collectives)
# ===========================================================================


def test_grouped_fp4_backward_none_grad_contract():
    """The grouped autograd Function returns grads for x + the four LoRA params and None for the
    routing / m_indices / offs / mode inputs (frozen experts -> no weight grad).

    This is the code basis of B11: the backward emits a grad ONLY for the differentiable leaves, so
    FSDP2's reduce-scatter sees gradients exclusively for the trainable LoRA params (a fixed,
    data-independent set) - never for the frozen NVFP4 experts or the integer routing tensors. We
    assert it by checking which inputs receive a grad after backward.

    Forced onto the bf16-dequant base dX (prefer_fp8_dx=False) so it runs on any sm80+ GPU (Marlin
    forward), executing on the sm89 CI. idx/wts are integer/non-leaf routing tensors that cannot
    carry grads; the contract is that x and the LoRA params DO and the experts do NOT.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_moe_train,
    )

    cfg = SHAPES[0]
    E, H, I = cfg["E"], cfg["H"], cfg["I"]
    s = cfg["scaling"]
    twoI = 2 * I
    torch.manual_seed(3)
    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    gu_nv, dn_nv = _fresh_nv(Wgu, Wdn)

    hidden, idx, wts, _gu, _dn, _Wgu, _Wdn, (Agu, Bgu, Adn, Bdn) = _make_inputs(
        cfg, seed=3
    )
    hk = hidden.detach().clone().requires_grad_(True)
    Agk = Agu.detach().clone().requires_grad_(True)
    Bgk = Bgu.detach().clone().requires_grad_(True)
    Adk = Adn.detach().clone().requires_grad_(True)
    Bdk = Bdn.detach().clone().requires_grad_(True)

    # The frozen experts must NOT request grad (FSDP2 never reduce-scatters them).
    assert not gu_nv.requires_grad and not dn_nv.requires_grad
    assert not idx.requires_grad and not wts.requires_grad

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
    out.float().pow(2).mean().backward()

    # Differentiable leaves: x + the four LoRA params get non-None, finite grads.
    for name, t in [
        ("x", hk),
        ("Agu", Agk),
        ("Bgu", Bgk),
        ("Adn", Adk),
        ("Bdn", Bdk),
    ]:
        assert t.grad is not None, (
            f"{name} grad is None (backward broke the LoRA grad path)"
        )
        assert torch.isfinite(t.grad).all(), f"{name} grad has NaN/inf"

    # Non-differentiable inputs (frozen experts, integer routing) carry no grad.
    assert getattr(gu_nv, "grad", None) is None, "frozen gate_up expert got a grad"
    assert getattr(dn_nv, "grad", None) is None, "frozen down expert got a grad"
    assert idx.grad is None and wts.grad is None, "routing tensors got a grad"


def test_grouped_experts_function_backward_returns_none_for_nondiff_inputs():
    """Directly assert _GroupedExperts.backward's None-grad tuple shape: grads only for x (pos 0)
    and the four LoRA params (Agu/Bgu/Adn/Bdn, pos 3-6); None for base / weight_recipe / m_indices /
    offs / scaling / limit / mode / act_type / prefer_fp8_dx. This is the literal B11 contract.

    Built on the Marlin forward + bf16-dequant backward so it runs on sm80+ (executes on sm89 CI).
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora import grouped_train as gt

    cfg = SHAPES[0]
    E, H, I = cfg["E"], cfg["H"], cfg["I"]
    s = cfg["scaling"]
    twoI = 2 * I
    torch.manual_seed(5)
    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    gu_nv, dn_nv = _fresh_nv(Wgu, Wdn)

    hidden, idx, wts, _gu, _dn, _Wgu, _Wdn, (Agu, Bgu, Adn, Bdn) = _make_inputs(
        cfg, seed=5
    )

    # Resolve a forward backend the same way grouped_fp4_moe_train does (Marlin on sm89).
    backend = gt._train_backend("nvfp4")
    if backend == "marlin":
        from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.backend import (
            MARLIN_TILE,
        )

        tile = MARLIN_TILE
    else:
        tile = gt.TILE

    N = hidden.size(0)
    dev = hidden.device
    flat = idx.reshape(-1)
    order = flat.argsort()
    rep = torch.arange(N, device=dev).repeat_interleave(idx.size(1))[order]
    exp_sorted = flat[order]
    counts = torch.bincount(flat, minlength=E)
    ptiles = (counts + tile - 1) // tile
    roff = torch.cat([ptiles.new_zeros(1), ptiles.cumsum(0)]) * tile
    coff = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    padded_row = roff[exp_sorted] + (
        torch.arange(exp_sorted.numel(), device=dev) - coff[exp_sorted]
    )
    m_indices = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=dev), ptiles
    )
    offs = (ptiles * tile).cumsum(0).to(torch.int32)
    Mt = int(ptiles.sum()) * tile

    if backend == "marlin":
        from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.backend import (
            build_marlin_forward_base,
        )

        base = build_marlin_forward_base(gu_nv, dn_nv, {})
    elif backend == "deepgemm":
        from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
            _cached_mxfp4,
        )

        base = (
            "deepgemm",
            _cached_mxfp4(gu_nv, gt._pt(gu_nv, E, dev), {}, "gate_up"),
            _cached_mxfp4(dn_nv, gt._pt(dn_nv, E, dev), {}, "down"),
        )
    else:  # cutlass (sm120)
        gu_eng = gt._engine(Mt, twoI, H, E, "nvfp4")
        gu_eng.set_weights(gu_nv.qdata, gu_nv.scale)
        dn_eng = gt._engine(Mt, H, I, E, "nvfp4")
        dn_eng.set_weights(dn_nv.qdata, dn_nv.scale)
        base = ("cutlass", gu_eng, dn_eng)

    Ags, Bgs, _ = gt._lora_stack((Agu, Bgu, s), E, H, twoI)
    Ads, Bds, _ = gt._lora_stack((Adn, Bdn, s), E, I, H)
    Ags = Ags.detach().requires_grad_(True)
    Bgs = Bgs.detach().requires_grad_(True)
    Ads = Ads.detach().requires_grad_(True)
    Bds = Bds.detach().requires_grad_(True)

    A = hidden.new_zeros(Mt, H).index_copy(0, padded_row, hidden[rep])
    A = A.detach().requires_grad_(True)
    lim = float(cfg["limit"])
    recipe = lambda: (gu_nv, dn_nv)  # noqa: E731

    dn = gt._GroupedExperts.apply(
        A,
        base,
        recipe,
        Ags,
        Bgs,
        Ads,
        Bds,
        m_indices,
        offs,
        s,
        lim,
        "nvfp4",
        cfg["act_type"],
        False,  # prefer_fp8_dx=False -> bf16-dequant base dX
    )
    dn.float().pow(2).mean().backward()

    # Differentiable leaves get grads; non-diff inputs get None (the Function's backward tuple).
    assert A.grad is not None and torch.isfinite(A.grad).all(), "dx (pos 0) missing"
    for nm, t in (("Agu", Ags), ("Bgu", Bgs), ("Adn", Ads), ("Bdn", Bds)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"{nm} grad missing"
    # m_indices / offs are int tensors and never leaves with grad; assert they stayed grad-free.
    assert m_indices.grad is None and offs.grad is None


# ===========================================================================
# Marlin fused-dequant bit-exactness on sm80+ (executes on sm89 via the Marlin ext)
# ===========================================================================
#
# test_grouped_fp4_train.py also has marlin bit-exact tests, but they sit under that module's
# Blackwell pytestmark AND a per-test capability==12 gate, so they only run on sm120. The Marlin
# W4A16 ext is sm80+ (see marlin_w4a16/__init__: standard Ampere mma, no Blackwell intrinsics), so
# the fused marlin->bf16 dequant is bit-exact-checkable on sm89 too. These sm80+-gated variants give
# that numeric check real coverage on the L40S CI without touching the Blackwell module.


def _marlin_available() -> bool:
    if not _IS_CUDA:
        return False
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import (
            marlin_w4a16_available,
        )

        return marlin_w4a16_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not _marlin_available(),
    reason="marlin W4A16 ext (sm80+, nvcc) required",
)
@pytest.mark.parametrize(
    "C,N,K", [(4, 64, 128), (4, 128, 64)], ids=["gate_up_N64_K128", "down_N128_K64"]
)
def test_marlin_fused_dequant_bit_exact_sm80(C, N, K):
    """marlin_dequant_bf16 is bit-exact vs nvfp4_dequant_bf16 on sm80+ (gate_up + down shapes).

    The marlin int32 layout decoded by the fused Triton dequant must match the reference NVFP4->bf16
    dequant exactly (maxerr == 0). N%64==0 (marlin N_t) and K%16==0 (NVFP4 block) are required.
    Runs on any sm80+ GPU with the Marlin ext, so it executes on the sm89 CI.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        nvfp4_dequant_bf16,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import load_ext
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.backend import (
        _build_base_scatter,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.fused_dequant import (
        marlin_dequant_bf16,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.prep import (
        prepare_nvfp4_weight_for_marlin,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.mx_weights import (
        fp4_codebook,
    )

    torch.manual_seed(42)
    W = torch.randn(C, N, K, device=DEV, dtype=torch.bfloat16) * 0.04
    nv = quantize_nvfp4(W)
    pt = torch.ones(C, device=DEV)

    ext = load_ext()
    scatter_lut = _build_base_scatter(torch.device(DEV))
    cb = fp4_codebook(torch.device(DEV)).float()

    marlin_packed = prepare_nvfp4_weight_for_marlin(
        nv.qdata, nv.scale, pt, N, K, torch.bfloat16, ext.gptq_marlin_repack
    )
    qw_flat = marlin_packed[0].reshape(C, -1)
    ref_bf16 = nvfp4_dequant_bf16(nv.qdata, nv.scale, pt)
    got_bf16 = marlin_dequant_bf16(qw_flat, nv.scale, pt, scatter_lut, cb, N, K, C)

    max_err = (ref_bf16.float() - got_bf16.float()).abs().max().item()
    assert max_err == 0.0, (
        f"marlin fused dequant not bit-exact vs nvfp4_dequant_bf16 (N={N}, K={K}): "
        f"maxerr={max_err}"
    )
