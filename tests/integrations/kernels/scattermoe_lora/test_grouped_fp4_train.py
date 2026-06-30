# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Correctness tests for the grouped/marlin NVFP4 MoE training path.

Covers:
  - grouped_fp4_moe_train fwd+bwd vs a bf16 per-expert oracle (DSV4 and gemma4 shapes)
  - Activation type dispatch (_detect_act_type) and wrong-activation detection
  - fp8-read dX NaN regression at non-128-aligned K (I=704, the gemma4 intermediate dim)
  - marlin->bf16 fused dequant bit-exactness vs nvfp4_dequant_bf16
  - GeGLU / clamped-SwiGLU activation kernel fwd+bwd vs torch reference

All tests use synthetic NVFP4 weights — no model download required.

H/I dimensions are kept at real model values (4096/2048 for DSV4, 2816/704 for gemma4) to satisfy
the marlin backend's 16-byte stride alignment requirement for torch._grouped_mm. E and N are small
(8-16 and 32) so the tests run fast (~2-10 s each on RTX PRO 6000).

Gate: skips unless CUDA is available AND device is Blackwell (sm100 / sm120).
"""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Skip gate: Blackwell only (sm100 / sm120)
# ---------------------------------------------------------------------------

_IS_CUDA = torch.cuda.is_available()
_IS_BLACKWELL = _IS_CUDA and torch.cuda.get_device_capability()[0] in (10, 12)

pytestmark = pytest.mark.skipif(
    not _IS_BLACKWELL,
    reason="grouped NVFP4 training backends require Blackwell (sm100/sm120)",
)

DEV = "cuda"


# ---------------------------------------------------------------------------
# Shape families — real H/I to satisfy 16-byte stride alignment in grouped_mm
# ---------------------------------------------------------------------------

# DSV4-Flash: clamped-SwiGLU, limit=7.0
_DSV4 = dict(
    name="dsv4",
    E=8,
    H=4096,
    I=2048,
    topk=6,
    N=32,
    r=8,
    act_type="silu",
    limit=7.0,
    scaling=2.0,
)

# gemma4-A4B: gelu_tanh GeGLU, no clamp
_GEMMA4 = dict(
    name="gemma4",
    E=8,
    H=2816,
    I=704,
    topk=8,
    N=32,
    r=8,
    act_type="gelu_tanh",
    limit=1e30,
    scaling=2.0,
)

SHAPES = [_DSV4, _GEMMA4]
SHAPE_IDS = ["dsv4", "gemma4"]


# ---------------------------------------------------------------------------
# NVFP4 quantizer — produces a real NVFP4Tensor (needed for the marlin backend
# which frees qdata.data after repacking). Self-contained, no scratch imports.
# ---------------------------------------------------------------------------


def _fp4_codebook(device):
    from axolotl.integrations.kernels.libs.scattermoe_lora.mx_weights import (
        fp4_codebook,
    )

    return fp4_codebook(device).float()


def quantize_nvfp4(W: torch.Tensor):
    """Quantize bf16 W[E,N,K] -> NVFP4Tensor matching nvfp4_dequant_bf16's inverse."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    dev = W.device
    cb = _fp4_codebook(dev)
    E, N, K = W.shape
    Wb = W.reshape(E, N, K // 16, 16).float()
    amax = Wb.abs().amax(-1).clamp_min(1e-6)
    scale_e4m3 = (amax / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale_e4m3.float().clamp_min(1e-9)
    nib = (
        ((Wb / scale_f.unsqueeze(-1)).unsqueeze(-1) - cb.view(1, 1, 1, 1, 16))
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
        per_tensor_scale=torch.ones((), device=dev),
    )


# ---------------------------------------------------------------------------
# bf16 per-expert MoE oracle
# ---------------------------------------------------------------------------


def _bf16_oracle(
    hidden, idx, wts, Wgu_b, Wdn_b, Agu, Bgu, Adn, Bdn, act_type, limit, scaling
):
    """Per-token per-expert bf16 MoE forward (no grouping), matching the kernel's intent.

    hidden [N,H]; idx/wts [N,topk]; Wgu_b/Wdn_b [E,out,in] bf16 dequant; *_lora scattermoe
    stacked layout [r*E, K] / [out, r*E]; act_type: 'silu' | 'gelu_tanh'; limit: clamp value.
    Returns [N,H] float32 accumulator (autograd-enabled via Agu/Bgu/Adn/Bdn).
    """
    N, H = hidden.shape
    E = Wgu_b.shape[0]
    r = Agu.shape[0] // E
    acc = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)
    for e in range(E):
        sel = idx == e  # [N, topk] bool
        tok = sel.any(-1)  # [N] bool
        if not tok.any():
            continue
        ti = tok.nonzero(as_tuple=True)[0]
        xe = hidden[ti].float()
        Ag = Agu[e * r : (e + 1) * r]  # [r, H]
        Bg = Bgu[:, e * r : (e + 1) * r]  # [2I, r]
        Ad = Adn[e * r : (e + 1) * r]  # [r, I]
        Bd = Bdn[:, e * r : (e + 1) * r]  # [H, r]
        # gate_up: base + LoRA
        gu = (
            xe.to(hidden.dtype) @ Wgu_b[e].T
            + scaling * (xe.to(hidden.dtype) @ Ag.T) @ Bg.T
        )
        # activation
        g, u = gu.chunk(2, -1)
        if act_type == "gelu_tanh":
            h = F.gelu(g.float(), approximate="tanh") * u.float()
        else:
            h = F.silu(g.float().clamp(max=limit)) * u.float().clamp(
                min=-limit, max=limit
            )
        h = h.to(hidden.dtype)
        # down: base + LoRA
        dn = h @ Wdn_b[e].T + scaling * (h @ Ad.T) @ Bd.T
        # weighted accumulate
        w = (wts * sel)[ti].sum(-1, keepdim=True).float()
        acc[ti] += dn.float() * w
    return acc


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


# ---------------------------------------------------------------------------
# Shared fixture: build inputs + weights for a shape family
# ---------------------------------------------------------------------------


def _make_inputs(cfg: dict, seed: int = 0):
    """Return (hidden, idx, wts, gu_nv, dn_nv, Wgu_b, Wdn_b, lora_params) plus fresh NV copies."""
    E, H, I, topk, N, r = cfg["E"], cfg["H"], cfg["I"], cfg["topk"], cfg["N"], cfg["r"]
    twoI = 2 * I
    torch.manual_seed(seed)
    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    gu_nv = quantize_nvfp4(Wgu)
    dn_nv = quantize_nvfp4(Wdn)

    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        nvfp4_dequant_bf16,
    )

    pt = torch.ones(E, device=DEV)
    Wgu_b = nvfp4_dequant_bf16(gu_nv.qdata, gu_nv.scale, pt)
    Wdn_b = nvfp4_dequant_bf16(dn_nv.qdata, dn_nv.scale, pt)

    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.stack([torch.randperm(E, device=DEV)[:topk] for _ in range(N)])
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1).to(torch.bfloat16)

    def mk(*s):
        return (
            torch.randn(*s, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)

    Agu = mk(r * E, H)
    Bgu = mk(twoI, r * E)
    Adn = mk(r * E, I)
    Bdn = mk(H, r * E)

    return hidden, idx, wts, gu_nv, dn_nv, Wgu_b, Wdn_b, (Agu, Bgu, Adn, Bdn)


# ---------------------------------------------------------------------------
# Fresh NVFP4Tensor helper (marlin frees qdata on first call; tests that need
# multiple independent calls must build separate NVFP4Tensors from the same W)
# ---------------------------------------------------------------------------


def _fresh_nv(Wgu, Wdn):
    """Build fresh NVFP4Tensor pair from the same weight matrices (for second-call tests)."""
    return quantize_nvfp4(Wgu), quantize_nvfp4(Wdn)


# ===========================================================================
# Test 1: forward output matches oracle
# ===========================================================================


@pytest.mark.parametrize("cfg", SHAPES, ids=SHAPE_IDS)
def test_grouped_fp4_fwd_matches_oracle(cfg):
    """Forward output cosine >= 0.97 vs bf16 per-expert oracle, no NaN/inf."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
        grouped_fp4_moe_train,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend available")

    H = cfg["H"]
    hidden, idx, wts, gu_nv, dn_nv, Wgu_b, Wdn_b, (Agu, Bgu, Adn, Bdn) = _make_inputs(
        cfg
    )
    _r, s = cfg["r"], cfg["scaling"]

    cache = {}
    out = grouped_fp4_moe_train(
        hidden.clone(),
        idx,
        wts,
        gu_nv,
        dn_nv,
        (Agu.detach(), Bgu.detach(), s),
        (Adn.detach(), Bdn.detach(), s),
        cfg["limit"],
        "nvfp4",
        act_type=cfg["act_type"],
        mxfp4_cache=cache,
    )
    assert torch.isfinite(out).all(), f"{cfg['name']}: fwd output has NaN/inf"
    assert out.shape == (cfg["N"], H), f"unexpected output shape {out.shape}"

    ref = _bf16_oracle(
        hidden.clone(),
        idx,
        wts,
        Wgu_b,
        Wdn_b,
        Agu,
        Bgu,
        Adn,
        Bdn,
        cfg["act_type"],
        cfg["limit"],
        s,
    )
    cos = _cos(out, ref)
    assert cos > 0.97, f"{cfg['name']}: fwd cosine {cos:.4f} < 0.97"


# ===========================================================================
# Test 1b: marlin qdata-free experts survive clone/state_dict (adapter save)
# ===========================================================================


@pytest.mark.parametrize("cfg", SHAPES, ids=SHAPE_IDS)
def test_grouped_fp4_save_after_qdata_free(cfg):
    """Regression: the marlin memory fix frees nv.qdata after repack. The freed NVFP4Tensor must
    still clone/serialize (PEFT save_pretrained -> state_dict -> NVFP4Tensor clone -> __new__ ->
    qdata.stride(-2)); a flat empty(0) qdata crashed with IndexError(dim -2). Freeing to a 3-D
    [E, N, 0] placeholder keeps clone/save working."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        _train_backend,
        grouped_fp4_available,
        grouped_fp4_moe_train,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend available")
    if _train_backend("nvfp4") != "marlin":
        pytest.skip("qdata-free only on the marlin backend")

    hidden, idx, wts, gu_nv, dn_nv, _Wgu, _Wdn, (Agu, Bgu, Adn, Bdn) = _make_inputs(cfg)
    s = cfg["scaling"]
    # one forward triggers the marlin repack + qdata free
    grouped_fp4_moe_train(
        hidden.clone(),
        idx,
        wts,
        gu_nv,
        dn_nv,
        (Agu.detach(), Bgu.detach(), s),
        (Adn.detach(), Bdn.detach(), s),
        cfg["limit"],
        "nvfp4",
        act_type=cfg["act_type"],
        mxfp4_cache={},
    )
    # qdata was freed to a 3-D zero-element placeholder (not flat empty(0))
    assert gu_nv.qdata.numel() == 0 and gu_nv.qdata.ndim == 3, (
        "qdata not freed to 3-D placeholder"
    )
    # The freed NVFP4Tensor must CLONE / round-trip through state_dict without raising — this is the
    # save path (PEFT save_pretrained -> state_dict -> NVFP4Tensor clone). A flat empty(0) qdata
    # crashed here with IndexError(dim -2). We don't assert the frozen expert's clone *shape* (it's
    # degenerate by design and filtered out of the LoRA adapter); only that save doesn't crash.
    import torch.nn as nn

    for nv in (gu_nv, dn_nv):
        nv.clone()  # would IndexError(dim -2) on a flat empty(0) qdata
        m = nn.Module()
        m.w = nn.Parameter(nv, requires_grad=False)
        sd = m.state_dict()  # invokes _save_to_state_dict -> clone on the NVFP4Tensor
        assert "w" in sd


# ===========================================================================
# Test 2: backward grads match oracle
# ===========================================================================


@pytest.mark.parametrize("cfg", SHAPES, ids=SHAPE_IDS)
def test_grouped_fp4_bwd_matches_oracle(cfg):
    """Backward grads (d_hidden, d_LoRA_A_gu, d_LoRA_A_dn) cosine >= 0.95, finite."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        nvfp4_dequant_bf16,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
        grouped_fp4_moe_train,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend available")

    E, H, I = cfg["E"], cfg["H"], cfg["I"]
    hidden, idx, wts, gu_nv, dn_nv, Wgu_b, Wdn_b, (Agu, Bgu, Adn, Bdn) = _make_inputs(
        cfg
    )
    _r, s = cfg["r"], cfg["scaling"]

    # Kernel path — use fresh NV tensors for backward (marlin frees qdata on first build)
    twoI = 2 * I
    torch.manual_seed(0)
    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    gu_nv_k, dn_nv_k = _fresh_nv(Wgu, Wdn)
    pt = torch.ones(E, device=DEV)
    Wgu_b_k = nvfp4_dequant_bf16(gu_nv_k.qdata, gu_nv_k.scale, pt)
    Wdn_b_k = nvfp4_dequant_bf16(dn_nv_k.qdata, dn_nv_k.scale, pt)

    hidden2 = hidden.clone()
    idx2 = idx.clone()
    wts2 = wts.clone()
    Agk = Agu.detach().clone().requires_grad_(True)
    Bgk = Bgu.detach().clone().requires_grad_(True)
    Adk = Adn.detach().clone().requires_grad_(True)
    Bdk = Bdn.detach().clone().requires_grad_(True)
    hk = hidden2.requires_grad_(True)

    cache = {}
    out = grouped_fp4_moe_train(
        hk,
        idx2,
        wts2,
        gu_nv_k,
        dn_nv_k,
        (Agk, Bgk, s),
        (Adk, Bdk, s),
        cfg["limit"],
        "nvfp4",
        act_type=cfg["act_type"],
        mxfp4_cache=cache,
    )
    out.float().pow(2).mean().backward()

    # Oracle path — same weights
    Agr = Agk.detach().clone().requires_grad_(True)
    Bgr = Bgk.detach().clone().requires_grad_(True)
    Adr = Adk.detach().clone().requires_grad_(True)
    Bdr = Bdk.detach().clone().requires_grad_(True)
    hr = hidden2.detach().clone().requires_grad_(True)

    ref = _bf16_oracle(
        hr,
        idx2,
        wts2,
        Wgu_b_k,
        Wdn_b_k,
        Agr,
        Bgr,
        Adr,
        Bdr,
        cfg["act_type"],
        cfg["limit"],
        s,
    )
    ref.pow(2).mean().backward()

    for name, gk, gr in [
        ("d_hidden", hk.grad, hr.grad),
        ("d_Agu", Agk.grad, Agr.grad),
        ("d_Adn", Adk.grad, Adr.grad),
    ]:
        assert gk is not None, (
            f"{cfg['name']}: {name} grad is None (backward didn't run)"
        )
        assert gr is not None, f"{cfg['name']}: oracle {name} grad is None"
        assert torch.isfinite(gk).all(), f"{cfg['name']}: {name} has NaN/inf"
        cos = _cos(gk, gr)
        assert cos > 0.95, f"{cfg['name']}: {name} cosine {cos:.4f} < 0.95"


# ===========================================================================
# Test 3: activation type dispatch
# ===========================================================================


def test_detect_act_type_dispatch():
    """_detect_act_type must return 'gelu_tanh' for gemma4-style and 'silu' for DSV4."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        _detect_act_type,
    )

    class FakeGemma4:
        act_fn = functools.partial(F.gelu, approximate="tanh")

    class FakeDSV4:
        act_fn = F.silu
        limit = 7.0

    class FakeNamedGeLU:
        class _GeLU:
            __name__ = "gelu_pytorch_tanh"

            def __call__(self, x):
                return F.gelu(x, approximate="tanh")

        act_fn = _GeLU()

    assert _detect_act_type(FakeGemma4()) == "gelu_tanh", (
        "gemma4 (partial F.gelu tanh) must be 'gelu_tanh'"
    )
    assert _detect_act_type(FakeDSV4()) == "silu", "dsv4 (F.silu) must be 'silu'"
    assert _detect_act_type(FakeNamedGeLU()) == "gelu_tanh", (
        "named gelu_pytorch_tanh must be 'gelu_tanh'"
    )


def test_wrong_activation_fails_gemma4():
    """Using act_type='silu' on a gemma4 shape (correct: 'gelu_tanh') degrades cosine vs oracle.

    Demonstrates TEETH: the correct activation (gelu_tanh) achieves cosine >= 0.999 vs the
    correct oracle, while the wrong activation (silu) achieves only ~0.992. The gap is large
    enough to be a reliable regression detector.

    Note: absolute cosine with wrong-act vs correct-oracle stays ~0.99 (gelu_tanh ≈ silu for
    moderate values), so the test checks the *relative gap* rather than an absolute threshold.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        nvfp4_dequant_bf16,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
        grouped_fp4_moe_train,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend available")

    cfg = _GEMMA4
    E, H, I, topk, N, r = cfg["E"], cfg["H"], cfg["I"], cfg["topk"], cfg["N"], cfg["r"]
    twoI = 2 * I
    torch.manual_seed(99)
    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.stack([torch.randperm(E, device=DEV)[:topk] for _ in range(N)])
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1).to(torch.bfloat16)
    Agu = torch.randn(r * E, H, device=DEV, dtype=torch.bfloat16) * 0.02
    Bgu = torch.randn(twoI, r * E, device=DEV, dtype=torch.bfloat16) * 0.02
    Adn = torch.randn(r * E, I, device=DEV, dtype=torch.bfloat16) * 0.02
    Bdn = torch.randn(H, r * E, device=DEV, dtype=torch.bfloat16) * 0.02

    pt = torch.ones(E, device=DEV)

    # Correct-act run
    gu_nv_c, dn_nv_c = _fresh_nv(Wgu, Wdn)
    Wgu_b = nvfp4_dequant_bf16(gu_nv_c.qdata, gu_nv_c.scale, pt)
    Wdn_b = nvfp4_dequant_bf16(dn_nv_c.qdata, dn_nv_c.scale, pt)
    cache_c = {}
    out_correct = grouped_fp4_moe_train(
        hidden,
        idx,
        wts,
        gu_nv_c,
        dn_nv_c,
        (Agu, Bgu, cfg["scaling"]),
        (Adn, Bdn, cfg["scaling"]),
        cfg["limit"],
        "nvfp4",
        act_type="gelu_tanh",
        mxfp4_cache=cache_c,
    )

    # Wrong-act run (fresh NV tensors; marlin freed qdata in first call)
    gu_nv_w, dn_nv_w = _fresh_nv(Wgu, Wdn)
    cache_w = {}
    out_wrong = grouped_fp4_moe_train(
        hidden,
        idx,
        wts,
        gu_nv_w,
        dn_nv_w,
        (Agu, Bgu, cfg["scaling"]),
        (Adn, Bdn, cfg["scaling"]),
        cfg["limit"],
        "nvfp4",
        act_type="silu",  # deliberate wrong activation
        mxfp4_cache=cache_w,
    )

    # Correct-activation oracle
    correct_ref = _bf16_oracle(
        hidden,
        idx,
        wts,
        Wgu_b,
        Wdn_b,
        Agu,
        Bgu,
        Adn,
        Bdn,
        "gelu_tanh",
        cfg["limit"],
        cfg["scaling"],
    )

    cos_correct = _cos(out_correct, correct_ref)
    cos_wrong = _cos(out_wrong, correct_ref)

    # The correct activation must match its oracle very well
    assert cos_correct > 0.99, (
        f"correct-act kernel cosine vs oracle {cos_correct:.4f} < 0.99 — something is wrong"
    )
    # The wrong activation must diverge detectably: gap >= 0.005 (measured ~0.007-0.008)
    gap = cos_correct - cos_wrong
    assert gap > 0.005, (
        f"TEETH CHECK FAILED: wrong activation (silu on gemma4) barely diverges from correct "
        f"(correct={cos_correct:.4f}, wrong={cos_wrong:.4f}, gap={gap:.5f} < 0.005). "
        "The activation correctness test lacks discriminating power."
    )


# ===========================================================================
# Test 4: NaN regression at non-128-aligned K (I=704, fp8-read dX)
# ===========================================================================


def test_grouped_dx_fp8_nan_regression_k704():
    """grouped_dx_fp8 must produce finite output for K=704 (non-BK-aligned), the gemma4 I dim.

    Original bug: the K-tile boundary had no mask, so the last tile read OOB fp8 values that
    decoded to NaN in the accumulator. Fixed by adding mk=rk<K on the W load and store.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        grouped_dx_fp8,
    )

    E, H, I = 16, 2816, 704  # gemma4 shapes; I=704 = 5.5*128, NOT BK-aligned
    tile = 64  # marlin routing tile
    Mt = E * tile

    torch.manual_seed(0)
    grad = torch.randn(Mt, H, dtype=torch.bfloat16, device=DEV) * 0.01
    w_fp8 = torch.zeros(E, H, I, dtype=torch.float8_e4m3fn, device=DEV)
    m_indices = torch.arange(E, dtype=torch.int32, device=DEV)

    out = grouped_dx_fp8(grad, w_fp8, m_indices, block_m=tile)
    assert not out.isnan().any(), "K=704 produced NaN in grouped_dx_fp8 (regression)"
    assert not out.isinf().any(), "K=704 produced Inf in grouped_dx_fp8"
    assert out.shape == (Mt, I)


def test_grouped_dx_fp8_k2048_no_regression():
    """grouped_dx_fp8 must also work correctly for K=2048 (DSV4, BK-aligned)."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        grouped_dx_fp8,
    )

    E, H, I = 16, 4096, 2048
    tile = 64
    Mt = E * tile

    torch.manual_seed(1)
    grad = torch.randn(Mt, H, dtype=torch.bfloat16, device=DEV) * 0.01
    w_fp8 = torch.zeros(E, H, I, dtype=torch.float8_e4m3fn, device=DEV)
    m_indices = torch.arange(E, dtype=torch.int32, device=DEV)

    out = grouped_dx_fp8(grad, w_fp8, m_indices, block_m=tile)
    assert not out.isnan().any(), "K=2048 produced NaN (regression)"
    assert not out.isinf().any()
    assert out.shape == (Mt, I)


def test_grouped_fp4_bwd_nan_k704():
    """Full fwd+bwd on the gemma4 shape (I=704) must produce no NaN in any gradient.

    This exercises the entire backward path including the chunked fp8-read dX at K=704.
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
        grouped_fp4_moe_train,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend available")

    cfg = _GEMMA4
    E, H, I, topk, N, r = cfg["E"], cfg["H"], cfg["I"], cfg["topk"], cfg["N"], cfg["r"]
    twoI = 2 * I
    torch.manual_seed(7)
    gu_nv = quantize_nvfp4(
        torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    )
    dn_nv = quantize_nvfp4(
        torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    )
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.stack([torch.randperm(E, device=DEV)[:topk] for _ in range(N)])
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1).to(torch.bfloat16)

    s = cfg["scaling"]
    Agu = (
        torch.randn(r * E, H, device=DEV, dtype=torch.bfloat16) * 0.02
    ).requires_grad_(True)
    Bgu = (
        torch.randn(twoI, r * E, device=DEV, dtype=torch.bfloat16) * 0.02
    ).requires_grad_(True)
    Adn = (
        torch.randn(r * E, I, device=DEV, dtype=torch.bfloat16) * 0.02
    ).requires_grad_(True)
    Bdn = (
        torch.randn(H, r * E, device=DEV, dtype=torch.bfloat16) * 0.02
    ).requires_grad_(True)
    hk = hidden.clone().requires_grad_(True)

    cache = {}
    out = grouped_fp4_moe_train(
        hk,
        idx,
        wts,
        gu_nv,
        dn_nv,
        (Agu, Bgu, s),
        (Adn, Bdn, s),
        cfg["limit"],
        "nvfp4",
        act_type=cfg["act_type"],
        mxfp4_cache=cache,
    )
    out.float().pow(2).mean().backward()

    for name, t in [
        ("out", out),
        ("d_hidden", hk.grad),
        ("d_Agu", Agu.grad),
        ("d_Bgu", Bgu.grad),
        ("d_Adn", Adn.grad),
        ("d_Bdn", Bdn.grad),
    ]:
        assert t is not None, f"gemma4 K=704 backward: {name} is None"
        assert torch.isfinite(t).all(), f"gemma4 K=704 backward: {name} has NaN/inf"


# ===========================================================================
# Test 5: marlin->bf16 fused dequant bit-exactness vs nvfp4_dequant_bf16
# ===========================================================================


@pytest.mark.skipif(
    not (_IS_CUDA and torch.cuda.get_device_capability()[0] >= 8),
    reason="marlin W4A16 backend needs Ampere (sm80)+ (the module's Blackwell pytestmark still "
    "gates this file to sm100/sm120 on CI; the sm80+ marlin bit-exact variant that executes on "
    "sm89 lives in test_grouped_fp4_dequant_path.py)",
)
def test_marlin_fused_dequant_bit_exact_gate_up():
    """marlin_dequant_bf16 is bit-exact vs nvfp4_dequant_bf16 for gate_up (N=64, K=128) shape."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import (
        marlin_w4a16_available,
    )

    if not marlin_w4a16_available():
        pytest.skip("marlin W4A16 ext not available")

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

    # N must be divisible by 64 (marlin N_t), K divisible by 16 (NVFP4 block)
    C, N, K = 4, 64, 128
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
        f"marlin fused dequant not bit-exact vs nvfp4_dequant_bf16 (gate_up N={N}, K={K}): "
        f"maxerr={max_err}"
    )


@pytest.mark.skipif(
    not (_IS_CUDA and torch.cuda.get_device_capability()[0] >= 8),
    reason="marlin W4A16 backend needs Ampere (sm80)+ (the module's Blackwell pytestmark still "
    "gates this file to sm100/sm120 on CI; the sm80+ marlin bit-exact variant that executes on "
    "sm89 lives in test_grouped_fp4_dequant_path.py)",
)
def test_marlin_fused_dequant_bit_exact_down():
    """marlin_dequant_bf16 is bit-exact vs nvfp4_dequant_bf16 for down (N=128, K=64) shape."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import (
        marlin_w4a16_available,
    )

    if not marlin_w4a16_available():
        pytest.skip("marlin W4A16 ext not available")

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

    # down: N=H, K=I; marlin requirement: N%64==0, K%16==0
    C, N, K = 4, 128, 64
    torch.manual_seed(43)
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
        f"marlin fused dequant not bit-exact vs nvfp4_dequant_bf16 (down N={N}, K={K}): "
        f"maxerr={max_err}"
    )


# ===========================================================================
# Test 6: activation kernel unit tests (GeGLU and clamped-SwiGLU vs torch)
# ===========================================================================


def _torch_geglu_fwd(gu):
    g, u = gu.chunk(2, dim=-1)
    return F.gelu(g.float(), approximate="tanh") * u.float()


def _torch_geglu_bwd(gu, dh):
    gu_f = gu.float().requires_grad_(True)
    g, u = gu_f.chunk(2, dim=-1)
    h = F.gelu(g, approximate="tanh") * u
    h.backward(dh.float())
    return gu_f.grad.to(gu.dtype)


def _torch_swiglu_fwd(gu, limit):
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g.float().clamp(max=limit)) * u.float().clamp(min=-limit, max=limit)


def _torch_swiglu_bwd(gu, dh, limit):
    gu_f = gu.float().requires_grad_(True)
    g, u = gu_f.chunk(2, dim=-1)
    h = F.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
    h.backward(dh.float())
    return gu_f.grad.to(gu.dtype)


def _check_rel_err(name, got, ref, rtol=5e-3):
    diff = (got.float() - ref.float()).abs()
    rel_err = diff.mean() / (ref.float().abs().mean() + 1e-8)
    assert rel_err < rtol, f"{name}: rel_err={rel_err:.5f} >= {rtol}"


@pytest.mark.parametrize(
    "I,seed", [(704, 42), (2048, 42)], ids=["I=704(gemma4)", "I=2048(dsv4)"]
)
def test_geglu_kernel_vs_torch(I, seed):
    """GeGLU (gelu_tanh) Triton kernel fwd+bwd matches torch reference."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.cutlass_fp4.swiglu import (
        swiglu_bwd,
        swiglu_fwd,
    )

    torch.manual_seed(seed)
    Mt = 512
    gu = torch.randn(Mt, 2 * I, dtype=torch.bfloat16, device=DEV) * 0.5
    dh = torch.randn(Mt, I, dtype=torch.bfloat16, device=DEV) * 0.1

    ref_h = _torch_geglu_fwd(gu)
    got_h = swiglu_fwd(gu, limit=1e30, act_type="gelu_tanh")
    _check_rel_err("geglu_fwd", got_h.float(), ref_h)

    ref_dgu = _torch_geglu_bwd(gu, dh)
    got_dgu = swiglu_bwd(gu, dh, limit=1e30, act_type="gelu_tanh")
    _check_rel_err("geglu_bwd_dg", got_dgu[:, :I].float(), ref_dgu[:, :I].float())
    _check_rel_err("geglu_bwd_du", got_dgu[:, I:].float(), ref_dgu[:, I:].float())


@pytest.mark.parametrize(
    "limit,I,seed",
    [(10.0, 704, 42), (7.0, 2048, 42)],
    ids=["swiglu_clamp_I704", "swiglu_clamp_I2048"],
)
def test_swiglu_clamped_kernel_vs_torch(limit, I, seed):
    """Clamped SwiGLU Triton kernel fwd+bwd matches torch reference."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.cutlass_fp4.swiglu import (
        swiglu_bwd,
        swiglu_fwd,
    )

    torch.manual_seed(seed)
    Mt = 512
    gu = torch.randn(Mt, 2 * I, dtype=torch.bfloat16, device=DEV) * 5.0
    dh = torch.randn(Mt, I, dtype=torch.bfloat16, device=DEV) * 0.1

    ref_h = _torch_swiglu_fwd(gu, limit)
    got_h = swiglu_fwd(gu, limit=limit, act_type="silu")
    _check_rel_err("swiglu_fwd", got_h.float(), ref_h)

    ref_dgu = _torch_swiglu_bwd(gu, dh, limit)
    got_dgu = swiglu_bwd(gu, dh, limit=limit, act_type="silu")
    _check_rel_err("swiglu_bwd_dg", got_dgu[:, :I].float(), ref_dgu[:, :I].float())
    _check_rel_err("swiglu_bwd_du", got_dgu[:, I:].float(), ref_dgu[:, I:].float())
