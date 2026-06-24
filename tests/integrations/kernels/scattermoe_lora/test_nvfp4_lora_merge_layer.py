# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Per-layer expert-LoRA merge (step 2 of the NVFP4 expert-LoRA merge).

``merge_layer_experts`` dequantizes the fused per-expert NVFP4 base (gate_up = cat of two
source projs on the out/row axis, down = single proj), adds the reconstructed LoRA delta,
UN-FUSES gate_up by splitting on the out axis, and requantizes each (expert, source_proj)
2D weight so gate and up of an expert SHARE one per-tensor scale (the NVIDIA checkpoint
invariant the loaders depend on).

The CPU tests load ``nvfp4_lora_merge`` by file path so the scattermoe_lora package
``__init__`` (which imports triton) is NOT executed. They cover the un-fuse split shapes
and the scheme-key mapping on synthetic tensors (no torchao needed).

The torchao tests are gated only on ``importorskip torchao`` (NOT on CUDA): ``to_nvfp4`` /
``nvfp4_quantize`` and ``dequantize`` are pure torch and run on CPU; only the NVFP4 matmul
path uses ``_scaled_mm`` (CUDA). They pin (a) requant self-consistency, dequant(requant(
merged_expert)) approx merged_expert within NVFP4 E2M1 rounding tolerance, (b) a round-trip
through the ACTUAL loader fusion (re-cat the per-proj qdata/scale, rebuild the fused
NVFP4Tensor with ONLY gate's weight_scale_2 as the loaders do) reproduces both gate and up
rows when a LoRA delta moves gate's amax far from up's, and (c) gate and up of an expert get
the SAME weight_scale_2.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch


def _load_merge_module():
    """Load nvfp4_lora_merge by file path so the scattermoe_lora package __init__ (which imports
    triton) is NOT executed. The CPU-only entry points need only torch."""
    import axolotl

    path = os.path.join(
        os.path.dirname(axolotl.__file__),
        "integrations",
        "kernels",
        "libs",
        "scattermoe_lora",
        "nvfp4_lora_merge.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_axolotl_nvfp4_lora_merge_layer_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merge_mod = _load_merge_module()

# The two real checkpoint schemes (mirrors nvfp4_moe_loading._NVFP4_MOE_SCHEMES).
DSV4_SCHEME = {
    "base_fmt": "layers.{layer}.ffn.experts.{e}.{proj}",
    "gate_up": ("w1", "w3"),
    "down": ("w2",),
}
GEMMA4_SCHEME = {
    "base_fmt": "model.language_model.layers.{layer}.experts.{e}.{proj}",
    "gate_up": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
}


# ---------------------------------------------------------------------------
# CPU: scheme-key mapping (no torchao)
# ---------------------------------------------------------------------------


def test_checkpoint_keys_dsv4():
    keys = merge_mod.checkpoint_keys_for(
        DSV4_SCHEME, layer=5, source_proj="w1", expert=3
    )
    assert keys == {
        "weight": "layers.5.ffn.experts.3.w1.weight",
        "weight_scale": "layers.5.ffn.experts.3.w1.weight_scale",
        "weight_scale_2": "layers.5.ffn.experts.3.w1.weight_scale_2",
    }


def test_checkpoint_keys_gemma4():
    keys = merge_mod.checkpoint_keys_for(
        GEMMA4_SCHEME, layer=0, source_proj="down_proj", expert=2
    )
    assert keys == {
        "weight": "model.language_model.layers.0.experts.2.down_proj.weight",
        "weight_scale": "model.language_model.layers.0.experts.2.down_proj.weight_scale",
        "weight_scale_2": "model.language_model.layers.0.experts.2.down_proj.weight_scale_2",
    }


# ---------------------------------------------------------------------------
# CPU: un-fuse split sizes (no torchao)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_sources", [2])
def test_gate_up_split_even(n_sources):
    sizes = merge_mod._gate_up_split_sizes(2 * 64, n_sources)
    assert sizes == [64] * n_sources
    assert sum(sizes) == 2 * 64


def test_gate_up_split_uneven_raises():
    with pytest.raises(ValueError):
        merge_mod._gate_up_split_sizes(65, 2)


# ---------------------------------------------------------------------------
# torchao (CPU-runnable: requant is pure torch, no CUDA)
# ---------------------------------------------------------------------------

torchao = pytest.importorskip("torchao", reason="torchao required for NVFP4 requant")
from torchao.prototype.mx_formats.nvfp4_tensor import (  # noqa: E402
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)

# gemma4-ish shapes: H (hidden) % 16 == 0, I (intermediate) % 16 == 0.
E, H, I = 3, 64, 32
BLOCK = 16


def _build_fused(W_list_per_proj, projs):
    """Fuse per-expert 2D bf16 weights into one fused NVFP4Tensor exactly like the loader:
    quantize each proj per expert (own per-tensor scale), stack experts, cat projs on dim 1.

    Returns ``(fused_nvfp4, dequantized_fused_bf16 [E, sumOut, in])`` where the dequant is
    rebuilt per expert from its own scalar scale (torchao's fused[e] index does not slice the
    [E,1,1] per_tensor_scale)."""
    qd_projs, sc_projs, pts_per_expert = [], [], []
    for pi, _ in enumerate(projs):
        qd_e, sc_e = [], []
        for e in range(E):
            pts = per_tensor_amax_to_scale(W_list_per_proj[pi][e].abs().max())
            t = NVFP4Tensor.to_nvfp4(
                W_list_per_proj[pi][e], block_size=BLOCK, per_tensor_scale=pts
            )
            qd_e.append(t.qdata)
            sc_e.append(t.scale)
            if pi == 0:  # gate/up share one weight_scale_2 per expert in the checkpoint
                pts_per_expert.append(pts.to(torch.float32))
        qd_projs.append(torch.stack(qd_e, 0))
        sc_projs.append(torch.stack(sc_e, 0))
    qdata = torch.cat(qd_projs, dim=1)
    scale = torch.cat(sc_projs, dim=1)
    pts = torch.stack(pts_per_expert).view(-1, 1, 1)
    fused = NVFP4Tensor(qdata, scale, BLOCK, torch.bfloat16, per_tensor_scale=pts)

    # per-expert dequant from the expert's own scalar scale
    pe = pts.reshape(-1)
    dq = []
    for e in range(E):
        t_e = NVFP4Tensor(
            qdata[e], scale[e], BLOCK, torch.bfloat16, per_tensor_scale=pe[e]
        )
        dq.append(t_e.dequantize(torch.bfloat16))
    return fused, torch.stack(dq, 0)


def _loader_fused_gate_up_dequant(merge_out, gate_proj, up_proj):
    """Reload merge output exactly as the loaders do and dequantize the fused gate||up block.

    Replicates ``nvfp4_moe_loading._build_expert_nvfp4`` / ``Nvfp4ExpertsDeserialize``: cat the
    per-proj qdata/scale on the row axis and build the fused NVFP4Tensor with ONLY gate's
    weight_scale_2 (the loaders discard up's). Returns bf16 ``[E, 2I, H]``."""
    dq = []
    for e in range(E):
        g_r, u_r = merge_out[gate_proj][e], merge_out[up_proj][e]
        qdata = torch.cat([g_r["weight"], u_r["weight"]], dim=0)
        scale = torch.cat([g_r["weight_scale"], u_r["weight_scale"]], dim=0)
        t = NVFP4Tensor(
            qdata, scale, BLOCK, torch.bfloat16, per_tensor_scale=g_r["weight_scale_2"]
        )
        dq.append(t.dequantize(torch.bfloat16))
    return torch.stack(dq, 0)


def _random_weights(out_f, in_f, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(out_f, in_f, generator=g, dtype=torch.bfloat16) for _ in range(E)
    ]


def test_merge_layer_unfuse_shapes_no_lora():
    """merge_layer_experts returns the source proj names with [out,in]-shaped requant tensors."""
    gate = _random_weights(I, H, 1)
    up = _random_weights(I, H, 2)
    down = _random_weights(H, I, 3)
    fused_gu, _ = _build_fused([gate, up], ("gate_proj", "up_proj"))
    fused_dn, _ = _build_fused([down], ("down_proj",))

    out = merge_mod.merge_layer_experts(fused_gu, fused_dn, None, None, GEMMA4_SCHEME)
    assert set(out.keys()) == {"gate_proj", "up_proj", "down_proj"}
    for proj in ("gate_proj", "up_proj"):
        assert set(out[proj].keys()) == set(range(E))
        for e in range(E):
            assert out[proj][e]["weight"].shape == (I, H // 2)  # uint8 packed (H/2)
            assert out[proj][e]["weight"].dtype == torch.uint8
            assert out[proj][e]["weight_scale"].shape == (I, H // BLOCK)
            assert out[proj][e]["weight_scale"].dtype == torch.float8_e4m3fn
            assert out[proj][e]["weight_scale_2"].dtype == torch.float32
            assert out[proj][e]["weight_scale_2"].numel() == 1
    for e in range(E):
        assert out["down_proj"][e]["weight"].shape == (H, I // 2)


def test_merge_layer_dsv4_scheme_names():
    """The DSV4 scheme drives the source proj names (w1/w3/w2), not gemma4 names."""
    gate = _random_weights(I, H, 10)
    up = _random_weights(I, H, 11)
    down = _random_weights(H, I, 12)
    fused_gu, _ = _build_fused([gate, up], ("w1", "w3"))
    fused_dn, _ = _build_fused([down], ("w2",))

    out = merge_mod.merge_layer_experts(fused_gu, fused_dn, None, None, DSV4_SCHEME)
    assert set(out.keys()) == {"w1", "w3", "w2"}


def test_merge_layer_requant_self_consistent():
    """dequant(requant(merged_expert)) approx merged_expert within NVFP4 rounding.

    With no LoRA, the merged weight equals the dequantized base. Requantizing it and
    dequantizing again is a single extra NVFP4 round-trip of an ALREADY-NVFP4 weight, so the
    residual is bounded by E2M1's per-block 4-bit (1 sign, 2 exp, 1 mantissa) relative
    resolution. Worst-case relative step within a block is ~1/6, so we allow rel tol 0.20 on
    the per-element max-abs error (scaled by the weight amax). The two NVFP4 grids (base and
    requant) share the same per-block + per-tensor scaling recipe, so in practice the error is
    far below this bound; the loose tol just guards the orientation/scale plumbing, not bit
    reproduction."""
    gate = _random_weights(I, H, 21)
    up = _random_weights(I, H, 22)
    down = _random_weights(H, I, 23)
    fused_gu, gu_bf16 = _build_fused([gate, up], ("gate_proj", "up_proj"))
    fused_dn, dn_bf16 = _build_fused([down], ("down_proj",))

    out = merge_mod.merge_layer_experts(fused_gu, fused_dn, None, None, GEMMA4_SCHEME)

    # gate is rows [0:I) of the fused dequant; up is rows [I:2I).
    refs = {
        "gate_proj": gu_bf16[:, :I, :],
        "up_proj": gu_bf16[:, I:, :],
        "down_proj": dn_bf16,
    }
    for proj, ref in refs.items():
        for e in range(E):
            r = out[proj][e]
            req = NVFP4Tensor(
                r["weight"],
                r["weight_scale"],
                BLOCK,
                torch.bfloat16,
                per_tensor_scale=r["weight_scale_2"],
            ).dequantize(torch.bfloat16)
            ref_e = ref[e].float()
            err = (req.float() - ref_e).abs().max().item()
            amax = ref_e.abs().max().item()
            assert err <= 0.20 * amax, f"{proj} expert {e}: err {err} amax {amax}"


def test_merge_layer_gate_up_shared_scale_survives_loader_reload():
    """A LoRA delta that bumps gate's amax far above up's must still round-trip through the
    ACTUAL loader fusion (which reads ONLY gate's weight_scale_2 and applies it to the whole
    fused gate||up block). gate and up therefore share one per-tensor scale, and the up rows
    must still reproduce the true merged up weights after a loader-style reload.

    Under the old distinct-per-proj-scale schema up was quantized with its own (smaller) scale
    but reloaded with gate's (much larger) one, corrupting every up row; that would fail here."""
    rank = 4
    gate = _random_weights(I, H, 31)
    up = _random_weights(I, H, 32)
    down = _random_weights(H, I, 33)
    fused_gu, _ = _build_fused([gate, up], ("gate_proj", "up_proj"))
    fused_dn, _ = _build_fused([down], ("down_proj",))

    # gate_up LoRA (scattermoe layout) whose delta is ~4x larger in the gate rows [0:I) than the
    # up rows [I:2I), so gate's amax (hence per-tensor scale) ends up far above up's.
    out_f = 2 * I
    g = torch.Generator().manual_seed(99)
    sm_A = torch.randn(rank * E, H, generator=g, dtype=torch.bfloat16)
    sm_B = torch.zeros(out_f, rank * E, dtype=torch.bfloat16)
    sm_B[:I, :] = 4.0 * torch.randn(I, rank * E, generator=g, dtype=torch.bfloat16)
    sm_B[I:, :] = torch.randn(I, rank * E, generator=g, dtype=torch.bfloat16)
    scaling = 1.0
    lora = (sm_A, sm_B, scaling)

    # True merged bf16 fused block (gate||up) the on-disk tensors must reproduce on reload.
    merged_true = merge_mod._merged_fused(fused_gu, lora, E)  # [E, 2I, H]

    out_lora = merge_mod.merge_layer_experts(
        fused_gu, fused_dn, lora, None, GEMMA4_SCHEME
    )

    # Sanity: the delta really did move gate's amax well above up's (so a distinct-scale bug bites).
    for e in range(E):
        gate_amax = merged_true[e, :I, :].abs().max().item()
        up_amax = merged_true[e, I:, :].abs().max().item()
        assert gate_amax > 2.0 * up_amax, f"delta did not separate amax for expert {e}"

    for e in range(E):
        gate_pts = out_lora["gate_proj"][e]["weight_scale_2"].item()
        up_pts = out_lora["up_proj"][e]["weight_scale_2"].item()
        assert gate_pts == up_pts, (
            f"gate and up must share weight_scale_2 for expert {e}: {gate_pts} vs {up_pts}"
        )

    # Reload exactly as the loaders do (fused with gate's pts only) and check the up rows survive.
    reloaded = _loader_fused_gate_up_dequant(out_lora, "gate_proj", "up_proj")
    for e in range(E):
        up_ref = merged_true[e, I:, :].float()
        up_got = reloaded[e, I:, :].float()
        err = (up_got - up_ref).abs().max().item()
        amax = up_ref.abs().max().item()
        assert err <= 0.20 * amax, f"up expert {e}: err {err} amax {amax}"
