# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the gemma4 NVFP4 MoE-expert WeightConverter shim.

``Nvfp4ExpertsDeserialize`` fuses the per-expert NVFP4 tensors a
``nvidia/Gemma-4-...-NVFP4`` checkpoint ships (separate ``gate_proj`` /
``up_proj`` / ``down_proj`` under ``...experts.E.*``) into the single fused
``gate_up_proj`` / ``down_proj`` ``NVFP4Tensor`` that ``Gemma4TextExperts``
expects, and assigns it to the module in place (mirroring transformers'
``Mxfp4Deserialize``).

The fusion is pure tensor stacking/concatenation of *already-quantized* bytes,
so the fused result must be **bit-exact** (maxerr 0.0) against the reference
fusion — that's the core correctness contract these tests pin.

The converter logic is dtype-only (no GPU kernels), but ``NVFP4Tensor`` /
``to_nvfp4`` quantization runs on CUDA, so the suite is CUDA-gated.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_weight_converter import (  # noqa: E402
    Nvfp4ExpertsDeserialize,
    nvfp4_experts_weight_converters,
    register_gemma4_nvfp4_converters,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (  # noqa: E402
    is_nvfp4_param,
)

DEV = "cuda"
# small but realistic gemma4-MoE-ish shapes (H % 16 == 0, I % 16 == 0)
E, H, I = 4, 128, 64
BLOCK = 16


def _quant(W: torch.Tensor, pts: torch.Tensor):
    """Per-expert NVFP4 of a 2D weight with a fixed per-tensor scale (as the
    checkpoint stores it)."""
    return NVFP4Tensor.to_nvfp4(W, block_size=BLOCK, per_tensor_scale=pts)


def _make_checkpoint_experts(seed: int = 0):
    """Build E experts of raw per-expert NVFP4 tensors exactly as the checkpoint
    stores them (separate gate/up/down, each its own qdata + e4m3 scale and its own
    per-tensor weight_scale_2 scalar). Per-expert scales are DISTINCT so the tests pin
    that the converter keeps each expert's scale (not expert-0's for all). Returns the
    fused per-expert scale tensor [E,1,1] (the converter's output shape)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    experts = []
    pts_each = []
    for e in range(E):
        pts = torch.tensor(
            0.5 + 0.1 * e, device=DEV, dtype=torch.float32
        )  # distinct per expert
        pts_each.append(pts)
        Wg = torch.randn(I, H, generator=g, device=DEV, dtype=torch.bfloat16)
        Wu = torch.randn(I, H, generator=g, device=DEV, dtype=torch.bfloat16)
        Wd = torch.randn(H, I, generator=g, device=DEV, dtype=torch.bfloat16)
        experts.append(
            {
                "gate_proj": _quant(Wg, pts),
                "up_proj": _quant(Wu, pts),
                "down_proj": _quant(Wd, pts),
            }
        )
    pts_per_expert = torch.stack(pts_each).view(
        E, 1, 1
    )  # matches converter output shape
    return experts, pts_per_expert


def _input_dict(experts, projs, full_layer_name):
    """Replicate transformers' ``WeightConverter.materialize_tensors`` output:
    keyed by source-pattern string, value = list of per-expert raw tensors."""
    d: dict[str, list[torch.Tensor]] = {}
    for proj in projs:
        d[f"experts.*.{proj}.weight"] = [e[proj].qdata for e in experts]
        d[f"experts.*.{proj}.weight_scale"] = [e[proj].scale for e in experts]
        d[f"experts.*.{proj}.weight_scale_2"] = [
            e[proj].per_tensor_scale for e in experts
        ]
    return d


def _ref_gate_up(experts):
    qdata = torch.cat(
        [
            torch.stack([e["gate_proj"].qdata for e in experts], 0),
            torch.stack([e["up_proj"].qdata for e in experts], 0),
        ],
        dim=1,
    )
    scale = torch.cat(
        [
            torch.stack([e["gate_proj"].scale for e in experts], 0),
            torch.stack([e["up_proj"].scale for e in experts], 0),
        ],
        dim=1,
    )
    return qdata, scale


def _ref_down(experts):
    qdata = torch.stack([e["down_proj"].qdata for e in experts], 0)
    scale = torch.stack([e["down_proj"].scale for e in experts], 0)
    return qdata, scale


def _run_convert(op, experts, proj, projs):
    """Drive the op the same way ``WeightConverter.convert`` does, against a tiny
    module, and return the mutated module."""
    full = f"model.language_model.layers.0.experts.{proj}"
    module = SimpleNamespace()
    setattr(module, proj, nn.Parameter(torch.zeros(1), requires_grad=False))

    missing = {full}
    op.convert(
        _input_dict(experts, projs, full),
        source_patterns=None,
        target_patterns=None,
        full_layer_name=full,
        model=_AttrModel(full, module),
        missing_keys=missing,
    )
    return module, missing


class _AttrModel:
    """Minimal stand-in so ``get_module_from_name(model, full_layer_name)`` returns our
    experts module. That helper rsplits the dotted path and calls
    ``model.get_submodule(parent_path)``; we map any parent path to the single leaf module."""

    def __init__(self, full_layer_name: str, module):
        self._full = full_layer_name
        self._leaf = module

    def get_submodule(self, name):
        return self._leaf


# ---------------------------------------------------------------------------
# correctness: fused NVFP4Tensor is bit-exact vs the reference fusion
# ---------------------------------------------------------------------------


def test_gate_up_fusion_bit_exact():
    experts, pts = _make_checkpoint_experts(seed=1)
    op = Nvfp4ExpertsDeserialize()
    module, missing = _run_convert(
        op, experts, "gate_up_proj", ("gate_proj", "up_proj")
    )

    fused = module.gate_up_proj
    assert is_nvfp4_param(fused), "gate_up_proj must be an NVFP4Tensor"

    ref_qd, ref_sc = _ref_gate_up(experts)
    assert fused.qdata.shape == (E, 2 * I, H // 2)
    assert fused.scale.shape == (E, 2 * I, H // BLOCK)
    # bit-exact: fusion is concatenation of already-quantized bytes -> maxerr 0.0
    assert torch.equal(fused.qdata, ref_qd), "qdata not bit-exact"
    assert torch.equal(fused.scale, ref_sc), "scale not bit-exact"
    # per-expert weight_scale_2 preserved as [E,1,1] (each expert keeps its own scale)
    assert torch.equal(fused.per_tensor_scale.to(torch.float32), pts), (
        "per_tensor_scale mismatch"
    )
    # loader contract: param consumed, requires_grad off
    assert "model.language_model.layers.0.experts.gate_up_proj" not in missing
    assert fused.requires_grad is False


def test_down_fusion_bit_exact():
    experts, pts = _make_checkpoint_experts(seed=2)
    op = Nvfp4ExpertsDeserialize()
    module, missing = _run_convert(op, experts, "down_proj", ("down_proj",))

    fused = module.down_proj
    assert is_nvfp4_param(fused)
    ref_qd, ref_sc = _ref_down(experts)
    assert fused.qdata.shape == (E, H, I // 2)
    assert fused.scale.shape == (E, H, I // BLOCK)
    assert torch.equal(fused.qdata, ref_qd)
    assert torch.equal(fused.scale, ref_sc)
    assert torch.equal(fused.per_tensor_scale.to(torch.float32), pts)


# ---------------------------------------------------------------------------
# correctness: fused tensor dequantizes to cat(dequant(gate), dequant(up))
# (the property the scattermoe forward relies on) at maxerr 0.0
# ---------------------------------------------------------------------------


def test_gate_up_dequant_self_consistent():
    experts, _ = _make_checkpoint_experts(seed=3)
    op = Nvfp4ExpertsDeserialize()
    module, _ = _run_convert(op, experts, "gate_up_proj", ("gate_proj", "up_proj"))

    fused = module.gate_up_proj
    pe = fused.per_tensor_scale.reshape(-1)  # [E] per-expert weight_scale_2
    for e in range(E):
        # torchao's per-expert index (fused[e]) does not slice the [E,1,1] per_tensor_scale, so
        # rebuild the expert-e slice with its own scalar scale (what the real per-expert kernel uses).
        fused_e = NVFP4Tensor(
            fused.qdata[e],
            fused.scale[e],
            BLOCK,
            torch.bfloat16,
            per_tensor_scale=pe[e],
        )
        fused_dq = fused_e.dequantize(torch.bfloat16)
        gate_dq = experts[e]["gate_proj"].dequantize(torch.bfloat16)
        up_dq = experts[e]["up_proj"].dequantize(torch.bfloat16)
        ref = torch.cat([gate_dq, up_dq], dim=0)
        err = (fused_dq - ref).abs().max().item()
        assert err == 0.0, f"expert {e}: dequant maxerr {err} != 0"


def test_down_dequant_self_consistent():
    experts, _ = _make_checkpoint_experts(seed=4)
    op = Nvfp4ExpertsDeserialize()
    module, _ = _run_convert(op, experts, "down_proj", ("down_proj",))

    fused = module.down_proj
    pe = fused.per_tensor_scale.reshape(-1)  # [E] per-expert weight_scale_2
    for e in range(E):
        # torchao's per-expert index does not slice the [E,1,1] per_tensor_scale; rebuild the slice.
        fused_e = NVFP4Tensor(
            fused.qdata[e],
            fused.scale[e],
            BLOCK,
            torch.bfloat16,
            per_tensor_scale=pe[e],
        )
        fused_dq = fused_e.dequantize(torch.bfloat16)
        ref = experts[e]["down_proj"].dequantize(torch.bfloat16)
        err = (fused_dq - ref).abs().max().item()
        assert err == 0.0, f"expert {e}: dequant maxerr {err} != 0"


# ---------------------------------------------------------------------------
# load-assertion: the experts land as NVFP4Tensor after the converter runs
# ---------------------------------------------------------------------------


def test_load_assertion_experts_are_nvfp4():
    """Both fused params on the module are NVFP4Tensor (the post-load invariant
    scattermoe's fused path requires)."""
    experts, _ = _make_checkpoint_experts(seed=5)
    op = Nvfp4ExpertsDeserialize()
    gu_mod, _ = _run_convert(op, experts, "gate_up_proj", ("gate_proj", "up_proj"))
    dn_mod, _ = _run_convert(op, experts, "down_proj", ("down_proj",))
    assert is_nvfp4_param(gu_mod.gate_up_proj)
    assert is_nvfp4_param(dn_mod.down_proj)
    assert gu_mod.gate_up_proj.qdata.dtype == torch.uint8
    assert gu_mod.gate_up_proj.scale.dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# registration wiring: the converters install into the live transformers cache
# ---------------------------------------------------------------------------


def test_registration_installs_converters():
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    register_gemma4_nvfp4_converters()
    mapping = get_checkpoint_conversion_mapping("gemma4_text")
    assert mapping is not None, "gemma4_text conversion mapping not registered"
    # idempotent re-registration must not raise
    register_gemma4_nvfp4_converters()

    converters = nvfp4_experts_weight_converters()
    assert len(converters) == 2
    targets = {c.target_patterns[0] for c in converters}
    assert targets == {"experts.gate_up_proj", "experts.down_proj"}
    # gate_up converter sources the unfused per-expert keys
    gu = next(c for c in converters if "gate_up_proj" in c.target_patterns[0])
    assert any("gate_proj.weight" in s for s in gu.source_patterns)
    assert any("up_proj.weight" in s for s in gu.source_patterns)


def test_end_to_end_from_pretrained_load():
    """Definitive load-assertion: save a tiny synthetic gemma4 NVFP4-modelopt
    checkpoint (per-expert gate/up/down qdata+scale+scalar), register the
    converters, ``from_pretrained``, and assert the experts land as NVFP4Tensor
    with bit-exact fused qdata vs the reference fusion and no leftover keys."""
    import json
    import os
    import shutil
    import tempfile

    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    cfg = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=H,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=32,
        enable_moe_block=True,
        num_experts=E,
        top_k_experts=2,
        moe_intermediate_size=I,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=64,
    )
    base = Gemma4TextModel(cfg).to(torch.bfloat16)
    sd = base.state_dict()

    experts, _ = _make_checkpoint_experts(seed=7)
    new_sd = {
        k: v
        for k, v in sd.items()
        if ".experts.gate_up_proj" not in k and ".experts.down_proj" not in k
    }
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t = experts[e][proj]
            pfx = f"layers.0.experts.{e}.{proj}"
            new_sd[f"{pfx}.weight"] = t.qdata.cpu()
            new_sd[f"{pfx}.weight_scale"] = t.scale.cpu()
            new_sd[f"{pfx}.weight_scale_2"] = t.per_tensor_scale.cpu()

    d = tempfile.mkdtemp()
    try:
        save_file(
            {k: v.contiguous() for k, v in new_sd.items()},
            os.path.join(d, "model.safetensors"),
            metadata={"format": "pt"},
        )
        cfg_dict = cfg.to_dict()
        cfg_dict["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        }
        json.dump(cfg_dict, open(os.path.join(d, "config.json"), "w"))

        register_gemma4_nvfp4_converters()
        loaded = Gemma4TextModel.from_pretrained(d, dtype=torch.bfloat16).to(DEV)
        ex = loaded.layers[0].experts

        assert is_nvfp4_param(ex.gate_up_proj)
        assert is_nvfp4_param(ex.down_proj)
        assert ex.gate_up_proj.qdata.shape == (E, 2 * I, H // 2)
        assert ex.down_proj.qdata.shape == (E, H, I // 2)

        ref_gu_qd, _ = _ref_gate_up(experts)
        ref_dn_qd, _ = _ref_down(experts)
        assert torch.equal(ex.gate_up_proj.qdata, ref_gu_qd.to(DEV))
        assert torch.equal(ex.down_proj.qdata, ref_dn_qd.to(DEV))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_source_pattern_alternation_disambiguates_scales():
    """transformers resolves a checkpoint key via ``re.search`` over the compiled
    ``(?P<g0>..)|(?P<g1>..)`` alternation and takes the first non-None group. The
    patterns are not end-anchored for many-to-one converters, so ``...weight`` would
    substring-match inside ``...weight_scale``/``...weight_scale_2`` unless the more
    specific suffixes are ordered first. This pins that ordering for both converters."""
    for conv in nvfp4_experts_weight_converters():
        rx = conv.compiled_sources

        # which source pattern does each concrete checkpoint key resolve to?
        def _resolve(key, rx=rx, conv=conv):  # bind loop vars per-iteration
            m = rx.search(key)
            assert m is not None, f"{key} matched no source pattern"
            gname = next(n for n, v in m.groupdict().items() if v is not None)
            return conv.source_patterns[int(gname[1:])]

        proj = "down_proj" if "down_proj" in conv.target_patterns[0] else "gate_proj"
        assert _resolve(f"experts.3.{proj}.weight").endswith(f"{proj}.weight")
        assert _resolve(f"experts.3.{proj}.weight_scale").endswith(
            f"{proj}.weight_scale"
        )
        assert _resolve(f"experts.3.{proj}.weight_scale_2").endswith(
            f"{proj}.weight_scale_2"
        )
