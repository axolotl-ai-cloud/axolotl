"""CPU-only tests for adapter matching/dispatch and non-expert quantization resolution."""

from unittest.mock import MagicMock

import pytest

import axolotl.integrations.kernels.adapters as adapters_mod
import axolotl.integrations.kernels.adapters.gemma4 as gemma4_mod
import axolotl.integrations.kernels.plugin as plugin_mod
from axolotl.integrations.kernels.adapters import get_active_adapters
from axolotl.integrations.kernels.adapters.dsv4 import DSV4Adapter
from axolotl.integrations.kernels.adapters.gemma4 import (
    Gemma4Adapter,
    resolve_nonexpert_quantization,
)
from axolotl.integrations.kernels.adapters.qwen3_moe import Qwen3MoeAdapter
from axolotl.integrations.kernels.plugin import KernelsPlugin


class Cfg(dict):
    """Minimal cfg supporting both attribute and .get access (like DictDefault)."""

    def __getattr__(self, k):
        return self.get(k)


def test_dsv4_adapter_matches_only_dsv4():
    active = get_active_adapters(Cfg(use_dsv4_kernels=True))
    assert [type(a) for a in active] == [DSV4Adapter]
    assert get_active_adapters(Cfg(use_dsv4_kernels=False)) == []


def test_gemma4_adapter_matches(monkeypatch):
    monkeypatch.setattr(gemma4_mod, "is_gemma4_nvfp4_modelopt", lambda cfg: True)
    active = get_active_adapters(Cfg(use_scattermoe=True))
    assert [type(a) for a in active] == [Gemma4Adapter]
    # requires scattermoe
    assert get_active_adapters(Cfg(use_scattermoe=False)) == []


def test_qwen3_moe_adapter_matches(monkeypatch):
    import axolotl.integrations.kernels.adapters.qwen3_moe as qwen3_moe_mod

    monkeypatch.setattr(qwen3_moe_mod, "is_qwen3_moe_nvfp4_modelopt", lambda cfg: True)
    # either expert backend activates it
    active = get_active_adapters(Cfg(use_scattermoe=True))
    assert Qwen3MoeAdapter in [type(a) for a in active]
    active = get_active_adapters(Cfg(use_sonicmoe=True))
    assert [type(a) for a in active] == [Qwen3MoeAdapter]
    # requires an expert backend
    assert get_active_adapters(Cfg()) == []


def test_moe_nvfp4_generic_adapter_matches(monkeypatch):
    import axolotl.integrations.kernels.adapters.nvfp4_moe as nvfp4_moe_mod
    from axolotl.integrations.kernels.adapters.nvfp4_moe import MoeNvfp4Adapter

    monkeypatch.setattr(nvfp4_moe_mod, "is_moe_nvfp4_modelopt", lambda cfg: True)
    # either expert backend activates the generic gate
    assert [type(a) for a in get_active_adapters(Cfg(use_sonicmoe=True))] == [
        MoeNvfp4Adapter
    ]
    assert MoeNvfp4Adapter in [
        type(a) for a in get_active_adapters(Cfg(use_scattermoe=True))
    ]
    # requires an expert backend
    assert get_active_adapters(Cfg()) == []


def test_moe_nvfp4_rejects_unsupported_expert_layout():
    from axolotl.integrations.kernels.adapters.nvfp4_moe import MoeNvfp4Adapter

    adapter = MoeNvfp4Adapter()
    # Mixtral-style w1/w2/w3 is not fusable by the routed converter -> loud failure
    with pytest.raises(ValueError, match="unsupported routed NVFP4 expert layout"):
        adapter._validate_supported_layout(
            Cfg(), "mixtral", {"routed_projs": ["w1", "w2", "w3"]}
        )
    # gate/up/down passes; a non-routed-only checkpoint (no routed projs) also passes
    adapter._validate_supported_layout(
        Cfg(), "qwen3_moe", {"routed_projs": ["gate_proj", "up_proj", "down_proj"]}
    )
    adapter._validate_supported_layout(Cfg(), "qwen3_moe", {"routed_projs": []})


def test_gemma4_match_failure_is_swallowed(monkeypatch):
    def boom(cfg):
        raise RuntimeError("network down")

    monkeypatch.setattr(gemma4_mod, "is_gemma4_nvfp4_modelopt", boom)
    # a raising matcher must not break adapter discovery
    assert get_active_adapters(Cfg(use_scattermoe=True)) == []


def test_plugin_dispatches_to_active_adapters(monkeypatch):
    a1, a2 = MagicMock(), MagicMock()
    monkeypatch.setattr(plugin_mod, "get_active_adapters", lambda cfg: [a1, a2])
    plugin = KernelsPlugin()
    cfg = Cfg(use_scattermoe=False, use_sonicmoe=False)  # skip generic registration
    model = object()

    plugin.pre_model_load(cfg)
    plugin.pre_lora_load(cfg, model)
    plugin.post_model_load(cfg, model)

    for a in (a1, a2):
        a.pre_model_load.assert_called_once_with(cfg)
        a.pre_lora_load.assert_called_once_with(cfg, model)
        a.post_model_load.assert_called_once_with(cfg, model)


def test_plugin_caches_adapter_discovery(monkeypatch):
    calls = {"n": 0}

    def counting(cfg):
        calls["n"] += 1
        return []

    monkeypatch.setattr(plugin_mod, "get_active_adapters", counting)
    plugin = KernelsPlugin()
    cfg = Cfg(use_scattermoe=False, use_sonicmoe=False)
    plugin.pre_model_load(cfg)
    plugin.pre_lora_load(cfg, object())
    plugin.post_model_load(cfg, object())
    assert calls["n"] == 1  # discovered once, cached across hooks


def test_resolve_nonexpert_quantization_intent_and_legacy():
    # intent field takes precedence
    assert (
        resolve_nonexpert_quantization(Cfg(nonexpert_quantization="fp8_blockwise"))
        == "fp8"
    )
    assert resolve_nonexpert_quantization(Cfg(nonexpert_quantization="nf4")) == "nf4"
    assert resolve_nonexpert_quantization(Cfg(nonexpert_quantization="none")) is None
    assert resolve_nonexpert_quantization(Cfg(nonexpert_quantization="bf16")) is None
    # legacy flags
    assert resolve_nonexpert_quantization(Cfg(gemma4_fp8_nonexpert=True)) == "fp8"
    assert resolve_nonexpert_quantization(Cfg(gemma4_nf4_nonexpert=True)) == "nf4"
    # nothing set
    assert resolve_nonexpert_quantization(Cfg()) is None


def test_adapters_module_get_active_is_used(monkeypatch):
    # sanity: the symbol the plugin caches resolves to the adapters module function
    assert plugin_mod.get_active_adapters is adapters_mod.get_active_adapters
