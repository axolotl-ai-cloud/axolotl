"""Unit tests for the accelerate fp8 MoE router exclusion monkeypatch."""

import accelerate.accelerator as acc
import torch.nn as nn

import axolotl.monkeypatch.accelerate.float8_moe_filter as float8_moe_filter
from axolotl.monkeypatch.accelerate.float8_moe_filter import (
    _is_router,
    patch_fp8_exclude_moe_router,
)


class Expert(nn.Module):
    """Expert MLP whose projections should stay eligible for fp8."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 64, bias=False)
        self.down_proj = nn.Linear(64, 32, bias=False)


class MoeBlock(nn.Module):
    """MoE block with a router gate linear."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(32, 16, bias=False)
        self.experts = nn.ModuleList([Expert() for _ in range(2)])


class ToyMoeModel(nn.Module):
    """Minimal model with first/last linears around an MoE block."""

    def __init__(self):
        super().__init__()
        self.embed_proj = nn.Linear(32, 32, bias=False)
        self.mlp = MoeBlock()
        self.lm_head = nn.Linear(32, 32, bias=False)


def test_is_router_matches_router_segments():
    assert _is_router("model.layers.0.mlp.gate")
    assert _is_router("model.layers.0.block_sparse_moe.gate")
    assert _is_router("model.layers.0.feed_forward.router")
    # routers wrapped in a submodule (DBRX, HunYuan, Switch/NLLB)
    assert _is_router("transformer.blocks.0.ffn.router.layer")
    assert _is_router("model.layers.0.mlp.gate.wg")
    assert _is_router("encoder.block.1.layer.1.mlp.router.classifier")


def test_is_router_skips_expert_projections():
    assert not _is_router("model.layers.0.mlp.experts.0.gate_proj")
    assert not _is_router("model.layers.0.mlp.gate_up_proj")
    assert not _is_router("model.layers.0.self_attn.q_proj")
    assert not _is_router("")


def _install_capture(monkeypatch):
    captured = {}

    def fake_convert(model, config=None, module_filter_func=None):
        captured["filter"] = module_filter_func

    monkeypatch.setattr(float8_moe_filter, "_PATCHED", False)
    monkeypatch.setattr(acc, "convert_model_to_fp8_ao", fake_convert)
    patch_fp8_exclude_moe_router()
    assert acc.convert_model_to_fp8_ao is not fake_convert
    return captured


def test_patched_filter_excludes_router_and_first_last(monkeypatch):
    captured = _install_capture(monkeypatch)

    model = ToyMoeModel()
    acc.convert_model_to_fp8_ao(model)
    module_filter = captured["filter"]

    assert module_filter(model.mlp.gate, "mlp.gate") is False
    assert module_filter(model.embed_proj, "embed_proj") is False
    assert module_filter(model.lm_head, "lm_head") is False
    assert (
        module_filter(model.mlp.experts[0].gate_proj, "mlp.experts.0.gate_proj") is True
    )
    assert (
        module_filter(model.mlp.experts[0].down_proj, "mlp.experts.0.down_proj") is True
    )


def test_patched_filter_wraps_user_supplied_filter(monkeypatch):
    captured = _install_capture(monkeypatch)

    model = ToyMoeModel()
    acc.convert_model_to_fp8_ao(model, module_filter_func=lambda module, fqn: True)
    module_filter = captured["filter"]

    assert module_filter(model.mlp.gate, "mlp.gate") is False
    assert module_filter(model.embed_proj, "embed_proj") is True
