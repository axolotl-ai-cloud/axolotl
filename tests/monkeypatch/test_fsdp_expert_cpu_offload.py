"""Unit tests for selective MoE expert CPU offload (fsdp_expert_cpu_offload).

Pure-CPU, no dist init: ``fully_shard`` is monkeypatched to a no-op so the block
detection, offload-policy selection, and offloaded-name capture can be asserted
without a real device mesh.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

from axolotl.monkeypatch.accelerate.fsdp2 import (
    _detect_moe_blocks,
    shard_moe_experts_cpu_offload,
)


class Experts(nn.Module):
    """Fused-experts layout: 3D gate_up_proj / down_proj (scattermoe canonical)."""

    def __init__(self, e: int = 8, i: int = 16, h: int = 8) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.zeros(e, h, i))


class MoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(8, 8)  # router
        self.experts = Experts()
        self.shared_experts = nn.Linear(8, 8)


class Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attn()
        self.mlp = MoEBlock()


class DenseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attn()
        self.mlp = nn.Linear(8, 8)


class MoEModel(nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])


def _patch_fully_shard(monkeypatch):
    import torch.distributed.fsdp as fsdp_module

    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(fsdp_module, "fully_shard", fake_fully_shard)
    return calls


def test_detect_returns_parent_block_not_experts():
    model = MoEModel(n_layers=2)
    blocks = list(_detect_moe_blocks(model))
    assert len(blocks) == 2
    for name, block in blocks:
        assert isinstance(block, MoEBlock)  # the block, not the Experts submodule
        assert name.endswith(".mlp")


def test_dense_only_model_detects_nothing():
    model = nn.ModuleList([DenseLayer(), DenseLayer()])
    assert list(_detect_moe_blocks(model)) == []


def test_offload_wraps_blocks_with_cpu_policy(monkeypatch):
    calls = _patch_fully_shard(monkeypatch)
    model = MoEModel(n_layers=2)

    n_blocks, names = shard_moe_experts_cpu_offload(
        model,
        fully_shard_kwargs={
            "mesh": None,
            "reshard_after_forward": True,
            "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        },
        pin_memory=True,
    )

    assert n_blocks == 2
    # every wrapped module is a MoEBlock, each with a fresh CPUOffloadPolicy
    assert len(calls) == 2
    for module, kwargs in calls:
        assert isinstance(module, MoEBlock)
        assert isinstance(kwargs["offload_policy"], CPUOffloadPolicy)
        assert kwargs["offload_policy"].pin_memory is True
        # outer wrap's policy/mesh are inherited unchanged
        assert kwargs["reshard_after_forward"] is True
        assert kwargs["mp_policy"].param_dtype == torch.bfloat16


def test_offloaded_names_cover_router_experts_shared_only(monkeypatch):
    _patch_fully_shard(monkeypatch)
    model = MoEModel(n_layers=2)

    _n, names = shard_moe_experts_cpu_offload(model, fully_shard_kwargs={})

    # experts + router + shared expert are offloaded; attention is not
    assert "layers.0.mlp.experts.gate_up_proj" in names
    assert "layers.0.mlp.experts.down_proj" in names
    assert "layers.0.mlp.gate.weight" in names
    assert "layers.0.mlp.shared_experts.weight" in names
    assert not any(".self_attn." in n for n in names)
    # both layers covered, nothing outside the mlp blocks
    assert all(".mlp." in n for n in names)
    assert len({n.split(".mlp.")[0] for n in names}) == 2


def test_pin_memory_false_is_propagated(monkeypatch):
    calls = _patch_fully_shard(monkeypatch)
    model = MoEModel(n_layers=1)

    shard_moe_experts_cpu_offload(model, fully_shard_kwargs={}, pin_memory=False)

    assert calls[0][1]["offload_policy"].pin_memory is False


def test_ignored_params_kwarg_is_dropped(monkeypatch):
    calls = _patch_fully_shard(monkeypatch)
    model = MoEModel(n_layers=1)

    shard_moe_experts_cpu_offload(
        model,
        fully_shard_kwargs={"ignored_params": {object()}},
    )

    # ignored_params is an outer-wrap concern; it must not leak into the per-block wrap
    assert "ignored_params" not in calls[0][1]


def test_no_moe_blocks_returns_empty():
    model = nn.ModuleList([DenseLayer()])
    n_blocks, names = shard_moe_experts_cpu_offload(model, fully_shard_kwargs={})
    assert n_blocks == 0
    assert names == set()
