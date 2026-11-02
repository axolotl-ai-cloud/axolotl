"""CPU tests for when the Marlin W4A16 prep caches under multi-process training.

Rank-resident experts (DDP) keep the single-GPU cache; FSDP/DeepSpeed-gathered experts (incl. EP,
which always runs under FSDP2) re-prep each forward. ``prepare_nvfp4_weight_for_marlin`` is stubbed.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.backend as mb
from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import (
    RUNTIME,
    configure_scattermoe_runtime,
)
from axolotl.utils.dict import DictDefault

_EXT = SimpleNamespace(gptq_marlin_repack=None)


class _FakeNV:
    def __init__(self):
        self.qdata = torch.zeros(2, 8, 4, dtype=torch.uint8)
        self.scale = torch.ones(2, 8, 1)
        self.per_tensor_scale = None


@pytest.fixture
def two_ranks(monkeypatch):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)
    monkeypatch.delenv("AXOLOTL_MARLIN_CACHE_MIN_FREE_GB", raising=False)
    calls = {"n": 0}

    def fake_prep(*args):
        calls["n"] += 1
        return ("marlin", calls["n"])

    monkeypatch.setattr(mb, "prepare_nvfp4_weight_for_marlin", fake_prep)
    yield calls
    RUNTIME.reset()


@pytest.mark.parametrize(
    "cfg,gathered",
    [
        ({}, False),
        ({"fsdp_config": {"fsdp_version": 2}}, True),
        ({"deepspeed": "zero3.json"}, True),
    ],
)
def test_configure_experts_gathered(cfg, gathered):
    configure_scattermoe_runtime(DictDefault(cfg))
    assert RUNTIME.experts_gathered is gathered
    RUNTIME.reset()


def test_rank_resident_experts_cache_once(two_ranks):
    RUNTIME.experts_gathered = False
    nv, cache = _FakeNV(), {}
    first = mb._cached_prep(nv, 8, 8, _EXT, cache, "marlin_gate_up")
    assert mb._cached_prep(nv, 8, 8, _EXT, cache, "marlin_gate_up") is first
    assert two_ranks["n"] == 1
    assert "marlin_gate_up_bwd" in cache
    assert nv.qdata.numel() == 0


def test_gathered_experts_reprep_each_forward(two_ranks):
    RUNTIME.experts_gathered = True
    nv, cache = _FakeNV(), {}
    mb._cached_prep(nv, 8, 8, _EXT, cache, "marlin_gate_up")
    mb._cached_prep(nv, 8, 8, _EXT, cache, "marlin_gate_up")
    assert two_ranks["n"] == 2
    assert cache == {}
    assert not hasattr(nv, "_marlin_w4a16")
    assert nv.qdata.numel() > 0
