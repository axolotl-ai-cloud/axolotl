"""CPU regression tests for the DeepGEMM mxfp4-requant cache under FSDP.

The bug (DSV4 NVFP4 LoRA OOM on 2xB200): the nvfp4->mxfp4 requant cached a full mxfp4 copy per
layer — in a module-level dict AND a per-tensor ``_dg_mxfp4`` attribute. Under FSDP2 the gathered
weight is full+fresh each step and its tensor object outlives the reshard, so both caches
accumulated a full-model mxfp4 copy on every rank (~230GB across 61 layers) on top of the sharded
base -> OOM. The base itself always sharded correctly.

Fix: ``RUNTIME.mxfp4_cache_persist`` is disabled under FSDP (``configure_scattermoe_runtime``), and
``_cached_mxfp4`` then recomputes-and-returns WITHOUT touching either cache, bounding resident mxfp4
to one layer. Single-GPU (persistent param) keeps the cache for speed.

These tests stub the GPU-only ``nvfp4_to_mxfp4_weight`` so they run on CI CPU.
"""

import axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped as dg
from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import (
    RUNTIME,
    configure_scattermoe_runtime,
)
from axolotl.utils.dict import DictDefault


class _FakeWeight:
    """Stand-in for a gathered NVFP4Tensor expert weight (settable ``_dg_mxfp4`` attribute)."""

    def __init__(self, tag):
        self.qdata = (
            tag  # passed straight to the stubbed requant so we can identify the call
        )
        self.scale = None


def _stub_requant(monkeypatch):
    calls = {"n": 0}

    def fake(qdata, scale, per_tensor):
        calls["n"] += 1
        return ("mxfp4", qdata, calls["n"])  # unique per call

    monkeypatch.setattr(dg, "nvfp4_to_mxfp4_weight", fake)
    return calls


def test_configure_runtime_disables_cache_under_fsdp():
    configure_scattermoe_runtime(DictDefault({"fsdp_config": {"fsdp_version": 2}}))
    assert RUNTIME.mxfp4_cache_persist is False
    configure_scattermoe_runtime(DictDefault({}))  # no FSDP -> keep the cache
    assert RUNTIME.mxfp4_cache_persist is True


def test_no_persist_never_caches(monkeypatch):
    """The core regression: under FSDP, neither the module dict nor the per-tensor attr is written,
    and every call recomputes (so nothing accumulates across layers)."""
    calls = _stub_requant(monkeypatch)
    monkeypatch.setattr(RUNTIME, "mxfp4_cache_persist", False)
    cache: dict = {}
    w = _FakeWeight("gate_up")

    r1 = dg._cached_mxfp4(w, None, cache, "gate_up")
    r2 = dg._cached_mxfp4(w, None, cache, "gate_up")

    assert calls["n"] == 2  # recomputed each time, no reuse
    assert r1 != r2  # distinct (uncached) results
    assert cache == {}  # module-level dict never populated
    assert not hasattr(
        w, "_dg_mxfp4"
    )  # per-tensor attr never set -> can't survive reshard


def test_persist_caches_and_reuses(monkeypatch):
    """Single-GPU (persistent param): the module cache is used so the requant runs once per key."""
    calls = _stub_requant(monkeypatch)
    monkeypatch.setattr(RUNTIME, "mxfp4_cache_persist", True)
    cache: dict = {}
    w = _FakeWeight("gate_up")

    r1 = dg._cached_mxfp4(w, None, cache, "gate_up")
    r2 = dg._cached_mxfp4(w, None, cache, "gate_up")

    assert calls["n"] == 1  # computed once, then served from cache
    assert r1 == r2
    assert "gate_up" in cache
