"""Deterministic s2s tile rule: pure-function selection, table fallback, no benchmarking."""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import (
    _LEGACY,
    _s2s_pick_cfg,
    _scatter2scatter_configs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_rule_is_pure_and_table_driven(monkeypatch):
    from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import ops

    a = _s2s_pick_cfg(4096, 6144, "bfloat16")
    assert a == _s2s_pick_cfg(4096, 6144, "bfloat16")  # pure fn -> rank determinism
    assert a["BLOCK_K"] == _LEGACY["BLOCK_K"]  # table ships empty -> legacy everywhere
    entry = {
        "k_mod": 64,
        "n_min": 4096,
        "dtypes": ("bfloat16",),
        "cfg": {"BLOCK_N": 128, "BLOCK_K": 64, "num_stages": 4, "num_warps": 4},
    }
    monkeypatch.setattr(ops, "_S2S_TILE_TABLE", [entry])
    assert _s2s_pick_cfg(4096, 6144, "bfloat16")["BLOCK_K"] == 64  # matches
    assert _s2s_pick_cfg(4096, 6100, "bfloat16")["BLOCK_K"] == 32  # K%64 fails
    assert _s2s_pick_cfg(2048, 6144, "bfloat16")["BLOCK_K"] == 32  # n_min fails
    assert _s2s_pick_cfg(4096, 6144, "float32")["BLOCK_K"] == 32  # dtype gate fails


def test_grid_contains_legacy_first():
    cfgs = _scatter2scatter_configs()
    c0 = cfgs[0]
    assert c0.kwargs["BLOCK_N"] == _LEGACY["BLOCK_N"]
    assert c0.kwargs["BLOCK_K"] == _LEGACY["BLOCK_K"]
