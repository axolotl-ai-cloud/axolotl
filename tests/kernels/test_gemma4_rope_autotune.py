# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU tests for backward RMSNorm/RoPE autotune profile selection."""

from itertools import product
from types import SimpleNamespace

import pytest

from axolotl.kernels import gemma4_rope_autotune as profiles


@pytest.fixture
def configs():
    return [
        SimpleNamespace(num_warps=w, num_stages=s)
        for w, s in product((2, 4, 8, 16), (1, 2, 3))
    ]


@pytest.fixture
def meta():
    data = {
        name: SimpleNamespace(dtype="torch.bfloat16")
        for name in ("dY_ptr", "dX_ptr", "X_ptr", "W_ptr", "COS_ptr", "SIN_ptr")
    }
    data.update(
        {
            name: SimpleNamespace(dtype="torch.float32")
            for name in ("RSTD_ptr", "dW_ptr")
        }
    )
    data.update(n_cols=256, n_rot=128, HAS_WEIGHT=True, UNIT_OFFSET=False)
    data["X_ptr"].device = SimpleNamespace(type="cuda", index=1)
    return data


@pytest.mark.parametrize(
    "capability,n_cols,count",
    [
        ((8, 0), 256, 7),
        ((8, 0), 512, 7),
        ((8, 6), 256, 6),
        ((8, 9), 256, 7),
        ((9, 0), 256, 8),
        ((10, 0), 256, 4),
        ((10, 0), 512, 6),
        ((10, 3), 256, 3),
        ((10, 3), 512, 4),
        ((12, 0), 256, 6),
    ],
)
def test_profile_counts(configs, meta, capability, n_cols, count):
    meta["n_cols"] = n_cols
    selected = profiles.profile_rope_bwd_configs(configs, capability, meta)
    assert len(selected) == count
    assert {c.num_stages for c in selected} == {1, 2, 3}
    assert all(any(c is original for original in configs) for c in selected)
    assert len(configs) == 12


@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (9, 0), (12, 0)])
def test_diverse_512_winners_keep_full_matrix(configs, meta, capability):
    meta["n_cols"] = 512
    assert profiles.profile_rope_bwd_configs(configs, capability, meta) is configs


@pytest.mark.parametrize("capability", [(7, 5), (12, 1), (13, 0)])
def test_uncovered_architectures_keep_full_matrix(configs, meta, capability):
    assert profiles.profile_rope_bwd_configs(configs, capability, meta) is configs


@pytest.mark.parametrize("n_cols", [16, 32, 64, 128, 1024])
def test_uncovered_head_dimensions_keep_full_matrix(configs, meta, n_cols):
    meta["n_cols"] = n_cols
    assert profiles.profile_rope_bwd_configs(configs, (10, 3), meta) is configs


@pytest.mark.parametrize(
    "pointer",
    ["dY_ptr", "dX_ptr", "X_ptr", "W_ptr", "COS_ptr", "SIN_ptr", "RSTD_ptr", "dW_ptr"],
)
def test_uncovered_dtype_signature_keeps_matrix(configs, meta, pointer):
    meta[pointer] = SimpleNamespace(dtype="torch.float16")
    assert profiles.profile_rope_bwd_configs(configs, (10, 3), meta) is configs


def test_missing_weight_keeps_matrix(configs, meta):
    meta["HAS_WEIGHT"] = False
    assert profiles.profile_rope_bwd_configs(configs, (10, 3), meta) is configs


def test_empty_intersection_keeps_supplied_fallback(configs, meta):
    configs = [c for c in configs if c.num_warps == 16]
    assert profiles.profile_rope_bwd_configs(configs, (10, 3), meta) is configs


def test_hopper_rare_winners_are_retained(configs, meta):
    selected = profiles.profile_rope_bwd_configs(configs, (9, 0), meta)
    assert {(8, 3), (16, 2)} <= {(c.num_warps, c.num_stages) for c in selected}


def test_hook_merges_metadata_and_uses_tensor_device(configs, meta, monkeypatch):
    monkeypatch.setattr(profiles.torch.version, "hip", None)
    devices = []

    def capability(device):
        devices.append(device)
        return (10, 3)

    monkeypatch.setattr(profiles.torch.cuda, "get_device_capability", capability)
    kwargs = {name: meta.pop(name) for name in ("HAS_WEIGHT", "UNIT_OFFSET")}
    assert len(profiles.prune_rope_bwd_configs(configs, meta, **kwargs)) == 3
    assert devices == [meta["X_ptr"].device]


@pytest.mark.parametrize("hip,device_type", [("6.3", "cuda"), (None, "cpu")])
def test_non_cuda_backend_keeps_matrix(configs, meta, monkeypatch, hip, device_type):
    monkeypatch.setattr(profiles.torch.version, "hip", hip)
    meta["X_ptr"].device.type = device_type

    def unexpected_query(device):
        pytest.fail("must not query CUDA capabilities for another backend")

    monkeypatch.setattr(profiles.torch.cuda, "get_device_capability", unexpected_query)
    assert profiles.prune_rope_bwd_configs(configs, meta) is configs
