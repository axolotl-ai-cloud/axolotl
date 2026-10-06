# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for bounded DeepSeek V4 backward autotune profiles."""

import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/axolotl/integrations/kernels/libs/dsv4/autotune_profiles.py"
)
_SPEC = importlib.util.spec_from_file_location("dsv4_profiles", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
profiles = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profiles)


def inputs(kernel):
    meta = dict(M=128, K=16384, N=24, BM=16, D=4096, HC=4, W=8)
    meta.update(
        {
            name: SimpleNamespace(
                dtype="torch.float32", device=SimpleNamespace(type="cuda", index=1)
            )
            for name in profiles._POINTERS[kernel]
        }
    )
    if kernel == "pool":
        meta["D"] = 512
        for name in ("KV", "GATE"):
            meta[name].dtype = "torch.bfloat16"
        tile, sizes, warps, stages = "BD", (32, 64, 128), (2, 4, 8), (3,)
    elif kernel == "collapse":
        tile, sizes, warps, stages = "BD", (256, 512, 1024), (4, 8), (3,)
    else:
        tile, sizes, warps, stages = "BK", (64, 128, 256), (4, 8), (2, 3)
    configs = [
        SimpleNamespace(kwargs={tile: t}, num_warps=w, num_stages=s)
        for t, w, s in product(sizes, warps, stages)
    ]
    return configs, meta


@pytest.mark.parametrize(
    "kernel,cap,count",
    [
        ("rmsln", (10, 0), 4),
        ("rmsln", (12, 1), 1),
        ("collapse", (10, 0), 2),
        ("collapse", (12, 1), 4),
        ("pool", (10, 0), 3),
    ],
)
def test_expected_reductions(kernel, cap, count):
    configs, meta = inputs(kernel)
    selected = profiles.profile_bwd_configs(configs, kernel, cap, meta)
    assert len(selected) == count
    assert all(any(c is original for original in configs) for c in selected)


@pytest.mark.parametrize("kernel", ["rmsln", "collapse", "pool"])
@pytest.mark.parametrize("cap", [(9, 0), (10, 3), (12, 0), (13, 0)])
def test_sparse_hardware_keeps_matrix(kernel, cap):
    configs, meta = inputs(kernel)
    assert profiles.profile_bwd_configs(configs, kernel, cap, meta) is configs


@pytest.mark.parametrize(
    "kernel,changes",
    [
        ("rmsln", {"K": 8192}),
        ("rmsln", {"N": 12}),
        ("rmsln", {"BM": 32}),
        ("rmsln", {"M": 64}),
        ("rmsln", {"M": 32769}),
        ("collapse", {"D": 2048}),
        ("collapse", {"HC": 8}),
        ("pool", {"D": 256}),
        ("pool", {"W": 4}),
        ("pool", {"M": 8193}),
    ],
)
def test_unknown_shapes_keep_matrix(kernel, changes):
    configs, meta = inputs(kernel)
    meta.update(changes)
    assert profiles.profile_bwd_configs(configs, kernel, (10, 0), meta) is configs


@pytest.mark.parametrize("kernel", ["rmsln", "collapse", "pool"])
def test_dtype_signature_and_missing_pointer_fallback(kernel):
    configs, meta = inputs(kernel)
    for pointer in profiles._POINTERS[kernel]:
        original = meta[pointer]
        meta[pointer] = SimpleNamespace(dtype="torch.float16")
        assert profiles.profile_bwd_configs(configs, kernel, (10, 0), meta) is configs
        del meta[pointer]
        assert profiles.profile_bwd_configs(configs, kernel, (10, 0), meta) is configs
        meta[pointer] = original


def test_no_matching_candidate_preserves_fallback():
    configs, meta = inputs("pool")
    configs = [c for c in configs if c.kwargs["BD"] == 128]
    assert profiles.profile_bwd_configs(configs, "pool", (10, 0), meta) is configs


def test_hook_uses_tensor_device_and_merges_keyword_constants(monkeypatch):
    configs, meta = inputs("rmsln")
    monkeypatch.setattr(profiles.torch.version, "hip", None)
    devices = []

    def capability(device):
        devices.append(device)
        return (10, 0)

    monkeypatch.setattr(profiles.torch.cuda, "get_device_capability", capability)
    kwargs = {name: meta.pop(name) for name in ("N", "BM")}
    assert len(profiles.bwd_pruner("rmsln")(configs, meta, **kwargs)) == 4
    assert devices == [meta["STREAMS"].device]


def test_hip_keeps_matrix_without_querying_cuda(monkeypatch):
    configs, meta = inputs("pool")
    monkeypatch.setattr(profiles.torch.version, "hip", "6.3")
    monkeypatch.setattr(
        profiles.torch.cuda,
        "get_device_capability",
        lambda device: pytest.fail("unexpected CUDA query"),
    )
    assert profiles.bwd_pruner("pool")(configs, meta) is configs


@pytest.mark.parametrize("kernel", ["rmsln", "collapse"])
def test_gb10_only_profiles_observed_row_counts(kernel):
    configs, meta = inputs(kernel)
    meta["M"] = 127
    assert profiles.profile_bwd_configs(configs, kernel, (12, 1), meta) is configs
