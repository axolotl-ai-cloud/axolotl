# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU tests for telemetry-derived forward search subsets."""

import importlib.util
import sys
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/axolotl/integrations/kernels/libs/scattermoe_lora/kernels/autotune_profiles.py"
)
_SPEC = importlib.util.spec_from_file_location("autotune_profiles", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
profile_fwd_configs = _MODULE.profile_fwd_configs


@pytest.fixture
def configs():
    return [
        SimpleNamespace(
            kwargs={"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k},
            num_warps=warps,
            num_stages=stages,
        )
        for m, n, k, warps, stages in product(
            (32, 64, 128), (32, 64), (32, 64, 128), (4, 8), (3, 4, 5)
        )
    ]


@pytest.fixture
def meta():
    return {
        "N": 1408,
        "K": 2816,
        "M_BUCKET": 131072,
        "BLOCK_R": 64,
        **{
            name: SimpleNamespace(dtype="torch.bfloat16")
            for name in ("X_ptr", "W_ptr", "Y_ptr", "LA_ptr", "LB_ptr")
        },
    }


@pytest.mark.parametrize(
    "capability,capacity,count",
    [
        ((9, 0), 232448, 12),
        ((10, 0), 232448, 12),
        ((10, 3), 232448, 18),
        ((12, 0), 101376, 30),
    ],
)
@pytest.mark.parametrize("shape", [(1408, 2816), (2816, 704)])
def test_reduced_matrix_keeps_stage_and_warp_choices(
    configs, meta, capability, capacity, count, shape
):
    meta.update(N=shape[0], K=shape[1])
    selected = profile_fwd_configs(configs, capability, capacity, meta)
    assert len(selected) == count
    assert {c.num_stages for c in selected} == {3, 4, 5}
    assert {c.num_warps for c in selected} == {4, 8}
    assert all(any(c is original for original in configs) for c in selected)
    assert len(configs) == 108


@pytest.mark.parametrize(
    "change",
    [
        {"N": 1024, "K": 2048},
        {"M_BUCKET": 0},
        {"M_BUCKET": 655360},
        {"BLOCK_R": 128},
        {"BLOCK_R": 0},
        {"X_ptr": SimpleNamespace(dtype="torch.float16")},
        {"W_ptr": SimpleNamespace(dtype="torch.float32")},
        {"Y_ptr": SimpleNamespace(dtype="torch.float32")},
        {"LA_ptr": SimpleNamespace(dtype="torch.float32")},
        {"LB_ptr": None},
    ],
)
def test_uncovered_inputs_keep_full_matrix(configs, meta, change):
    meta.update(change)
    assert profile_fwd_configs(configs, (9, 0), 232448, meta) is configs


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (12, 1), (13, 0)])
def test_sparse_or_unknown_architecture_keeps_matrix(configs, meta, capability):
    assert profile_fwd_configs(configs, capability, 232448, meta) is configs


def test_capacity_mismatch_keeps_matrix(configs, meta):
    assert profile_fwd_configs(configs, (9, 0), 101376, meta) is configs


def test_resource_pruning_is_not_undone(configs, meta):
    surviving = [c for c in configs if c.num_stages == 3]
    selected = profile_fwd_configs(surviving, (9, 0), 232448, meta)
    assert len(selected) == 4
    assert all(c.num_stages == 3 for c in selected)


def test_empty_profile_intersection_keeps_resource_fallback(configs, meta):
    surviving = [configs[0]]
    assert profile_fwd_configs(surviving, (9, 0), 232448, meta) is surviving


@pytest.mark.parametrize("bucket", [1024, 524288])
def test_observed_bucket_boundaries_are_inclusive(configs, meta, bucket):
    meta["M_BUCKET"] = bucket
    assert len(profile_fwd_configs(configs, (9, 0), 232448, meta)) == 12


@pytest.fixture
def lora_ops(monkeypatch):
    package = ModuleType("_test_scattermoe_kernels")
    package.__path__ = [str(_PATH.parent)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, f"{package.__name__}.autotune_profiles", _MODULE)
    language = SimpleNamespace(constexpr=object())

    def autotune(**options):
        def decorate(fn):
            fn.autotune_keys = options["key"]
            return fn

        return decorate

    triton = SimpleNamespace(
        Config=lambda kwargs, **options: SimpleNamespace(kwargs=kwargs, **options),
        jit=lambda fn: fn,
        heuristics=lambda rules: lambda fn: fn,
        autotune=autotune,
        language=language,
    )
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=object,
            cuda=SimpleNamespace(get_device_capability=lambda: (12, 0)),
        ),
    )
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.lora_ops", _PATH.with_name("lora_ops.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_get_smem_capacity", lambda: 101376)
    return module


@pytest.mark.parametrize("rank", [16, 32, 64, 128])
def test_forward_resource_pruning_uses_keyword_rank(lora_ops, meta, rank):
    configs = lora_ops._scatter2scatter_lora_configs()
    meta["BLOCK_R"] = rank
    from_keywords = lora_ops._prune_fwd_configs(configs, {}, **meta)
    from_named = lora_ops._prune_fwd_configs(configs, meta)
    assert from_keywords == from_named
    assert from_keywords
    for config in from_keywords:
        m, n, k = (config.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        smem = lora_ops._estimate_smem_usage(config.num_stages, m, n, k)
        smem += config.num_stages * rank * k * 2 + n * rank * 2
        assert smem <= 101376 - lora_ops._SMEM_SLACK


def test_forward_cache_distinguishes_rank(lora_ops):
    assert "BLOCK_R" in lora_ops._scatter2scatter_lora.autotune_keys


@pytest.fixture
def mx_meta(meta):
    meta.update(N=6144, K=2048, M_BUCKET=131072)
    for name, dtype in (
        ("DY_ptr", "torch.bfloat16"),
        ("DX_ptr", "torch.bfloat16"),
        ("Wp_ptr", "torch.uint8"),
        ("Ws_ptr", "torch.float32"),
        ("Codebook_ptr", "torch.float32"),
    ):
        meta[name] = SimpleNamespace(dtype=dtype)
    return meta


@pytest.mark.parametrize(
    "shape,bucket",
    [
        ((6144, 512), 1024),
        ((1024, 6144), 8192),
        ((6144, 2048), 8192),
        ((4096, 6144), 262144),
    ],
)
def test_mx_subset_is_within_safety_matrix(lora_ops, mx_meta, shape, bucket):
    mx_meta.update(N=shape[0], K=shape[1], M_BUCKET=bucket)
    configs = lora_ops._scatter2scatter_lora_dX_mx_configs()
    selected = _MODULE.profile_dx_mx_configs(configs, (9, 0), 232448, mx_meta)
    assert len(configs) == 16
    assert len(selected) == 4
    assert all(c in configs for c in selected)


@pytest.mark.parametrize(
    "change",
    [
        {"N": 1},
        {"M_BUCKET": 4096},
        {"M_BUCKET": 393216},
        {"BLOCK_R": 128},
        {"Ws_ptr": SimpleNamespace(dtype="torch.float8_e4m3fn")},
    ],
)
def test_mx_uncovered_inputs_keep_safety_matrix(lora_ops, mx_meta, change):
    mx_meta.update(change)
    configs = lora_ops._scatter2scatter_lora_dX_mx_configs()
    assert _MODULE.profile_dx_mx_configs(configs, (9, 0), 232448, mx_meta) is configs


@pytest.mark.parametrize(
    "capability,smem", [((10, 0), 232448), ((12, 0), 101376), ((9, 0), 101376)]
)
def test_mx_uncovered_hardware_keeps_safety_matrix(lora_ops, mx_meta, capability, smem):
    configs = lora_ops._scatter2scatter_lora_dX_mx_configs()
    assert _MODULE.profile_dx_mx_configs(configs, capability, smem, mx_meta) is configs


@pytest.mark.parametrize("rank", [16, 64, 128, 256])
def test_mx_profiles_obey_resource_limits(lora_ops, mx_meta, monkeypatch, rank):
    monkeypatch.setattr(lora_ops.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(lora_ops, "_get_smem_capacity", lambda: 232448)
    mx_meta["BLOCK_R"] = rank
    configs = lora_ops._scatter2scatter_lora_dX_mx_configs()
    selected = lora_ops._prune_dX_mx_configs(configs, {}, **mx_meta)
    assert selected
    for c in selected:
        m, k, n = (c.kwargs[name] for name in ("BLOCK_M", "BLOCK_K", "BLOCK_N"))
        smem = lora_ops._estimate_smem_usage(c.num_stages, m, k, n)
        smem += 2 * c.num_stages * k * n + 2 * c.num_stages * n * rank + 2 * rank * k
        assert smem <= lora_ops._DX_MX_VALIDATED_SMEM - lora_ops._SMEM_SLACK
    assert "BLOCK_R" in lora_ops._scatter2scatter_lora_dX_mx.autotune_keys


def test_mx_single_config_override_is_preserved(lora_ops, mx_meta, monkeypatch):
    monkeypatch.setenv("AXOLOTL_MX_DX_SAFE_CONFIG", "1")
    configs = lora_ops._scatter2scatter_lora_dX_mx_configs()
    assert len(configs) == 1
    assert _MODULE.profile_dx_mx_configs(configs, (9, 0), 232448, mx_meta) is configs
