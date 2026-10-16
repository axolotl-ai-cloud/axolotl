# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU round-trip for the Triton autotune warm-start cache."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from axolotl.integrations.kernels import autotune_cache
from axolotl.integrations.kernels.autotune_collector import collect_autotune_configs


class FakeConfig:
    def __init__(self, kwargs, num_warps=4, num_stages=3, num_ctas=1):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas


def _kernel(name):
    def fn():
        pass

    fn.__name__ = name
    return fn


def _autotuner(name, configs, keys):
    return SimpleNamespace(cache={}, base_fn=_kernel(name), configs=configs, keys=keys)


@pytest.fixture
def kernel_module(monkeypatch, isolated_kernel_modules):
    configs = [
        FakeConfig({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, 4, 4),
        FakeConfig({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, 4, 4),
    ]
    module = ModuleType("axolotl.fake.scattermoe_lora.lora_ops")
    module._scatter2scatter_lora = _autotuner(
        "_scatter2scatter_lora", configs, ["M_BUCKET", "N", "K", "BLOCK_R"]
    )
    module._other = _autotuner("_other", configs[:1], ["K"])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        autotune_cache, "_gpu_identity", lambda: {"gpu_name": "NVIDIA B200"}
    )
    return module


def _measured_key():
    return (131072, 5120, 2560, 32, "torch.bfloat16", "torch.int32")


def test_round_trip_seeds_matching_autotuner(kernel_module, tmp_path):
    tuner = kernel_module._scatter2scatter_lora
    tuner.cache[_measured_key()] = tuner.configs[1]
    records = collect_autotune_configs()
    assert len(records) == 1
    path = tmp_path / "b200.json"
    assert autotune_cache.write_autotune_cache(str(path), records) == 1
    data = json.loads(path.read_text())
    assert data["gpu_name"] == "NVIDIA B200"
    assert data["records"][0]["key"]["_extra"] == ["torch.bfloat16", "torch.int32"]

    tuner.cache.clear()
    assert autotune_cache.load_autotune_cache(str(path)) == 1
    assert tuner.cache == {_measured_key(): tuner.configs[1]}
    assert kernel_module._other.cache == {}


def test_measured_entries_and_other_gpus_are_left_alone(kernel_module, tmp_path):
    tuner = kernel_module._scatter2scatter_lora
    tuner.cache[_measured_key()] = tuner.configs[1]
    path = tmp_path / "b200.json"
    autotune_cache.write_autotune_cache(str(path), collect_autotune_configs())

    tuner.cache[_measured_key()] = tuner.configs[0]
    applied, skipped = autotune_cache.seed_autotune_caches(
        autotune_cache.read_autotune_cache(str(path))["records"]
    )
    assert applied == 0 and skipped["already in cache"] == 1
    assert tuner.cache[_measured_key()] is tuner.configs[0]

    tuner.cache.clear()
    autotune_cache._gpu_identity = lambda: {"gpu_name": "NVIDIA H100"}
    assert autotune_cache.load_autotune_cache(str(path)) == 0
    assert tuner.cache == {}


def test_changed_config_matrix_is_skipped(kernel_module, tmp_path):
    tuner = kernel_module._scatter2scatter_lora
    tuner.cache[_measured_key()] = tuner.configs[1]
    path = tmp_path / "b200.json"
    autotune_cache.write_autotune_cache(str(path), collect_autotune_configs())
    tuner.cache.clear()
    tuner.configs.append(FakeConfig({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}))
    applied, skipped = autotune_cache.seed_autotune_caches(
        autotune_cache.read_autotune_cache(str(path))["records"]
    )
    assert applied == 0 and skipped["config matrix changed"] == 1


def test_write_merges_with_existing_file(kernel_module, tmp_path):
    tuner = kernel_module._scatter2scatter_lora
    path = tmp_path / "b200.json"
    tuner.cache[_measured_key()] = tuner.configs[1]
    autotune_cache.write_autotune_cache(str(path), collect_autotune_configs())
    tuner.cache.clear()
    tuner.cache[(262144, 5120, 2560, 32, "torch.bfloat16", "torch.int32")] = (
        tuner.configs[0]
    )
    assert (
        autotune_cache.write_autotune_cache(str(path), collect_autotune_configs()) == 2
    )


def test_timing_capture_records_every_benchmarked_config(kernel_module, tmp_path):
    tuner = kernel_module._scatter2scatter_lora
    tuner.nargs = {}
    tuner.prune_configs = lambda kwargs: list(tuner.configs)
    tuner._bench = lambda *args, config=None, **kwargs: (
        1.0 + tuner.configs.index(config)
    )
    assert autotune_cache.install_timing_capture() == 1
    key = {"M_BUCKET": 131072, "N": 5120, "K": 2560, "BLOCK_R": 32}
    tuner.prune_configs(key)
    for config in tuner.configs:
        tuner._bench(config=config)
    tuner.cache[_measured_key()] = tuner.configs[0]
    records = autotune_cache._with_timings(collect_autotune_configs())
    assert list(records[0]["timings_ms"].values()) == [1.0, 2.0]
    path = tmp_path / "b200.json"
    autotune_cache.write_autotune_cache(str(path), records)
    assert "timings_ms" in json.loads(path.read_text())["records"][0]


def test_corrupt_cache_is_ignored_and_rewritten(tmp_path, kernel_module):
    path = tmp_path / "b200.json"
    path.write_text('{"version": 1, "records": [')
    assert autotune_cache.read_autotune_cache(str(path)) is None
    assert autotune_cache.load_autotune_cache(str(path)) == 0
    path.write_text("[]")
    assert autotune_cache.read_autotune_cache(str(path)) is None
    tuner = kernel_module._scatter2scatter_lora
    tuner.cache[_measured_key()] = tuner.configs[0]
    autotune_cache.write_autotune_cache(str(path), collect_autotune_configs())
    assert autotune_cache.read_autotune_cache(str(path))["records"]
