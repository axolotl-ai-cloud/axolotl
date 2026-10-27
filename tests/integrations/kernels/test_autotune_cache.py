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


def _benchmark(tuner, key, times):
    """What Triton and quack do on a miss: store the winner, then the timing table."""
    timings = {
        config: [t, t, t] for config, t in zip(tuner.configs, times, strict=True)
    }
    tuner.cache[key] = min(timings, key=timings.get)
    tuner.configs_timings = timings


def _second_key():
    return (262144, 5120, 2560, 32, "torch.bfloat16", "torch.int32")


def test_timing_capture_records_every_benchmarked_config(kernel_module, tmp_path):
    tuner = kernel_module._scatter2scatter_lora
    assert autotune_cache.install_timing_capture() >= 2
    assert isinstance(tuner.cache, autotune_cache._TimingCache)
    assert autotune_cache.install_timing_capture() == 0
    _benchmark(tuner, _measured_key(), [2.0, 1.0])
    _benchmark(tuner, _second_key(), [3.0, float("inf")])

    records = autotune_cache.collect_triton_records()
    by_key = {r["key"]["M_BUCKET"]: r for r in records}
    first = by_key[131072]
    assert first["source"] == "measured"
    assert list(first["timings_ms"].values()) == [1.0, 2.0]
    assert first["configs_benchmarked"] == 2 and first["failed_configs"] == 0
    assert by_key[262144]["failed_configs"] == 1
    assert list(by_key[262144]["timings_ms"].values()) == [3.0]

    path = tmp_path / "b200.json"
    autotune_cache.write_autotune_cache(str(path), records)
    data = json.loads(path.read_text())
    assert all("timings_ms" in r for r in data["records"])
    assert "packages" in data


def test_profile_pinned_key_does_not_inherit_the_previous_table(kernel_module):
    tuner = kernel_module._scatter2scatter_lora
    autotune_cache.install_timing_capture()
    _benchmark(tuner, _measured_key(), [2.0, 1.0])
    tuner._axolotl_profile_hit = True
    tuner.cache[_second_key()] = tuner.configs[0]
    untimed = (524288, 5120, 2560, 32, "torch.bfloat16", "torch.int32")
    tuner.cache[untimed] = tuner.configs[0]

    by_key = {r["key"]["M_BUCKET"]: r for r in autotune_cache.collect_triton_records()}
    assert by_key[131072]["source"] == "measured"
    assert by_key[262144]["source"] == "profile"
    assert by_key[524288]["source"] == "untimed"
    assert "timings_ms" not in by_key[262144]
    assert "timings_ms" not in by_key[524288]


def test_profile_hit_is_flagged_by_the_profile_wrapper(kernel_module):
    from axolotl.integrations.kernels import autotune_profiles

    tuner = kernel_module._scatter2scatter_lora
    tuner.nargs = {}
    tuner.prune_configs = lambda kwargs: list(tuner.configs)
    autotune_profiles._install(
        tuner,
        [
            {
                "kernel": "*",
                "match": {"N": 5120},
                "configs": [
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 64,
                        "BLOCK_K": 64,
                        "num_warps": 4,
                        "num_stages": 4,
                    }
                ],
            }
        ],
    )
    tuner.prune_configs({"N": 4096})
    assert tuner._axolotl_profile_hit is False
    tuner.prune_configs({"N": 5120})
    assert tuner._axolotl_profile_hit is True


def test_seeded_rerun_keeps_the_measured_timings_on_file(kernel_module, tmp_path):
    tuner = kernel_module._scatter2scatter_lora
    path = tmp_path / "b200.json"
    autotune_cache.install_timing_capture()
    _benchmark(tuner, _measured_key(), [2.0, 1.0])
    autotune_cache.write_autotune_cache(
        str(path), autotune_cache.collect_triton_records()
    )

    tuner.cache.clear()
    tuner.configs_timings = None
    assert autotune_cache.load_autotune_cache(str(path)) == 1
    (rerun,) = autotune_cache.collect_triton_records()
    assert rerun["source"] == "seeded"
    autotune_cache.write_autotune_cache(str(path), [rerun])
    (on_file,) = json.loads(path.read_text())["records"]
    assert on_file["source"] == "measured"
    assert list(on_file["timings_ms"].values()) == [1.0, 2.0]


def test_untimed_record_never_replaces_a_timed_one(tmp_path, monkeypatch):
    monkeypatch.setattr(autotune_cache, "_gpu_identity", lambda: {})
    timed = {
        "kernel": "k",
        "module_fqn": "m",
        "key": {"N": 1},
        "config": {"BLOCK": 64},
        "source": "measured",
        "timings_ms": {"a": 1.0},
    }
    path = tmp_path / "c.json"
    autotune_cache.write_autotune_cache(str(path), [timed])
    untimed = {k: v for k, v in timed.items() if k != "timings_ms"}
    untimed["source"] = "untimed"
    autotune_cache.write_autotune_cache(str(path), [untimed])
    assert json.loads(path.read_text())["records"][0]["timings_ms"] == {"a": 1.0}
    changed = {**untimed, "config": {"BLOCK": 128}}
    autotune_cache.write_autotune_cache(str(path), [changed])
    assert json.loads(path.read_text())["records"][0]["config"] == {"BLOCK": 128}


def test_gather_prefers_the_rank_that_benchmarked(monkeypatch):
    seeded = {"kernel": "k", "module_fqn": "m", "key": {"N": 1}, "source": "seeded"}
    measured = {**seeded, "source": "measured", "timings_ms": {"a": 1.0}}

    def all_gather_object(out, _local):
        out[:] = [[seeded], [measured]]

    dist = autotune_cache.torch.distributed
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    (record,) = autotune_cache.gather_autotune_records(lambda: [seeded])
    assert record["source"] == "measured"


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
