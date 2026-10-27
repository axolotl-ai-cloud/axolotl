# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for match-scoped quack autotune profiles, sweep timing and quack seeding."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from axolotl.integrations.kernels import autotune_cache, autotune_profiles
from axolotl.integrations.kernels.sonicmoe_autotune import _is_sonic_module

REPO = Path(__file__).resolve().parents[3]
PACKAGE = "_sonic_moe_cuda_89d4054"


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 256
    cluster_m: int = 1
    cluster_n: int = 1
    is_dynamic_persistent: bool = False


class AutotuneConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __hash__(self):
        return hash(tuple(sorted(self.kwargs.items())))

    def __eq__(self, other):
        return isinstance(other, AutotuneConfig) and self.kwargs == other.kwargs

    def __str__(self):
        return ", ".join(f"{k}: {v}" for k, v in self.kwargs.items())


GRID = [
    GemmConfig(m, n, c, 1, d)
    for m, n in ((128, 256), (128, 224), (256, 256), (256, 512))
    for c in (1, 2)
    for d in (False, True)
]


class Autotuner:
    """The parts of quack's Autotuner.__call__ that axolotl relies on."""

    __module__ = f"{PACKAGE}.quack.autotuner"
    stringify = True

    def __init__(self, name, module_name, sweep_seconds):
        self.fn = SimpleNamespace(__name__=name, __module__=module_name)
        self.arg_names = ["A", "B", "out", "activation", "dynamic_scheduler"]
        self.keys = ["activation", "dynamic_scheduler"]
        self.configs = [AutotuneConfig(config=c) for c in GRID]
        self.cache = {}
        self.configs_timings = None
        self.bench_time = None
        self.nargs = None
        self.benched = []
        self.sweep_seconds = sweep_seconds

    def prune_configs(self, kwargs):
        return list(self.configs)

    def _bench(self, *args, config, **meta):
        self.benched.append(config)
        inner = config.kwargs["config"]
        return [1e5 / (inner.tile_m * inner.tile_n * inner.cluster_m)] * 3

    def _key(self, all_args):
        key = [all_args[k] for k in self.keys if k in all_args]
        if self.stringify:
            key = [str(v) for v in key]
        for value in all_args.values():
            if isinstance(value, torch.Tensor):
                if self.stringify:
                    key += [str(value.shape), str(value.dtype)]
                else:
                    key += [tuple(value.shape), value.dtype]
        return tuple(key)

    def __call__(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args, strict=False))
        key = self._key({**self.nargs, **kwargs})
        if key not in self.cache:
            pruned = self.prune_configs(kwargs)
            timings = {c: self._bench(*args, config=c, **kwargs) for c in pruned}
            self.bench_time = self.sweep_seconds(args[2].shape[0])
            self.cache[key] = min(timings, key=timings.get)
            self.configs_timings = timings
        self.nargs = None
        return self.cache[key].kwargs["config"]


class Autotuner065(Autotuner):
    __module__ = "quack.autotuner"
    stringify = False


def _sweep(rows):
    return 444.0 if rows >= 65536 else 20.0


def _call(tuner, rows, n=5120, k=5120):
    A = torch.empty((16384, k), device="meta", dtype=torch.bfloat16)
    B = torch.empty((32, k, n), device="meta", dtype=torch.bfloat16)
    out = torch.empty((rows, n // 2), device="meta", dtype=torch.bfloat16)
    return tuner(A, B, out, activation="swiglu", dynamic_scheduler=False)


@pytest.fixture(params=["0.6.1", "0.6.5"])
def quack(request, monkeypatch, isolated_kernel_modules, tmp_path):
    vendored = request.param == "0.6.1"
    package = f"{PACKAGE}.quack" if vendored else "quack"
    module = ModuleType(f"{package}.gemm_interface")
    cls = Autotuner if vendored else Autotuner065
    module.gemm_gated_tuned = cls("gemm_gated_tuned", module.__name__, _sweep)
    monkeypatch.setattr(
        autotune_cache, "_gpu_identity", lambda: {"gpu_name": "NVIDIA B200"}
    )
    monkeypatch.setattr(autotune_profiles, "_gpu_name", lambda: "NVIDIA B200")
    hidden = {
        k: None
        for k in sys.modules
        if _is_sonic_module(k) and not k.startswith("axolotl.")
    }
    with patch.dict(sys.modules, {**hidden, module.__name__: module}):
        yield SimpleNamespace(
            module=module, tuner=module.gemm_gated_tuned, vendored=vendored
        )


def _profile(configs, match=None):
    profile = {"kernel": "gemm_gated_tuned", "backend": "quack", "configs": configs}
    if match is not None:
        profile["match"] = match
    return [profile]


def test_match_narrows_only_the_matched_shapes(quack):
    tuner = quack.tuner
    profiles = _profile(
        [{"tile_m": 256, "tile_n": 512, "cluster_m": 2}],
        match={"out.dim0": [65536, 131072], "B.shape": [32, 5120, 5120]},
    )
    assert autotune_profiles.apply_autotune_profiles(profiles) == 1
    winner = _call(tuner, 131072)
    assert (winner.tile_m, winner.tile_n, winner.cluster_m) == (256, 512, 2)
    assert len(tuner.benched) == 2
    tuner.benched.clear()
    _call(tuner, 16384)
    assert len(tuner.benched) == len(GRID)
    tuner.benched.clear()
    _call(tuner, 65536, n=4096)
    assert len(tuner.benched) == len(GRID)


def test_single_matched_config_skips_the_benchmark(quack, tmp_path):
    tuner = quack.tuner
    autotune_cache.install_timing_capture()
    pin = {"tile_m": 128, "tile_n": 224, "cluster_m": 2, "is_dynamic_persistent": True}
    autotune_profiles.apply_autotune_profiles(
        _profile([pin], match={"activation": "swiglu", "out.dim0": 131072})
    )
    assert _call(tuner, 131072) == GemmConfig(128, 224, 2, 1, True)
    assert tuner.benched == []
    (record,) = autotune_cache.collect_sonicmoe_records()
    assert record["source"] == "profile"
    assert "timings_ms" not in record and "sweep_seconds" not in record


def test_unknown_config_is_built_from_the_autotuners_own_classes(quack):
    tuner = quack.tuner
    autotune_profiles.apply_autotune_profiles(
        _profile([{"tile_m": 192, "tile_n": 160}], match={"out.dim0": 131072})
    )
    config = _call(tuner, 131072)
    assert tuner.benched == []
    assert type(config) is GemmConfig and (config.tile_m, config.tile_n) == (192, 160)


def test_bad_match_key_is_rejected(quack):
    with pytest.raises(ValueError, match="arg.dimN"):
        autotune_profiles.apply_autotune_profiles(
            _profile([{"tile_m": 128}], match={"out.rows": 1})
        )


def test_profiles_reach_quack_imported_after_train_start(quack, tmp_path):
    module = quack.module
    path = str(tmp_path / "b200.json")
    callback = autotune_cache.AutotuneCacheCallback(
        path,
        profiles=_profile([{"tile_m": 256, "tile_n": 512}], match={"out.dim0": 131072}),
        sonicmoe=True,
    )
    with patch.dict(sys.modules, {module.__name__: None}):
        callback.on_train_begin(None, None, None)
    assert getattr(quack.tuner, "_axolotl_quack_profiles", None) is None
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    _call(quack.tuner, 131072)
    assert len(quack.tuner.benched) == 4


def test_sweep_time_and_call_args_feed_a_pasteable_profile(quack, tmp_path):
    tuner = quack.tuner
    autotune_cache.install_timing_capture()
    winner = _call(tuner, 131072)
    _call(tuner, 16384)
    path = tmp_path / "b200.json"
    callback = autotune_cache.AutotuneCacheCallback(str(path), sonicmoe=True)
    callback.on_save(None, None, None)
    records = json.loads(Path(autotune_cache.quack_cache_path(str(path))).read_text())[
        "records"
    ]
    by_rows = {r["call_args"]["out.shape"][0]: r for r in records}
    assert by_rows[131072]["sweep_seconds"] == 444.0
    assert by_rows[16384]["sweep_seconds"] == 20.0
    assert by_rows[131072]["call_args"]["B.shape"] == [32, 5120, 5120]
    assert by_rows[131072]["call_args"]["activation"] == "swiglu"

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "autotune_report.py"), str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    snippet = out[out.index("autotune_profiles:") :]
    (profile,) = yaml.safe_load(snippet)["autotune_profiles"]
    assert profile["match"]["out.shape"] == [131072, 2560]

    fresh = type(tuner)("gemm_gated_tuned", quack.module.__name__, _sweep)
    quack.module.gemm_gated_tuned = fresh
    autotune_profiles.apply_autotune_profiles([profile])
    assert _call(fresh, 131072) == winner
    assert fresh.benched == []
    _call(fresh, 16384)
    assert len(fresh.benched) == len(GRID)


def test_measured_winners_seed_only_stringified_keys(quack, tmp_path):
    tuner = quack.tuner
    autotune_cache.install_timing_capture()
    winner = _call(tuner, 131072)
    path = str(tmp_path / "b200.json")
    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)

    fresh = type(tuner)("gemm_gated_tuned", quack.module.__name__, _sweep)
    quack.module.gemm_gated_tuned = fresh
    applied = autotune_cache.load_quack_cache(autotune_cache.quack_cache_path(path))
    if quack.vendored:
        assert applied == 1
        assert _call(fresh, 131072) == winner
        assert fresh.benched == []
    else:
        assert applied == 0


def test_report_reads_quack_disk_cache(tmp_path):
    entry = tmp_path / "ABC" / "gemm_tuned.autotune.json"
    entry.parent.mkdir()
    entry.write_text(
        json.dumps(
            {
                "key": ["False", "torch.Size([131072, 2560])"],
                "configs_timings": [
                    ["config: GemmConfig(tile_m=128)", [0.46, 0.46, 0.47]],
                    ["config: GemmConfig(tile_m=256)", [float("inf")] * 3],
                ],
            }
        )
    )
    out = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "autotune_report.py"),
            "--quack-cache-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "gemm_tuned [quack disk cache]: 1 key(s)" in out
    assert "best 0.46 ms config: GemmConfig(tile_m=128)" in out
    assert "1/2 failed" in out
