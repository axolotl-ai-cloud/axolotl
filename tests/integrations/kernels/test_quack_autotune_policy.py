# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for quack autotune key bucketing and the no-sweep (heuristic) mode."""

import builtins
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from axolotl.integrations.kernels import (
    autotune_cache,
    autotune_profiles,
    quack_autotune_policy,
)
from axolotl.integrations.kernels.args import KernelsArgs
from axolotl.integrations.kernels.sonicmoe_autotune import _is_sonic_module

PACKAGE = "_sonic_moe_cuda_90d2076"
AUTOTUNER_MODULE = f"{PACKAGE}.quack.autotuner"
INTERFACE_MODULE = f"{PACKAGE}.quack.gemm_interface"


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    pingpong: bool = False
    is_dynamic_persistent: bool = True
    cluster_m: int = 2
    cluster_n: int = 1
    swap_ab: bool = False
    device_capacity: int = 10
    use_tma_gather: bool = False


class AutotuneConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def all_kwargs(self):
        return self.kwargs

    def __hash__(self):
        return hash(tuple(self.kwargs.items()))

    def __eq__(self, other):
        return tuple(self.kwargs.items()) == tuple(other.all_kwargs().items())

    def __str__(self):
        return ", ".join(f"{k}: {v}" for k, v in self.kwargs.items())


def _grid(capacity=10):
    tiles = [(128, 64), (128, 256), (256, 256)]
    if capacity == 10:
        tiles.append((256, 512))
    clusters = [(1, 1), (1, 2), (2, 1)]
    return [
        AutotuneConfig(
            config=GemmConfig(
                tile_m=m,
                tile_n=n,
                cluster_m=cm,
                cluster_n=cn,
                is_dynamic_persistent=clc,
                use_tma_gather=tma,
                device_capacity=capacity,
            )
        )
        for m, n in tiles
        for cm, cn in clusters
        for clc in (True, False)
        for tma in (True, False)
    ]


def prune_invalid(configs, named_args, **kwargs):
    """The parts of quack's ``prune_invalid_gemm_configs`` that matter for these fakes."""
    args = named_args | kwargs
    capacity = args.get("_capacity", 10)
    configs = [c for c in configs if c.kwargs["config"].device_capacity == capacity]
    gather = args.get("A_idx") is not None
    if gather:
        configs = [c for c in configs if c.kwargs["config"].cluster_n == 1]
    else:
        configs = [c for c in configs if not c.kwargs["config"].use_tma_gather]
    reject = args.get("_reject")
    if reject is not None:
        configs = [c for c in configs if not reject(c.kwargs["config"], args)]
    return configs


SIGNATURES = {
    "gemm_tuned": (
        ["A", "B", "out", "C", "bias", "alpha", "beta", "cu_seqlens_m",
         "cu_seqlens_k", "A_idx", "batch_idx_permute", "add_to_output",
         "dynamic_scheduler", "config", "SFA", "SFB", "_capacity", "_reject"],
        ["dynamic_scheduler"],
    ),
    "gemm_gated_tuned": (
        ["A", "B", "preact_out", "postact_out", "C", "bias", "activation",
         "cu_seqlens_m", "A_idx", "dynamic_scheduler", "config", "_capacity",
         "_reject"],
        ["activation", "dynamic_scheduler"],
    ),
    "gemm_dgated_tuned": (
        ["A", "B", "PreAct", "dx_out", "postact_out", "colvec_scale", "activation",
         "colvec_reduce", "cu_seqlens_m", "A_idx", "dynamic_scheduler", "config",
         "_capacity", "_reject"],
        ["activation", "colvec_reduce", "dynamic_scheduler"],
    ),
}  # fmt: skip


class Autotuner:
    """quack 0.6.1's ``Autotuner.__call__``/``check_disk_cache`` flow (autotuner.py 255-453)."""

    __module__ = AUTOTUNER_MODULE

    def __init__(self, name, capacity=10, cache_results=False, cache_dir=None):
        self.fn = SimpleNamespace(__name__=name, __module__=INTERFACE_MODULE)
        self.arg_names, self.keys = SIGNATURES[name]
        self.configs = _grid(capacity)
        self.cache = {}
        self.cache_results = cache_results
        self.cache_dir = cache_dir
        self.early_config_prune = prune_invalid
        self.perf_model = None
        self.nargs = None
        self.configs_timings = None
        self.bench_time = None
        self.benched = []
        self.calls = []

    def prune_configs(self, kwargs):
        return self.early_config_prune(self.configs, self.nargs, **kwargs)

    def _bench(self, *args, config, **meta):
        self.benched.append(config)
        inner = config.kwargs["config"]
        return [1000.0 / (inner.tile_m * inner.tile_n * inner.cluster_m)] * 3

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        config_str_list = [str(c) for c in configs]
        digest = hashlib.sha256(
            "-".join(["0.6.1", str(tuning_key)] + config_str_list).encode()
        ).hexdigest()
        path = os.path.join(self.cache_dir, digest, f"{self.fn.__name__}.autotune.json")
        if os.path.exists(path):
            str2config = dict(zip(config_str_list, configs, strict=True))
            with open(path, encoding="utf-8") as handle:
                timings = json.load(handle)["configs_timings"]
            timings = {str2config[c]: t for c, t in timings}
            self.cache[tuning_key] = builtins.min(timings, key=timings.get)
            self.configs_timings = timings
            self.bench_time = 0
            return
        bench_fn()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "key": tuning_key,
                    "configs_timings": [
                        (str(c), t) for c, t in self.configs_timings.items()
                    ],
                },
                handle,
            )

    def __call__(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args, strict=False))
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        key = [str(_args[k]) for k in self.keys if k in _args]
        for arg in _args.values():
            if isinstance(arg, torch.Tensor):
                key.append(str(arg.shape))
                key.append(str([s if s in {0, 1} else 2 for s in arg.stride()]))
                key.append(str(arg.dtype))
        key = tuple(key)
        self.calls.append(key)
        if key not in self.cache:
            pruned = self.prune_configs(kwargs)

            def benchmark():
                sys.modules[AUTOTUNER_MODULE]._gpu_warmup()
                timings = {c: self._bench(*args, config=c, **kwargs) for c in pruned}
                self.bench_time = 12.5
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings

            if self.cache_results:
                self.check_disk_cache(key, pruned, benchmark)
            else:
                benchmark()
        config = self.cache[key]
        self.nargs = None
        return config.kwargs["config"]


@pytest.fixture
def quack(monkeypatch, isolated_kernel_modules):
    autotuner_module = ModuleType(AUTOTUNER_MODULE)
    autotuner_module.warmups = 0

    def _gpu_warmup(duration_ms=200):
        autotuner_module.warmups += 1

    autotuner_module._gpu_warmup = _gpu_warmup
    interface = ModuleType(INTERFACE_MODULE)
    monkeypatch.setattr(
        autotune_cache, "_gpu_identity", lambda: {"gpu_name": "NVIDIA B200"}
    )
    monkeypatch.setattr(autotune_profiles, "_gpu_name", lambda: "NVIDIA B200")
    hidden = {
        k: None
        for k in sys.modules
        if _is_sonic_module(k) and not k.startswith("axolotl.")
    }
    modules = {AUTOTUNER_MODULE: autotuner_module, INTERFACE_MODULE: interface}

    def add(name, **kw):
        tuner = Autotuner(name, **kw)
        setattr(interface, name, tuner)
        return tuner

    with patch.dict(sys.modules, {**hidden, **modules}):
        yield SimpleNamespace(add=add, module=interface, autotuner=autotuner_module)


def _t(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, device="meta", dtype=dtype)


def _cu(experts=32):
    return _t(experts + 1, dtype=torch.int32)


def gated(tuner, rows, tokens=None, n=5120, k=5120, gather=False, **extra):
    A = _t(tokens or rows, k)
    B = _t(32, k, n)
    idx = _t(rows, dtype=torch.int32) if gather else None
    return tuner(
        A,
        B,
        _t(rows, n),
        _t(rows, n // 2),
        activation="swiglu",
        cu_seqlens_m=_cu(),
        A_idx=idx,
        dynamic_scheduler=False,
        **extra,
    )


def dgated(tuner, rows, n=2560, k=5120, gather=False, **extra):
    idx = _t(rows, dtype=torch.int32) if gather else None
    return tuner(
        _t(rows, k),
        _t(32, k, n),
        _t(rows, 2 * n),
        _t(rows, 2 * n),
        _t(rows, n),
        _t(rows, dtype=torch.float32),
        activation="swiglu",
        colvec_reduce=False,
        cu_seqlens_m=_cu(),
        A_idx=idx,
        dynamic_scheduler=True,
        **extra,
    )


def down(tuner, rows, n=5120, k=2560, **extra):
    """Varlen-M ``gemm_tuned``: the down-projection forward."""
    return tuner(_t(rows, k), _t(32, k, n), _t(rows, n), cu_seqlens_m=_cu(), **extra)


def dense(tuner, rows, n=5120, k=2560, **extra):
    return tuner(_t(rows, k), _t(k, n), _t(rows, n), **extra)


def weight_grad(tuner, total_k, m=5120, n=128, **extra):
    """Varlen-K ``gemm_tuned``: a rank-r LoRA gradient over the routed rows."""
    return tuner(
        _t(m, total_k), _t(total_k, n), _t(32, m, n), cu_seqlens_k=_cu(), **extra
    )


def test_one_sweep_per_pow2_bucket(quack):
    tuner = quack.add("gemm_gated_tuned")
    assert quack_autotune_policy.install_quack_key_policy("pow2") == 1
    first = gated(tuner, 600_000)
    swept = len(tuner.benched)
    assert swept > 0
    assert gated(tuner, 720_000) == first
    assert len(tuner.benched) == swept
    (bucket,) = tuner.cache
    assert "torch.Size([1048576, 5120])" in bucket
    assert bucket[-1] == "axolotl_bucket=pow2"

    gated(tuner, 1_100_000)
    assert len(tuner.benched) == 2 * swept
    assert len(tuner.cache) == 2


def test_shape_buckets_drop_token_dims(quack):
    tuner = quack.add("gemm_gated_tuned")
    assert quack_autotune_policy.install_quack_key_policy("shape") == 1
    first = gated(tuner, 600_000, tokens=70_000, gather=True)
    swept = len(tuner.benched)
    assert swept > 0
    assert gated(tuner, 1_100_000, tokens=150_000, gather=True) == first
    assert gated(tuner, 20_000, tokens=4_000, gather=True) == first
    assert len(tuner.benched) == swept
    (bucket,) = tuner.cache
    assert "torch.Size([-1, 5120])" in bucket and "torch.Size([-1])" in bucket
    assert bucket[-1] == "axolotl_bucket=shape"

    gated(tuner, 600_000, n=3000, tokens=70_000, gather=True)
    assert len(tuner.benched) == 2 * swept


def test_shape_buckets_drop_the_varlen_k_dim(quack):
    tuner = quack.add("gemm_tuned")
    quack_autotune_policy.install_quack_key_policy("shape")
    weight_grad(tuner, 600_000)
    swept = len(tuner.benched)
    weight_grad(tuner, 90_000)
    assert len(tuner.benched) == swept
    (bucket,) = tuner.cache
    assert (
        "torch.Size([5120, -1])" in bucket and "torch.Size([32, 5120, 128])" in bucket
    )


def test_weight_dims_are_never_bucketed(quack):
    tuner = quack.add("gemm_gated_tuned")
    quack_autotune_policy.install_quack_key_policy("pow2")
    gated(tuner, 600_000)
    swept = len(tuner.benched)
    gated(tuner, 600_000, n=3000)
    assert len(tuner.benched) == 2 * swept
    keys = list(tuner.cache)
    assert any("torch.Size([32, 5120, 3000])" in k for k in keys)


def test_gather_source_rows_are_bucketed(quack):
    tuner = quack.add("gemm_gated_tuned")
    quack_autotune_policy.install_quack_key_policy("pow2")
    gated(tuner, 600_000, tokens=70_000, gather=True)
    swept = len(tuner.benched)
    gated(tuner, 610_000, tokens=80_000, gather=True)
    assert len(tuner.benched) == swept
    (bucket,) = tuner.cache
    assert "torch.Size([131072, 5120])" in bucket
    assert "torch.Size([1048576])" in bucket


def test_varlen_k_buckets_the_reduction_dim(quack):
    tuner = quack.add("gemm_tuned")
    quack_autotune_policy.install_quack_key_policy("pow2")
    weight_grad(tuner, 600_000)
    swept = len(tuner.benched)
    weight_grad(tuner, 700_000)
    assert len(tuner.benched) == swept
    (bucket,) = tuner.cache
    assert "torch.Size([5120, 1048576])" in bucket
    assert "torch.Size([1048576, 128])" in bucket
    assert "torch.Size([32, 5120, 128])" in bucket


def test_non_grouped_calls_keep_exact_keys(quack):
    tuner = quack.add("gemm_tuned")
    quack_autotune_policy.install_quack_key_policy("pow2")
    dense(tuner, 600_000)
    swept = len(tuner.benched)
    dense(tuner, 700_000)
    assert len(tuner.benched) == 2 * swept
    assert all(k[-1] != "axolotl_bucket=pow2" for k in tuner.cache)


def test_none_disables_bucketing(quack):
    tuner = quack.add("gemm_tuned")
    assert quack_autotune_policy.install_quack_key_policy("none") == 0
    down(tuner, 600_000)
    swept = len(tuner.benched)
    down(tuner, 700_000)
    assert len(tuner.benched) == 2 * swept
    assert not hasattr(tuner, "_axolotl_key_policy")


def test_reused_winner_rejected_by_pruning_tunes_the_exact_key(quack):
    tuner = quack.add("gemm_tuned")
    autotune_cache.install_timing_capture()
    quack_autotune_policy.install_quack_key_policy("pow2")

    def reject(config, args):
        return args["out"].shape[0] == 700_000 and config.tile_m == 256

    first = down(tuner, 600_000, _reject=reject)
    assert first.tile_m == 256
    swept = len(tuner.benched)
    exact = down(tuner, 700_000, _reject=reject)
    assert exact.tile_m == 128
    assert len(tuner.benched) > swept
    assert len(tuner.cache) == 2
    again = len(tuner.benched)
    assert down(tuner, 700_000, _reject=reject) == exact
    assert down(tuner, 650_000, _reject=reject) == first
    assert len(tuner.benched) == again

    records = autotune_cache.collect_sonicmoe_records()
    by_rows = {r["call_args"]["out.shape"][0]: r for r in records}
    assert by_rows[700_000]["bucket_fallback"] is True
    assert by_rows[700_000]["key"]["signature"][-1] != "axolotl_bucket=pow2"
    assert by_rows[600_000]["bucket_reuses"] == 1
    assert by_rows[600_000]["source"] == "measured"


def test_disk_cache_uses_the_bucketed_key(quack, tmp_path):
    cache_dir = str(tmp_path / "quack")
    tuner = quack.add("gemm_tuned", cache_results=True, cache_dir=cache_dir)
    quack_autotune_policy.install_quack_key_policy("pow2")
    winner = down(tuner, 600_000)
    (entry,) = list((tmp_path / "quack").glob("*/gemm_tuned.autotune.json"))
    stored = json.loads(entry.read_text())["key"]
    assert stored[-1] == "axolotl_bucket=pow2"
    assert "torch.Size([1048576, 5120])" in stored

    fresh = quack.add("gemm_tuned", cache_results=True, cache_dir=cache_dir)
    quack_autotune_policy.install_quack_key_policy("pow2")
    assert down(fresh, 900_000) == winner
    assert fresh.benched == []
    assert len(list((tmp_path / "quack").glob("*/*.json"))) == 1


def test_records_serialize_with_bucketed_key_and_actual_shapes(quack, tmp_path):
    tuner = quack.add("gemm_gated_tuned")
    autotune_cache.install_timing_capture()
    quack_autotune_policy.install_quack_key_policy("pow2")
    gated(tuner, 600_000)
    gated(tuner, 700_000)
    (record,) = autotune_cache.collect_sonicmoe_records()
    json.dumps(record)
    assert record["source"] == "measured"
    assert record["call_args"]["postact_out.shape"] == [600_000, 2560]
    assert "torch.Size([1048576, 2560])" in record["key"]["signature"]
    assert record["bucket_reuses"] == 1
    assert record["sweep_seconds"] == 12.5

    path = str(tmp_path / "b200.json")
    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)
    fresh = quack.add("gemm_gated_tuned")
    applied = autotune_cache.load_quack_cache(autotune_cache.quack_cache_path(path))
    assert applied == 1
    quack_autotune_policy.install_quack_key_policy("pow2")
    gated(fresh, 800_000)
    assert fresh.benched == []

    from axolotl.integrations.kernels.sonicmoe_autotune import (
        SonicMoEAutotuneReportCallback,
    )

    callback = SonicMoEAutotuneReportCallback.__new__(SonicMoEAutotuneReportCallback)
    json.dumps(callback._collect_configs(), default=str)


def test_seeded_bucket_winner_outside_the_profile_retunes_the_bucket(quack, tmp_path):
    tuner = quack.add("gemm_tuned")
    autotune_cache.install_timing_capture()
    quack_autotune_policy.install_quack_key_policy("shape")
    assert weight_grad(tuner, 300_000, n=64).is_dynamic_persistent
    path = str(tmp_path / "b200.json")
    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)
    fresh = quack.add("gemm_tuned")
    autotune_cache.install_timing_capture()
    assert autotune_cache.load_quack_cache(autotune_cache.quack_cache_path(path)) == 1
    autotune_profiles.apply_autotune_profiles(
        [
            {
                "kernel": "gemm_tuned",
                "backend": "quack",
                "match": {"out.shape": [32, 5120, 64]},
                "configs": [
                    {"tile_m": 128, "tile_n": 64, "cluster_m": 2, "cluster_n": 1,
                     "is_dynamic_persistent": False},
                    {"tile_m": 256, "tile_n": 256, "cluster_m": 2, "cluster_n": 1,
                     "is_dynamic_persistent": False},
                ],
            }
        ]
    )  # fmt: skip
    quack_autotune_policy.install_quack_key_policy("shape")
    first = weight_grad(fresh, 400_000, n=64)
    assert not first.is_dynamic_persistent and len(fresh.benched) == 2
    for k in (500_000, 600_000, 700_000):
        assert weight_grad(fresh, k, n=64) == first
    assert len(fresh.benched) == 2 and len(fresh.cache) == 1
    assert not fresh.cache.exact_fallbacks


def _read_records(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["records"]


def _strip_fingerprints(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    for record in data["records"]:
        record.pop("swept_configs_sha256", None)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def test_seeded_bucket_winner_without_a_fingerprint_heals_through_the_bucket_check(
    quack, tmp_path
):
    tuner = quack.add("gemm_tuned")
    autotune_cache.install_timing_capture()
    quack_autotune_policy.install_quack_key_policy("shape")
    weight_grad(tuner, 300_000, n=64)
    path = str(tmp_path / "b200.json")
    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)
    quack_path = autotune_cache.quack_cache_path(path)
    _strip_fingerprints(quack_path)

    fresh = quack.add("gemm_tuned")
    autotune_cache.install_timing_capture()
    assert autotune_cache.load_quack_cache(quack_path) == 1
    assert not getattr(fresh.cache, "unverified", None)
    autotune_profiles.apply_autotune_profiles(
        [{"kernel": "gemm_tuned", "backend": "quack",
          "allow": {"is_dynamic_persistent": [False]}}]
    )  # fmt: skip
    quack_autotune_policy.install_quack_key_policy("shape")
    first = weight_grad(fresh, 400_000, n=64)
    assert not first.is_dynamic_persistent
    swept = len(fresh.benched)
    assert swept > 0
    weight_grad(fresh, 500_000, n=64)
    assert len(fresh.benched) == swept and not fresh.cache.exact_fallbacks

    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)
    (record,) = _read_records(quack_path)
    assert record["source"] == "measured"
    assert record["config"]["config"]["is_dynamic_persistent"] is False
    assert record["swept_configs_sha256"]


def test_seeded_winner_swept_over_other_candidates_is_retuned(quack, tmp_path):
    """The seeded winner is still a candidate, but this run's profiles offer others."""
    tuner = quack.add("gemm_gated_tuned")
    autotune_cache.install_timing_capture()
    autotune_profiles.apply_autotune_profiles(
        [{"kernel": "gemm_gated_tuned", "backend": "quack", "allow": {"tile_m": [256]}}]
    )
    quack_autotune_policy.install_quack_key_policy("shape")
    winner = gated(tuner, 600_000)
    narrow = len(tuner.benched)
    path = str(tmp_path / "b200.json")
    quack_path = autotune_cache.quack_cache_path(path)
    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)
    (old,) = _read_records(quack_path)
    assert old["source"] == "measured" and old["configs_benchmarked"] == narrow

    fresh = quack.add("gemm_gated_tuned")
    autotune_cache.install_timing_capture()
    assert autotune_cache.load_quack_cache(quack_path) == 1
    quack_autotune_policy.install_quack_key_policy("shape")
    assert gated(fresh, 700_000) == winner
    full = len(fresh.benched)
    assert full > narrow
    gated(fresh, 800_000)
    assert len(fresh.benched) == full and len(fresh.cache) == 1

    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)
    (new,) = _read_records(quack_path)
    assert new["configs_benchmarked"] == full
    assert new["swept_configs_sha256"] != old["swept_configs_sha256"]

    again = quack.add("gemm_gated_tuned")
    autotune_cache.install_timing_capture()
    assert autotune_cache.load_quack_cache(quack_path) == 1
    quack_autotune_policy.install_quack_key_policy("shape")
    assert gated(again, 900_000) == winner
    assert again.benched == [] and not again.cache.unverified


def test_seeded_exact_key_swept_over_other_candidates_is_retuned(quack, tmp_path):
    tuner = quack.add("gemm_tuned")
    autotune_cache.install_timing_capture()
    autotune_profiles.apply_autotune_profiles(
        [{"kernel": "gemm_tuned", "backend": "quack", "allow": {"tile_m": [128]}}]
    )
    assert dense(tuner, 4096).tile_m == 128
    path = str(tmp_path / "b200.json")
    quack_path = autotune_cache.quack_cache_path(path)
    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)

    fresh = quack.add("gemm_tuned")
    autotune_cache.install_timing_capture()
    assert autotune_cache.load_quack_cache(quack_path) == 1
    assert quack_autotune_policy.install_quack_key_policy("none") == 1
    retuned = dense(fresh, 4096)
    assert retuned.tile_m == 256 and fresh.benched
    swept = len(fresh.benched)
    assert dense(fresh, 4096) == retuned and len(fresh.benched) == swept

    autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(None, None, None)
    (record,) = _read_records(quack_path)
    assert record["config"]["config"]["tile_m"] == 256


def test_merge_replaces_a_record_swept_over_other_candidates():
    old = {"source": "measured", "config": {"a": 1}, "timings_ms": {"x": 1.0},
           "swept_configs_sha256": "old"}  # fmt: skip
    pinned = {"source": "profile", "config": {"a": 1}, "swept_configs_sha256": "new"}
    assert autotune_cache._merge_record(old, pinned) is pinned
    same = {"source": "profile", "config": {"a": 1}, "swept_configs_sha256": "old"}
    assert autotune_cache._merge_record(old, same) is old
    seeded = {"source": "seeded", "config": {"a": 2}}
    assert autotune_cache._merge_record(old, seeded) is old


def test_gpu_warmup_is_skipped(quack):
    tuner = quack.add("gemm_gated_tuned")
    quack_autotune_policy.install_quack_key_policy("pow2")
    gated(tuner, 600_000)
    assert tuner.benched and quack.autotuner.warmups == 0


def test_gpu_warmup_is_skipped_with_profiles_only(quack):
    tuner = quack.add("gemm_gated_tuned")
    autotune_profiles.apply_autotune_profiles(
        [{"kernel": "gemm_gated_tuned", "backend": "quack", "allow": {"tile_m": [256]}}]
    )
    gated(tuner, 600_000)
    assert tuner.benched and quack.autotuner.warmups == 0


def test_gpu_warmup_patch_tolerates_a_quack_without_it(quack):
    tuner = quack.add("gemm_gated_tuned")
    del quack.autotuner._gpu_warmup
    assert autotune_profiles.noop_quack_gpu_warmup(tuner) is False
    assert not hasattr(quack.autotuner, "_gpu_warmup")


def _fields(config):
    return (
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.is_dynamic_persistent,
        config.use_tma_gather,
    )


@pytest.mark.parametrize("gather", [False, True])
def test_heuristic_gated_and_dgated(quack, gather):
    up = quack.add("gemm_gated_tuned")
    back = quack.add("gemm_dgated_tuned")
    autotune_cache.install_timing_capture()
    quack_autotune_policy.install_quack_key_policy("pow2", tune=False)
    assert _fields(gated(up, 600_000, gather=gather)) == (256, 512, 2, 1, True, gather)
    assert _fields(dgated(back, 600_000, gather=gather)) == (
        256, 256, 2, 1, True, gather,
    )  # fmt: skip
    assert up.benched == [] and back.benched == []
    assert quack.autotuner.warmups == 0
    sources = {
        r["kernel"]: r["source"] for r in autotune_cache.collect_sonicmoe_records()
    }
    assert sources == {
        "gemm_gated_tuned": "heuristic",
        "gemm_dgated_tuned": "heuristic",
    }


def test_heuristic_gemm_tuned_rules(quack):
    tuner = quack.add("gemm_tuned")
    quack_autotune_policy.install_quack_key_policy("pow2", tune=False)
    assert _fields(down(tuner, 600_000)) == (256, 256, 2, 1, True, False)
    assert _fields(down(tuner, 600_000, n=128, k=5120)) == (128, 64, 1, 1, True, False)
    assert _fields(weight_grad(tuner, 600_000, n=64)) == (128, 64, 2, 1, True, False)
    assert _fields(down(tuner, 600_000, n=5120, k=64)) == (128, 256, 1, 2, False, False)
    assert tuner.benched == []


def test_heuristic_falls_back_to_the_first_valid_config(quack, caplog):
    tuner = quack.add("gemm_gated_tuned", capacity=9)
    quack_autotune_policy.install_quack_key_policy("pow2", tune=False)
    with caplog.at_level("INFO"):
        config = gated(tuner, 600_000, _capacity=9)
        gated(tuner, 2_000_000, _capacity=9)
    first = prune_invalid(tuner.configs, {"A_idx": None, "_capacity": 9})[0]
    assert config == first.kwargs["config"]
    assert tuner.benched == []
    assert sum("first valid one" in r.getMessage() for r in caplog.records) == 1


def test_heuristic_skips_a_rule_config_pruned_for_the_call(quack):
    tuner = quack.add("gemm_gated_tuned")
    quack_autotune_policy.install_quack_key_policy("pow2", tune=False)

    def reject(config, args):
        return config.tile_n == 512

    config = gated(tuner, 600_000, _reject=reject)
    assert config.tile_n != 512
    assert tuner.benched == []


def test_profiles_override_the_heuristic(quack):
    tuner = quack.add("gemm_gated_tuned")
    autotune_cache.install_timing_capture()
    autotune_profiles.apply_autotune_profiles(
        [
            {
                "kernel": "gemm_gated_tuned",
                "backend": "quack",
                "match": {"postact_out.dim0": {"ge": 262144}},
                "configs": [{"tile_m": 128, "tile_n": 256, "cluster_m": 1,
                             "cluster_n": 1, "is_dynamic_persistent": False}],
            }
        ]
    )  # fmt: skip
    quack_autotune_policy.install_quack_key_policy("pow2", tune=False)
    assert _fields(gated(tuner, 600_000)) == (128, 256, 1, 1, False, False)
    assert _fields(gated(tuner, 100_000)) == (256, 512, 2, 1, True, False)
    assert tuner.benched == []
    sources = {
        r["call_args"]["postact_out.shape"][0]: r["source"]
        for r in autotune_cache.collect_sonicmoe_records()
    }
    assert sources == {600_000: "profile", 100_000: "heuristic"}


def test_callback_installs_the_policy_for_lazily_imported_quack(quack):
    callback = autotune_cache.AutotuneCacheCallback(
        None, sonicmoe=True, quack_policy=("pow2", False)
    )
    callback.on_train_begin(None, None, None)
    tuner = quack.add("gemm_gated_tuned")
    callback.on_step_begin(None, SimpleNamespace(global_step=0), None)
    gated(tuner, 600_000)
    assert tuner.benched == []
    callback.on_step_end(None, SimpleNamespace(global_step=0), None)


def test_kernels_args_validation():
    ok = KernelsArgs.model_validate(
        {
            "use_sonicmoe": True,
            "quack_autotune_key_buckets": "none",
            "quack_autotune": False,
        }
    )
    assert ok.quack_autotune_key_buckets == "none" and ok.quack_autotune is False
    shape = KernelsArgs.model_validate(
        {"use_sonicmoe": True, "quack_autotune_key_buckets": "shape"}
    )
    assert shape.quack_autotune_key_buckets == "shape"
    with pytest.raises(ValueError, match="pow2"):
        KernelsArgs.model_validate(
            {"use_sonicmoe": True, "quack_autotune_key_buckets": "pow3"}
        )
    with pytest.raises(ValueError, match="only apply with use_sonicmoe"):
        KernelsArgs.model_validate({"use_scattermoe": True, "quack_autotune": False})
