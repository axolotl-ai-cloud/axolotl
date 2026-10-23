# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for quack ``allow`` filters on the SM100 GemmConfig grid."""

import json
import re
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
README = REPO / "src" / "axolotl" / "integrations" / "kernels" / "README.md"
PACKAGE = "_sonic_moe_cuda_89d4054"


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int
    swap_ab: bool
    is_dynamic_persistent: bool
    use_tma_gather: bool


class AutotuneConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __hash__(self):
        return hash(tuple(sorted(self.kwargs.items())))

    def __eq__(self, other):
        return isinstance(other, AutotuneConfig) and self.kwargs == other.kwargs


def _sm100_tiles():
    """quack's ``_get_sm100_configs`` (tile_m, tile_n, cluster) combos: 37 of them."""
    tile_ns = (64, 128, 160, 192, 224, 256)
    combos = [(128, n, c) for n in tile_ns for c in ((1, 1), (1, 2), (2, 1), (2, 2))]
    combos += [(256, n, c) for n in tile_ns for c in ((2, 1), (2, 2))]
    return combos + [(256, 512, (2, 1))]


GRID = [
    GemmConfig(m, n, cm, cn, swap, dyn, gather)
    for m, n, (cm, cn) in _sm100_tiles()
    for swap in (False, True)
    for dyn in (False, True)
    for gather in (False, True)
]


def _valid_gated(configs, nargs, **kwargs):
    return [
        c
        for c in configs
        if c.kwargs["config"].cluster_n == 1 and not c.kwargs["config"].swap_ab
    ]


def _valid_varlen(configs, nargs, **kwargs):
    return [
        c
        for c in configs
        if not c.kwargs["config"].swap_ab and not c.kwargs["config"].use_tma_gather
    ]


class Autotuner:
    __module__ = f"{PACKAGE}.quack.autotuner"
    stringify = True
    sweep = 400.0

    def __init__(self, name, module_name, early_config_prune):
        self.fn = SimpleNamespace(__name__=name, __module__=module_name)
        self.arg_names = ["A", "B", "out", "dynamic_scheduler"]
        self.keys = ["dynamic_scheduler"]
        self.configs = [AutotuneConfig(config=c) for c in GRID]
        self.early_config_prune = early_config_prune
        self.cache = {}
        self.configs_timings = None
        self.bench_time = None
        self.nargs = None
        self.benched = []

    def prune_configs(self, kwargs):
        return self.early_config_prune(self.configs, self.nargs, **kwargs)

    def _bench(self, *args, config, **meta):
        self.benched.append(config)
        c = config.kwargs["config"]
        return [1e5 / (c.tile_m * c.tile_n)] * 3

    def __call__(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args, strict=False))
        all_args = {**self.nargs, **kwargs}
        key = [all_args[k] for k in self.keys if k in all_args]
        for value in all_args.values():
            if isinstance(value, torch.Tensor):
                key.append(str(value.shape) if self.stringify else tuple(value.shape))
        key = tuple(str(v) for v in key) if self.stringify else tuple(key)
        if key not in self.cache:
            pruned = self.prune_configs(kwargs)
            timings = {c: self._bench(*args, config=c, **kwargs) for c in pruned}
            self.bench_time = self.sweep
            self.cache[key] = min(timings, key=timings.get)
            self.configs_timings = timings
        self.nargs = None
        return self.cache[key]


class Autotuner065(Autotuner):
    __module__ = "quack.autotuner"
    stringify = False


def _call(tuner, rows, n=5120):
    A = torch.empty((rows, 2560), device="meta")
    B = torch.empty((32, 2560, n), device="meta")
    out = torch.empty((rows, n), device="meta")
    return tuner(A, B, out, dynamic_scheduler=False)


@pytest.fixture(params=["0.6.1", "0.6.5"])
def kernels(request, monkeypatch, isolated_kernel_modules):
    vendored = request.param == "0.6.1"
    package = f"{PACKAGE}.quack" if vendored else "quack"
    module = ModuleType(f"{package}.gemm_interface")
    cls = Autotuner if vendored else Autotuner065
    module.gemm_gated_tuned = cls("gemm_gated_tuned", module.__name__, _valid_gated)
    module.gemm_tuned = cls("gemm_tuned", module.__name__, _valid_varlen)
    # gather_A backward: same validity rules as the gated forward over the full 296 grid
    module.gemm_dgated_tuned = cls("gemm_dgated_tuned", module.__name__, _valid_gated)
    module.gemm_dgated_tuned.sweep = 511.29
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
        yield module


def _readme_allow_profiles():
    text = README.read_text()
    section = text[text.index("### Allow filters (quack)") :]
    (block,) = [
        b for b in re.findall(r"```yaml\n(.*?)```", section, re.S) if "allow:" in b
    ]
    return yaml.safe_load(block)["autotune_profiles"]


MEASURED_WINNERS = {
    "gemm_dgated_tuned": [(256, 256, 2, 1)],
    "gemm_gated_tuned": [(128, 256, 2, 1), (256, 512, 2, 1)],
    "gemm_tuned": [
        (128, 256, 2, 1),
        (128, 224, 2, 1),
        (256, 256, 2, 1),
        (128, 256, 1, 1),
        (128, 160, 1, 2),
        (256, 192, 2, 2),
        (128, 192, 1, 1),
    ],
}


def test_validity_pruning_matches_the_cluster_counts(kernels):
    assert len(GRID) == 37 * 8
    assert len(_valid_gated(kernels.gemm_gated_tuned.configs, {})) == 76
    assert len(_valid_varlen(kernels.gemm_tuned.configs, {})) == 74


def test_readme_beam_filter_keeps_every_measured_winner(kernels):
    autotune_cache.install_timing_capture()
    autotune_profiles.apply_autotune_profiles(_readme_allow_profiles())
    expected = {"gemm_gated_tuned": 24, "gemm_tuned": 42, "gemm_dgated_tuned": 52}
    for name, count in expected.items():
        tuner = getattr(kernels, name)
        _call(tuner, 131072)
        assert len(tuner.benched) == count
        kept = {
            (c.tile_m, c.tile_n, c.cluster_m, c.cluster_n)
            for c in (b.kwargs["config"] for b in tuner.benched)
        }
        assert set(MEASURED_WINNERS[name]) <= kept
    records = {r["kernel"]: r for r in autotune_cache.collect_sonicmoe_records()}
    assert records["gemm_gated_tuned"]["candidate_config_count"] == 24
    assert records["gemm_tuned"]["candidate_config_count"] == 42
    assert records["gemm_tuned"]["source"] == "measured"
    assert records["gemm_tuned"]["configs_benchmarked"] == 42


def test_unfiltered_kernels_record_the_validity_pruned_count(kernels):
    autotune_cache.install_timing_capture()
    _call(kernels.gemm_tuned, 16384)
    (record,) = [
        r
        for r in autotune_cache.collect_sonicmoe_records()
        if r["kernel"] == "gemm_tuned"
    ]
    assert record["candidate_config_count"] == 74


def test_allow_with_match_filters_only_matched_calls(kernels):
    tuner = kernels.gemm_tuned
    autotune_profiles.apply_autotune_profiles(
        [
            {
                "kernel": "gemm_tuned",
                "backend": "quack",
                "match": {"A.dim0": 131072},
                "allow": {"tile_m": 256, "cluster": [[2, 1]]},
            }
        ]
    )
    _call(tuner, 131072)
    assert len(tuner.benched) == 7 * 2
    tuner.benched.clear()
    _call(tuner, 16384)
    assert len(tuner.benched) == 74


def test_where_clause_only_constrains_configs_it_selects(kernels):
    tuner = kernels.gemm_tuned
    autotune_profiles.apply_autotune_profiles(
        [
            {
                "kernel": "gemm_tuned",
                "backend": "quack",
                "allow": {
                    "is_dynamic_persistent": True,
                    "where": [{"if": {"tile_m": 128}, "then": {"cluster": [1, 1]}}],
                },
            }
        ]
    )
    _call(tuner, 65536)
    configs = [b.kwargs["config"] for b in tuner.benched]
    assert all(c.is_dynamic_persistent for c in configs)
    assert {(c.cluster_m, c.cluster_n) for c in configs if c.tile_m == 128} == {(1, 1)}
    assert {(c.cluster_m, c.cluster_n) for c in configs if c.tile_m == 256} == {
        (2, 1),
        (2, 2),
    }


def test_filter_that_removes_everything_falls_back_and_warns_once(kernels, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        autotune_profiles.LOG, "warning", lambda *a, **k: warnings.append(a)
    )
    tuner = kernels.gemm_gated_tuned
    autotune_profiles.apply_autotune_profiles(
        [{"kernel": "gemm_gated_tuned", "backend": "quack", "allow": {"swap_ab": True}}]
    )
    _call(tuner, 131072)
    assert len(tuner.benched) == 76
    tuner.cache.clear()
    _call(tuner, 131072)
    _call(tuner, 65536)
    assert len(warnings) == 2


def test_pin_with_no_valid_config_falls_through_to_the_next_profile(kernels):
    tuner = kernels.gemm_gated_tuned
    autotune_profiles.apply_autotune_profiles(
        [
            {
                "kernel": "gemm_gated_tuned",
                "backend": "quack",
                "configs": [{"tile_m": 128, "tile_n": 256, "cluster_n": 2}],
            },
            {
                "kernel": "gemm_gated_tuned",
                "backend": "quack",
                "allow": {"tile_m": 256},
            },
        ]
    )
    _call(tuner, 131072)
    assert len(tuner.benched) == 28
    assert {b.kwargs["config"].tile_m for b in tuner.benched} == {256}


def test_profile_without_backend_is_not_applied_to_quack(kernels):
    tuner = kernels.gemm_tuned
    assert (
        autotune_profiles.apply_autotune_profiles(
            [{"kernel": "gemm_*", "configs": [{"BLOCK_M": 64}]}], quiet=True
        )
        == 0
    )
    _call(tuner, 65536)
    assert len(tuner.benched) == 74


def test_bad_allow_rules_are_rejected(kernels):
    for allow in (
        {"tile_q": [1]},
        {"where": [{"if": {"tile_m": 128}}]},
        {"where": [{"if": {"bogus": 1}, "then": {"tile_m": 256}}]},
    ):
        with pytest.raises(ValueError, match="allow"):
            autotune_profiles.apply_autotune_profiles(
                [{"kernel": "gemm_tuned", "backend": "quack", "allow": allow}]
            )
    with pytest.raises(ValueError, match="backend: quack"):
        autotune_profiles.apply_autotune_profiles(
            [{"kernel": "_scatter2scatter_lora", "allow": {"BLOCK_M": [64]}}]
        )


def _run_report(*argv):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "autotune_report.py"), *argv],
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_allow_flags_pruned_winners_and_near_configs(kernels, tmp_path):
    autotune_cache.install_timing_capture()
    _call(kernels.gemm_tuned, 131072)
    path = tmp_path / "b200.json"
    autotune_cache.AutotuneCacheCallback(str(path), sonicmoe=True).on_save(
        None, None, None
    )
    disk = tmp_path / "quack" / "HASH" / "gemm_tuned.autotune.json"
    disk.parent.mkdir(parents=True)
    disk.write_text(
        json.dumps(
            {
                "key": ["False", "torch.Size([65536, 2560])"],
                "configs_timings": [
                    [
                        "config: GemmConfig(tile_m=128, tile_n=160, cluster_m=1, "
                        "cluster_n=2, swap_ab=False, is_dynamic_persistent=True, "
                        "use_tma_gather=False)",
                        [0.30, 0.30, 0.30],
                    ],
                    [
                        "config: GemmConfig(tile_m=128, tile_n=64, cluster_m=1, "
                        "cluster_n=1, swap_ab=False, is_dynamic_persistent=False, "
                        "use_tma_gather=False)",
                        [0.31, 0.31, 0.31],
                    ],
                ],
            }
        )
    )
    failed = tmp_path / "quack" / "HASH2" / "gemm_tuned.autotune.json"
    failed.parent.mkdir(parents=True)
    failed.write_text(
        json.dumps(
            {
                "key": ["False", "torch.Size([32768, 2560])"],
                "configs_timings": [
                    [
                        "config: GemmConfig(tile_m=128, tile_n=64, cluster_m=1, "
                        "cluster_n=1, swap_ab=False, is_dynamic_persistent=False, "
                        "use_tma_gather=False)",
                        [None, None, None],
                    ]
                ],
            }
        )
    )
    readme_config = tmp_path / "readme.yaml"
    readme_config.write_text(
        yaml.safe_dump({"autotune_profiles": _readme_allow_profiles()})
    )
    ok = _run_report(
        str(path),
        "--quack-cache-dir",
        str(tmp_path / "quack"),
        "--check-allow",
        str(readme_config),
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "prunes 1/2 configs within 5% of best" in ok.stdout
    assert 'tile_n": 64' in ok.stdout

    strict = tmp_path / "strict.yaml"
    strict.write_text(
        yaml.safe_dump(
            {
                "autotune_profiles": [
                    {
                        "kernel": "gemm_tuned",
                        "backend": "quack",
                        "allow": {"cluster": [[1, 1], [2, 1]], "tile_m": [256]},
                    }
                ]
            }
        )
    )
    bad = _run_report(
        str(path),
        "--quack-cache-dir",
        str(tmp_path / "quack"),
        "--check-allow",
        str(strict),
    )
    assert bad.returncode == 1
    # the captured 131072-row winner (256x512 c2x1) passes; the disk-cache 128x160 c1x2 does not
    assert bad.stdout.count("PRUNES WINNER") == 1
    assert '"tile_m": 128, "tile_n": 160' in bad.stdout


def test_slow_key_profiles_cover_all_three_kernels_slowest_first(kernels, tmp_path):
    autotune_cache.install_timing_capture()
    for name in ("gemm_gated_tuned", "gemm_tuned", "gemm_dgated_tuned"):
        _call(getattr(kernels, name), 131072)
    kernels.gemm_tuned.sweep = 25.0
    _call(kernels.gemm_tuned, 16384)
    path = tmp_path / "b200.json"
    autotune_cache.AutotuneCacheCallback(str(path), sonicmoe=True).on_save(
        None, None, None
    )
    out = _run_report(str(path), "--min-sweep-seconds", "60")
    assert out.returncode == 0, out.stderr
    snippet = out.stdout[out.stdout.index("autotune_profiles:") :]
    profiles = yaml.safe_load(snippet)["autotune_profiles"]
    assert [p["kernel"] for p in profiles] == [
        "gemm_dgated_tuned",
        "gemm_gated_tuned",
        "gemm_tuned",
    ]
    assert all(p["match"]["A.shape"][0] == 131072 for p in profiles)


ROW_ARGS = {
    "gemm_gated_tuned": ["A", "B", "preact_out", "postact_out"],
    "gemm_tuned": ["A", "B", "out"],
    "gemm_dgated_tuned": ["A", "B", "PreAct", "dx_out", "postact_out"],
}


def _call_rows(tuner, rows):
    """Every non-weight tensor gets ``rows`` rows, passed by the kernel's own argument names."""
    names = ROW_ARGS[tuner.fn.__name__]
    tuner.arg_names = [*names, "dynamic_scheduler"]
    tensors = {
        name: torch.empty(
            (32, 2560, 5120) if name == "B" else (rows, 2560), device="meta"
        )
        for name in names
    }
    return tuner(dynamic_scheduler=False, **tensors).kwargs["config"]


def _readme_pin_profiles():
    text = README.read_text()
    section = text[text.index("### Pinning the large training buckets") :]
    block = re.findall(r"```yaml\n(.*?)```", section, re.S)[0]
    return yaml.safe_load(block)["autotune_profiles"]


def test_readme_large_bucket_pins_then_allow_filter(kernels):
    autotune_profiles.apply_autotune_profiles(
        _readme_pin_profiles() + _readme_allow_profiles()
    )
    pins = {
        "gemm_gated_tuned": (256, 512, 2, 1, True, True),
        "gemm_dgated_tuned": (256, 256, 2, 1, True, True),
    }
    for rows in (1048576, 1081344, 262144):
        for name, want in pins.items():
            tuner = getattr(kernels, name)
            c = _call_rows(tuner, rows)
            got = (c.tile_m, c.tile_n, c.cluster_m, c.cluster_n)
            got += (c.is_dynamic_persistent, c.use_tma_gather)
            assert got == want and tuner.benched == []
        tuned = kernels.gemm_tuned
        tuned.benched.clear()
        c = _call_rows(tuned, rows)
        assert len(tuned.benched) == 4
        assert {
            (b.kwargs["config"].tile_m, b.kwargs["config"].cluster_m)
            for b in tuned.benched
        } == {
            (256, 2),
            (128, 1),
        }
    expected = {"gemm_gated_tuned": 24, "gemm_tuned": 42, "gemm_dgated_tuned": 52}
    for name, count in expected.items():
        tuner = getattr(kernels, name)
        tuner.benched.clear()
        _call_rows(tuner, 131072)
        assert len(tuner.benched) == count


def test_readme_gemm_tuned_pin_skips_weight_grads_and_other_weights(kernels):
    autotune_profiles.apply_autotune_profiles(
        _readme_pin_profiles() + _readme_allow_profiles()
    )
    tuner = kernels.gemm_tuned
    tuner.arg_names = ["A", "B", "out", "dynamic_scheduler"]
    calls = [
        ((5120, 1048576), (1048576, 2560)),  # weight gradient: A is [hidden, rows]
        ((1048576, 5120), (32, 5120, 5120)),  # same rows, another weight
    ]
    for a_shape, b_shape in calls:
        tuner.benched.clear()
        A = torch.empty(a_shape, device="meta")
        B = torch.empty(b_shape, device="meta")
        tuner(A=A, B=B, out=torch.empty((1, 1), device="meta"), dynamic_scheduler=False)
        assert len(tuner.benched) == 42


def test_range_match_bounds(kernels):
    tuner = kernels.gemm_tuned
    autotune_profiles.apply_autotune_profiles(
        [
            {
                "kernel": "gemm_tuned",
                "backend": "quack",
                "match": {"A.dim0": {"gt": 65536, "le": 262144}},
                "allow": {"tile_m": 256},
            }
        ]
    )
    for rows, count in ((65536, 74), (65537, 26), (262144, 26), (262145, 74)):
        tuner.benched.clear()
        _call_rows(tuner, rows)
        assert len(tuner.benched) == count, rows
    with pytest.raises(ValueError, match="range takes"):
        autotune_profiles.apply_autotune_profiles(
            [
                {
                    "kernel": "gemm_tuned",
                    "backend": "quack",
                    "match": {"A.dim0": {"min": 1}},
                    "allow": {"tile_m": 256},
                }
            ]
        )


def _gemm_config(tile_m, tile_n, gather):
    return (
        f"config: GemmConfig(tile_m={tile_m}, tile_n={tile_n}, tile_k=None, "
        "num_warps=None, pingpong=False, is_dynamic_persistent=True, cluster_m=2, "
        "cluster_n=1, cluster_k=1, swap_ab=False, max_swizzle_size=8, "
        f"device_capacity=10, use_tma_gather={gather})"
    )


def test_readme_gated_ratio_rule_costs_the_one_exception_under_2_percent(tmp_path):
    """The one measured gated shape whose winner breaks tile_n in {tile_m, 2 x tile_m}.

    Beam B200, 884736 expert rows: top three of quack's own disk-cache timings.
    """
    entry = tmp_path / "quack" / "HASH" / "gemm_gated_tuned.autotune.json"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        json.dumps(
            {
                "key": ["swiglu", "False", "torch.Size([884736, 2560])"],
                "configs_timings": [
                    [_gemm_config(256, 224, False), [8.9851] * 3],
                    [_gemm_config(256, 256, True), [9.1249] * 3],
                    [_gemm_config(256, 512, True), [9.1372] * 3],
                ],
            }
        )
    )
    (gated,) = [
        p for p in _readme_allow_profiles() if p["kernel"] == "gemm_gated_tuned"
    ]
    config = tmp_path / "gated.yaml"
    config.write_text(yaml.safe_dump({"autotune_profiles": [gated]}))
    out = _run_report(
        "--quack-cache-dir", str(tmp_path / "quack"), "--check-allow", str(config)
    )
    assert out.returncode == 1 and out.stdout.count("PRUNES WINNER") == 1
    assert '"tile_m": 256, "tile_n": 224' in out.stdout
    allowed = [
        t
        for c, (t, *_rest) in json.loads(entry.read_text())["configs_timings"]
        if autotune_profiles.config_allowed(
            {
                "tile_m": int(c.split("tile_m=")[1].split(",")[0]),
                "tile_n": int(c.split("tile_n=")[1].split(",")[0]),
            },
            gated["allow"],
        )
    ]
    assert min(allowed) / 8.9851 - 1 < 0.02
