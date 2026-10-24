# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU persistence of SonicMoE/quack autotune winners next to the Triton cache."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from axolotl.integrations.kernels import autotune_cache
from axolotl.integrations.kernels.sonicmoe_autotune import _is_sonic_module

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192


class AutotuneConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __hash__(self):
        return hash(tuple(sorted(self.kwargs.items())))

    def __eq__(self, other):
        return self.kwargs == other.kwargs


class QuackAutotuner:
    __module__ = "quack.autotuner"

    def __init__(self, module_name):
        self.fn = SimpleNamespace(__name__="gemm_gated_tuned", __module__=module_name)
        self.keys = ["activation", "dynamic_scheduler"]
        self.configs = [
            AutotuneConfig(config=GemmConfig(tile_n=n)) for n in (64, 128, 192)
        ]
        self.cache = {}
        self.configs_timings = None

    def benchmark(self, key, times):
        timings = {c: (t, t, t) for c, t in zip(self.configs, times, strict=True)}
        self.cache[key] = min(timings, key=timings.get)
        self.configs_timings = timings


SIGNATURE = ("swiglu", False, (8192, 5120), (5120, 1), "torch.bfloat16")


@pytest.fixture
def quack_module(monkeypatch, isolated_kernel_modules):
    name = "quack.gemm_interface"
    module = ModuleType(name)
    module.gemm_gated_tuned = QuackAutotuner(name)
    monkeypatch.setattr(
        autotune_cache, "_gpu_identity", lambda: {"gpu_name": "NVIDIA B200"}
    )
    hidden = {
        k: None
        for k in sys.modules
        if _is_sonic_module(k) and not k.startswith("axolotl.")
    }
    with patch.dict(sys.modules, {**hidden, name: module}):
        yield module


def test_quack_timings_and_failed_configs_are_persisted(quack_module, tmp_path):
    tuner = quack_module.gemm_gated_tuned
    assert autotune_cache.install_timing_capture() == 1
    inf = float("inf")
    tuner.benchmark(SIGNATURE, [inf, 0.5, inf])

    path = str(tmp_path / "b200.json")
    callback = autotune_cache.AutotuneCacheCallback(path, sonicmoe=True)
    callback.on_save(None, None, None)

    quack_path = autotune_cache.quack_cache_path(path)
    assert quack_path.endswith("b200.quack.json")
    data = json.loads(Path(quack_path).read_text())
    assert data["gpu_name"] == "NVIDIA B200"
    assert set(data["packages"]) <= {
        "quack-kernels",
        "nvidia-cutlass-dsl",
        "sonic-moe",
        "triton",
    }
    (record,) = data["records"]
    assert record["backend"] == "quack"
    assert record["kernel"] == "gemm_gated_tuned"
    assert record["config"] == {"config": {"tile_m": 128, "tile_n": 128}}
    assert record["key"]["signature"][0] == "swiglu"
    assert record["source"] == "measured"
    assert record["configs_benchmarked"] == 3 and record["failed_configs"] == 2
    assert list(record["timings_ms"].values()) == [0.5]
    assert not Path(path).exists()


def test_quack_file_is_only_written_with_sonicmoe(quack_module, tmp_path):
    quack_module.gemm_gated_tuned.cache[SIGNATURE] = (
        quack_module.gemm_gated_tuned.configs[0]
    )
    path = str(tmp_path / "b200.json")
    autotune_cache.AutotuneCacheCallback(path).on_train_end(None, None, None)
    assert not Path(autotune_cache.quack_cache_path(path)).exists()


def test_early_steps_persist_when_quack_cache_grows(quack_module, tmp_path):
    tuner = quack_module.gemm_gated_tuned
    path = str(tmp_path / "b200.json")
    callback = autotune_cache.AutotuneCacheCallback(path, sonicmoe=True)
    state = SimpleNamespace(global_step=1)
    callback.on_step_begin(None, state, None)
    assert isinstance(tuner.cache, autotune_cache._TimingCache)
    tuner.benchmark(SIGNATURE, [0.3, 0.2, 0.1])
    callback.on_step_end(None, state, None)
    data = json.loads(Path(autotune_cache.quack_cache_path(path)).read_text())
    assert data["records"][0]["source"] == "measured"


def test_report_summarizes_both_files(quack_module, tmp_path):
    tuner = quack_module.gemm_gated_tuned
    autotune_cache.install_timing_capture()
    tuner.benchmark(SIGNATURE, [0.3, 0.2, float("inf")])
    path = tmp_path / "b200.json"
    autotune_cache.write_autotune_cache(
        str(path),
        [
            {
                "kernel": "_scatter2scatter_lora",
                "module_fqn": "m",
                "key": {"N": 5120},
                "config": {"BLOCK_M": 128},
                "source": "profile",
            }
        ],
    )
    callback = autotune_cache.AutotuneCacheCallback(str(path), sonicmoe=True)
    callback.on_train_end(None, None, None)
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "autotune_report.py"), str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "_scatter2scatter_lora: 1 key(s) (1 profile)" in out
    assert "gemm_gated_tuned [quack]: 1 key(s) (1 measured)" in out
    assert "1/3 benchmarked configs failed" in out


HUB_PACKAGE = "_sonic_moe_cuda_89d4054"


class VendoredQuackAutotuner(QuackAutotuner):
    __module__ = f"{HUB_PACKAGE}.quack.autotuner"


def test_vendored_hub_quack_is_captured_with_its_own_version(quack_module, tmp_path):
    quack = ModuleType(f"{HUB_PACKAGE}.quack")
    quack.__version__ = "0.6.1"  # type: ignore[attr-defined]
    gemm = ModuleType(f"{HUB_PACKAGE}.quack.gemm_interface")
    tuner = VendoredQuackAutotuner(gemm.__name__)
    gemm.gemm_gated_tuned = tuner  # type: ignore[attr-defined]
    functional = ModuleType(f"{HUB_PACKAGE}.functional.triton_kernels")
    functional.routing = SimpleNamespace(  # type: ignore[attr-defined]
        fn=SimpleNamespace(__name__="routing", __module__=functional.__name__),
        keys=["N"],
        configs=[],
        cache={},
    )
    vendored = {m.__name__: m for m in (quack, gemm, functional)}
    path = str(tmp_path / "b200.json")
    with patch.dict(sys.modules, {**vendored, "quack.gemm_interface": None}):
        assert autotune_cache.install_timing_capture() == 2
        # 0.6.1 stringifies every key element
        signature = ("swiglu", "False", "torch.Size([8192, 5120])", "torch.bfloat16")
        tuner.benchmark(signature, [float("inf"), 0.4, 0.2])
        autotune_cache.AutotuneCacheCallback(path, sonicmoe=True).on_save(
            None, None, None
        )
    data = json.loads(Path(autotune_cache.quack_cache_path(path)).read_text())
    assert data["quack_loaded"][f"{HUB_PACKAGE}.quack"]["version"] == "0.6.1"
    assert "quack-kernels" not in data["packages"]
    (record,) = data["records"]
    assert record["backend"] == "quack"
    assert record["module_fqn"] == f"{HUB_PACKAGE}.quack.gemm_interface"
    assert record["key"]["signature"] == list(signature)
    assert record["failed_configs"] == 1
    assert list(record["timings_ms"].values()) == [0.2, 0.4]
