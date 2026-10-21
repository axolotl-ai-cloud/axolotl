# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Profiles must decide the candidates for unseen keys without touching the serial path."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from axolotl.integrations.kernels import autotune_profiles as ap


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
    tuner = SimpleNamespace(
        cache={}, base_fn=_kernel(name), configs=configs, keys=keys, nargs={}
    )
    tuner.prune_configs = lambda kwargs: list(configs)
    return tuner


@pytest.fixture
def module(monkeypatch, isolated_kernel_modules):
    if "triton" not in sys.modules:
        fake = ModuleType("triton")
        fake.Config = FakeConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "triton", fake)
    configs = [
        FakeConfig({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, 4, 4),
        FakeConfig({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, 4, 4),
        FakeConfig({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, 8, 2),
    ]
    mod = ModuleType("axolotl.fake.scattermoe_lora.lora_ops")
    mod._scatter2scatter_lora = _autotuner(
        "_scatter2scatter_lora", configs, ["M_BUCKET", "N", "K"]
    )
    mod._other = _autotuner("_other", configs, ["K"])
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(ap, "_gpu_name", lambda: "NVIDIA B200")
    return mod


def test_pinned_profile_returns_one_existing_config(module):
    n = ap.apply_autotune_profiles(
        [
            {
                "kernel": "_scatter2scatter_lora",
                "gpu": "b200",
                "match": {"N": 5120, "K": [2560, 5120]},
                "configs": [
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 64,
                        "BLOCK_K": 64,
                        "num_stages": 4,
                        "num_warps": 4,
                    }
                ],
            }
        ]
    )
    assert n == 1
    tuner = module._scatter2scatter_lora
    pruned = tuner.prune_configs({"M_BUCKET": 262144, "N": 5120, "K": 5120})
    assert pruned == [tuner.configs[1]]
    # a key outside the match falls through to the kernel's own pruning
    assert len(tuner.prune_configs({"M_BUCKET": 1024, "N": 256, "K": 128})) == 3
    assert len(module._other.prune_configs({"K": 5120})) == 3


def test_profile_can_narrow_the_search_and_build_new_configs(module):
    ap.apply_autotune_profiles(
        [
            {
                "kernel": "axolotl.fake.*lora_ops._scatter2scatter_lora",
                "configs": [
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 64,
                        "BLOCK_K": 64,
                        "num_stages": 4,
                        "num_warps": 4,
                    },
                    {
                        "BLOCK_M": 256,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "num_stages": 3,
                        "num_warps": 8,
                    },
                ],
            }
        ]
    )
    pruned = module._scatter2scatter_lora.prune_configs(
        {"M_BUCKET": 131072, "N": 5120, "K": 2560}
    )
    assert pruned[0] is module._scatter2scatter_lora.configs[0]
    assert pruned[1].kwargs == {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}
    assert pruned[1].num_warps == 8 and pruned[1].num_stages == 3


def test_other_gpu_profiles_are_ignored(module):
    assert (
        ap.apply_autotune_profiles(
            [
                {
                    "kernel": "_scatter2scatter_lora",
                    "gpu": "H100",
                    "configs": [{"BLOCK_M": 64}],
                }
            ]
        )
        == 0
    )
    assert (
        len(module._scatter2scatter_lora.prune_configs({"M_BUCKET": 1, "N": 1, "K": 1}))
        == 3
    )


def test_profile_without_configs_is_rejected(module):
    with pytest.raises(ValueError):
        ap.apply_autotune_profiles([{"kernel": "_scatter2scatter_lora", "configs": []}])
