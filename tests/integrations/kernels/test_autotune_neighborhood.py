# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""A new bucket searches only around the winner of the same shape."""

import sys
from itertools import product
from types import ModuleType, SimpleNamespace

import pytest

from axolotl.integrations.kernels import autotune_neighborhood as an


class FakeConfig:
    def __init__(self, kwargs, num_warps=4, num_stages=3, num_ctas=1):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas


def _grid():
    return [
        FakeConfig({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k}, w, s)
        for m, n, k, w, s in product(
            [32, 64, 128], [32, 64], [32, 64, 128], [4, 8], [2, 3, 4]
        )
    ]


def _kernel(name):
    def fn():
        pass

    fn.__name__ = name
    return fn


@pytest.fixture
def tuner(monkeypatch, isolated_kernel_modules):
    configs = _grid()
    tuner = SimpleNamespace(
        cache={},
        base_fn=_kernel("_scatter2scatter_lora_mx"),
        configs=configs,
        keys=["M_BUCKET", "N", "K"],
        nargs={},
        prune_configs=lambda kwargs: list(configs),
    )
    mod = ModuleType("axolotl.fake.scattermoe_lora.lora_ops")
    mod._scatter2scatter_lora_mx = tuner
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return tuner


def _winner(tuner):
    return next(
        c
        for c in tuner.configs
        if c.kwargs == {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}
        and c.num_warps == 4
        and c.num_stages == 4
    )


def test_unseen_shape_keeps_the_full_grid(tuner):
    assert an.install_neighborhood_search(1) == 1
    assert len(tuner.prune_configs({"M_BUCKET": 131072, "N": 5120, "K": 2560})) == 108


def test_new_bucket_of_a_measured_shape_searches_the_neighborhood(tuner):
    an.install_neighborhood_search(1)
    tuner.cache[(131072, 5120, 2560, "torch.bfloat16")] = _winner(tuner)
    near = tuner.prune_configs({"M_BUCKET": 262144, "N": 5120, "K": 2560})
    assert _winner(tuner) in near
    # one parameter away: 2 other BLOCK_M + 1 BLOCK_N + 2 BLOCK_K + 1 warps + 2 stages, plus itself
    assert len(near) == 9
    # a different shape is untouched
    assert len(tuner.prune_configs({"M_BUCKET": 262144, "N": 5120, "K": 5120})) == 108


def test_distance_two_covers_pairs_and_picks_the_closest_bucket(tuner):
    an.install_neighborhood_search(2)
    far = tuner.configs[0]
    tuner.cache[(4096, 5120, 2560, "torch.bfloat16")] = far
    tuner.cache[(131072, 5120, 2560, "torch.bfloat16")] = _winner(tuner)
    near = tuner.prune_configs({"M_BUCKET": 262144, "N": 5120, "K": 2560})
    assert _winner(tuner) in near and far not in near
    assert 9 < len(near) < 108
    with pytest.raises(ValueError):
        an.install_neighborhood_search(0)
