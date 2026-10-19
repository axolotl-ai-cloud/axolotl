# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for SonicMoE expert-parallel row padding (pow2 and high-water)."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from axolotl.integrations.kernels.args import KernelsArgs
from axolotl.integrations.kernels.libs.sonicmoe import experts


@pytest.fixture(autouse=True)
def default_padding():
    yield
    experts.configure_ep_row_padding()


def test_pow2_is_the_default():
    experts.configure_ep_row_padding()
    assert [experts.ep_padded_rows(n, True) for n in (0, 1, 1000, 1025, 1 << 20)] == [
        1024,
        1024,
        1024,
        2048,
        1 << 20,
    ]
    assert experts.ep_padded_rows((1 << 20) + 1, True) == 1 << 21


def test_pow2_pads_a_typical_beam_receive_by_half_again():
    # EP=8, 16384 tokens/rank, top_k 8: at most 131072 tokens, ~86k on average
    assert experts.ep_padded_rows(86000, True) == 131072


def test_highwater_bucketing_sequence_with_defaults():
    experts.configure_ep_row_padding("highwater")
    granule = 4096
    # 86000 * 1.03 = 88580 -> 22 granules
    assert experts.ep_padded_rows(86000, True) == 22 * granule
    assert experts.ep_padded_rows(80000, True) == 22 * granule
    assert experts.ep_padded_rows(22 * granule, True) == 22 * granule
    # a count above the mark raises it: 95000 * 1.03 = 97850 -> 24 granules
    assert experts.ep_padded_rows(95000, True) == 24 * granule
    assert experts.ep_padded_rows(22 * granule + 1, True) == 24 * granule
    assert experts.ep_padded_rows(0, True) == 24 * granule
    # the per-rank cap (8 x 16384) is a power of two: headroom does not pad past it
    assert experts.ep_padded_rows(131072, True) == 131072
    assert experts.ep_padded_rows(100000, True) == 131072


def test_highwater_marks_are_separate_for_train_and_eval():
    experts.configure_ep_row_padding("highwater", granule=4096, headroom=1.0)
    assert experts.ep_padded_rows(500000, True) == 503808
    assert experts.ep_padded_rows(5000, False) == 8192
    assert experts.ep_padded_rows(6000, False) == 8192
    assert experts.ep_padded_rows(10, True) == 503808


def test_highwater_never_pads_below_1024_and_reconfigure_resets():
    experts.configure_ep_row_padding("highwater", granule=256, headroom=1.0)
    assert experts.ep_padded_rows(10, True) == 1024
    assert experts.ep_padded_rows(2000, True) == 2048
    experts.configure_ep_row_padding("highwater", granule=256, headroom=1.0)
    assert experts.ep_padded_rows(10, True) == 1024


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "bucketed"}, {"mode": "highwater", "granule": 0}, {"headroom": 0.9}],
)
def test_bad_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        experts.configure_ep_row_padding(**kwargs)


def test_config_schema():
    args = KernelsArgs(
        sonicmoe_ep_row_padding="highwater",
        sonicmoe_ep_row_padding_granule=32768,
        sonicmoe_ep_row_padding_headroom=1.05,
    )
    assert args.sonicmoe_ep_row_padding == "highwater"
    with pytest.raises(ValueError):
        KernelsArgs(sonicmoe_ep_row_padding="next_pow2")
    with pytest.raises(ValueError):
        KernelsArgs(sonicmoe_ep_row_padding_headroom=0.5)


def _scattermoe_stub():
    """The pieces of scattermoe_lora.experts the EP forward imports (it needs Triton)."""
    package = ModuleType("axolotl.integrations.kernels.libs.scattermoe_lora")
    package.__path__ = []
    stub = ModuleType("axolotl.integrations.kernels.libs.scattermoe_lora.experts")
    stub._get_base_param = lambda param: param
    stub._w1_name = lambda module: "gate_up_proj"
    stub.is_nvfp4_param = lambda param: False
    stub.scattermoe_experts_forward_ep = None
    return {package.__name__: package, stub.__name__: stub}


@pytest.mark.parametrize("mode", ["pow2", "highwater"])
def test_ep_forward_pads_with_sentinels_and_slices_back(mode, monkeypatch):
    experts.configure_ep_row_padding(mode, granule=1536, headroom=1.0)
    seen = {}

    def fake_forward(module, hidden_states, top_k_index, top_k_weights):
        seen.update(hidden=hidden_states, index=top_k_index, weights=top_k_weights)
        return hidden_states * 2

    monkeypatch.setattr(experts, "_sonicmoe_forward", fake_forward)
    module = SimpleNamespace(
        num_experts=4,
        training=True,
        gate_up_proj=torch.zeros(1),
        down_proj=torch.zeros(1),
    )
    rows = 1100
    hidden = torch.randn(rows, 8)
    index = torch.randint(0, 4, (rows, 2))
    weights = torch.rand(rows, 2)
    with patch.dict(sys.modules, _scattermoe_stub()):
        out = experts._sonicmoe_ep_forward(module, hidden, index, weights)

    padded = 2048 if mode == "pow2" else 1536
    assert seen["hidden"].shape == (padded, 8)
    assert torch.equal(seen["hidden"][:rows], hidden)
    assert not seen["hidden"][rows:].any()
    assert (seen["index"][rows:] == module.num_experts).all()
    assert torch.equal(seen["index"][:rows], index)
    assert not seen["weights"][rows:].any()
    assert torch.equal(out, hidden * 2)
