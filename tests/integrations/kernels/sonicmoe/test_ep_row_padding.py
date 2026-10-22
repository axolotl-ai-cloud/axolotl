# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for SonicMoE expert-parallel row padding (pow2 and high-water)."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from axolotl.integrations.kernels.args import KernelsArgs
from axolotl.integrations.kernels.libs.sonicmoe import experts


@pytest.fixture(autouse=True)
def default_padding():
    yield
    experts.configure_ep_row_padding()
    experts.configure_ep_compact_routing()


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
    experts.promote_ep_padding_marks()
    assert experts.ep_padded_rows(80000, True) == 22 * granule
    assert experts.ep_padded_rows(22 * granule, True) == 22 * granule
    # a count above the mark pads by its own size: 95000 * 1.03 = 97850 -> 24 granules
    assert experts.ep_padded_rows(95000, True) == 24 * granule
    # ... and the mark only follows at the step boundary
    assert experts.ep_padded_rows(80000, True) == 22 * granule
    experts.promote_ep_padding_marks()
    assert experts.ep_padded_rows(80000, True) == 24 * granule
    assert experts.ep_padded_rows(0, True) == 24 * granule
    # the per-rank cap (8 x 16384) is a power of two: headroom does not pad past it
    assert experts.ep_padded_rows(131072, True) == 131072
    experts.promote_ep_padding_marks()
    assert experts.ep_padded_rows(100000, True) == 131072


def test_highwater_marks_are_separate_for_train_and_eval():
    experts.configure_ep_row_padding("highwater", granule=4096, headroom=1.0)
    assert experts.ep_padded_rows(500000, True) == 503808
    assert experts.ep_padded_rows(5000, False) == 8192
    experts.promote_ep_padding_marks()
    assert experts.ep_padded_rows(6000, False) == 8192
    assert experts.ep_padded_rows(10, True) == 503808


def test_highwater_never_pads_below_1024_and_reconfigure_resets():
    experts.configure_ep_row_padding("highwater", granule=256, headroom=1.0)
    assert experts.ep_padded_rows(10, True) == 1024
    assert experts.ep_padded_rows(2000, True) == 2048
    experts.promote_ep_padding_marks()
    experts.configure_ep_row_padding("highwater", granule=256, headroom=1.0)
    assert experts.ep_padded_rows(10, True) == 1024


def test_a_later_layer_does_not_change_an_earlier_layers_recompute_size():
    experts.configure_ep_row_padding("highwater", granule=1024, headroom=1.0)
    first_forward = experts.ep_padded_rows(3000, True)
    assert experts.ep_padded_rows(9000, True) == 9216  # a later layer, larger receive
    assert experts.ep_padded_rows(3000, True) == first_forward == 3072


def test_marks_promote_at_step_and_evaluation_end():
    experts.configure_ep_row_padding("highwater", granule=1024, headroom=1.0)
    callback = experts.EPPaddingMarksCallback()
    experts.ep_padded_rows(9000, True)
    experts.ep_padded_rows(2000, False)
    callback.on_step_end(None, None, None)
    assert experts.ep_padded_rows(3000, True) == 9216
    assert experts.ep_padded_rows(100, False) == 2048
    experts.ep_padded_rows(5000, False)
    assert experts.ep_padded_rows(100, False) == 2048
    callback.on_evaluate(None, None, None)
    assert experts.ep_padded_rows(100, False) == 5120


class _PaddedLayer(torch.nn.Module):
    """Pads its rows like the EP forward, so its saved tensors carry the padded shape."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        rows = x.shape[0]
        padded = F.pad(x, (0, 0, 0, experts.ep_padded_rows(rows, True) - rows))
        return (padded @ self.weight).square()[:rows]


def test_seeded_marks_pad_the_first_step_to_one_size():
    experts.configure_ep_row_padding("highwater", granule=32768)
    experts.configure_ep_compact_routing(True, granule=32768)
    experts.seed_ep_padding_marks(rows=131072, pairs=360000)
    assert experts.ep_padded_rows(5000, True) == 131072
    assert experts.ep_padded_rows(90000, True) == 131072
    # pairs round up to the granule; eval keeps its own unseeded mark (pow2-capped)
    assert experts._EP_PAIR_MARKS.frozen == {True: 360448}
    assert experts.ep_padded_rows(5000, False) == 8192


def test_seed_never_lowers_a_mark_and_marks_still_grow():
    experts.configure_ep_row_padding("highwater", granule=32768)
    experts.ep_padded_rows(200000, True)
    experts.promote_ep_padding_marks()
    experts.seed_ep_padding_marks(rows=32768)
    assert experts.ep_padded_rows(1000, True) == 229376
    experts.ep_padded_rows(240000, True)
    experts.promote_ep_padding_marks()
    assert experts.ep_padded_rows(1000, True) == 262144


@pytest.mark.parametrize(
    "factor, cp, expected_rows, expected_pairs",
    [
        (None, None, 131072, None),
        (2.75, None, 131072, int(131072 * 8 / 8 * 2.75)),
        (2.75, 2, 65536, int(65536 * 8 / 8 * 2.75)),
    ],
)
def test_plugin_seeds_the_token_cap_and_scaled_pairs(
    monkeypatch, factor, cp, expected_rows, expected_pairs
):
    from axolotl.integrations.kernels import plugin

    seeded = {}
    monkeypatch.setattr(
        experts,
        "seed_ep_padding_marks",
        lambda rows=None, pairs=None: seeded.update(rows=rows, pairs=pairs),
    )
    cfg = SimpleNamespace(
        sequence_len=16384,
        micro_batch_size=1,
        expert_parallel_size=8,
        context_parallel_size=cp,
        sonicmoe_ep_compact_routing=True,
        sonicmoe_ep_pair_seed_factor=factor,
    )
    model = SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=8))
    plugin._seed_ep_padding(cfg, model)
    assert seeded == {"rows": expected_rows, "pairs": expected_pairs}


def test_checkpoint_recompute_after_a_larger_layer_does_not_raise():
    experts.configure_ep_row_padding("highwater", granule=1024, headroom=1.0)
    early, late = _PaddedLayer(), _PaddedLayer()
    x = torch.randn(3000, 4, requires_grad=True)
    y = torch.utils.checkpoint.checkpoint(early, x, use_reentrant=False)
    # a later layer receives more rows before the early layer's recompute in backward
    z = late(torch.randn(9000, 4))
    (y.sum() + z.sum()).backward()
    assert x.grad is not None and early.weight.grad is not None


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

    def fake_forward(module, hidden_states, top_k_index, top_k_weights, pairs=None):
        assert pairs is None
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
