# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Parity tests for :func:`shared_dequant_across_shards`.

The helper hoists the per-shard MXFP4 dequant out of the orthogonal
Strategy A path so that overlapping active-expert sets across shards
dequantize the union once instead of N times. The optimization is
only valid if a shard's slice of the union buffer is *byte-identical*
to what the per-shard ``selective_expert_weights`` call would have
produced.
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
    get_active_experts,
    selective_expert_weights,
    shared_dequant_across_shards,
)

torchao = pytest.importorskip("torchao")
from torchao.prototype.mx_formats.mx_tensor import MXTensor  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MX kernels"
)


class _MockExperts:
    """Same bare wrapper used by ``test_mxfp4_expert_weights.py``."""

    def __init__(self, mx_param, num_experts):
        self.gate_up_proj = mx_param
        self.num_experts = num_experts


def _make_mxfp4(E, N, K, seed):
    torch.manual_seed(seed)
    W = torch.randn(E, N, K, device=DEVICE, dtype=DTYPE)
    return MXTensor.to_mx(W, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)


def _make_overlapping_shard_sei(E, num_shards, seed):
    """Build N shards of ``sorted_expert_idxs`` with deliberate overlap.

    Each shard picks ~E/2 experts at random; with 4 shards the union is
    typically ~E (full coverage) while the intersection is non-empty,
    which is the regime the helper is intended to optimise.
    """
    torch.manual_seed(seed)
    sei = []
    for _ in range(num_shards):
        # Each shard sees ~E/2 distinct experts repeated a few times to
        # mimic top-k routing.
        chosen = torch.randperm(E, device=DEVICE)[: max(2, E // 2)]
        # Repeat each id ~3x and sort so the tensor satisfies the
        # ``sorted_expert_idxs`` contract.
        repeated = chosen.repeat_interleave(3)
        sei.append(torch.sort(repeated).values)
    return sei


def test_shared_dequant_matches_per_shard_bitwise():
    """Union dequant + index gather == per-shard selective dequant, bitwise.

    Uses N=4 shards over E=16 experts with deliberate overlap (each shard
    picks ~half the experts, union covers most of E). The helper must
    yield the exact same compact buffer that a per-shard
    ``selective_expert_weights`` call would have produced for that shard.
    """
    E, N, K = 16, 128, 256
    num_shards = 4
    mx = _make_mxfp4(E, N, K, seed=13)
    experts = _MockExperts(mx, E)

    sei_per_shard = _make_overlapping_shard_sei(E, num_shards, seed=13)
    # Sanity: at least one pair of shards must overlap for this test to
    # actually exercise the dedup path.
    actives = [get_active_experts(sei, E) for sei in sei_per_shard]
    union = torch.unique(torch.cat(actives))
    sum_per_shard = sum(a.numel() for a in actives)
    assert sum_per_shard > union.numel(), (
        "shard active sets must overlap to exercise the shared-dequant path"
    )

    union_active, union_buf, shard_into_union = shared_dequant_across_shards(
        experts, "gate_up_proj", sei_per_shard, E
    )

    assert torch.equal(union_active, union)
    assert union_buf.shape == (union.numel(), N, K)

    for i, sei in enumerate(sei_per_shard):
        active_i = get_active_experts(sei, E)
        reference = selective_expert_weights(experts, "gate_up_proj", active_i)
        shared_slice = union_buf.index_select(0, shard_into_union[i])
        assert torch.equal(shared_slice, reference), (
            f"shard {i}: max abs diff = {(shared_slice - reference).abs().max().item()}"
        )


def test_shared_dequant_disjoint_shards():
    """When shards do NOT overlap, the helper still produces the right
    union and the per-shard slices remain bitwise identical."""
    E, N, K = 12, 64, 128
    mx = _make_mxfp4(E, N, K, seed=21)
    experts = _MockExperts(mx, E)

    # Two shards splitting the expert ids into disjoint halves.
    halves = torch.arange(E, device=DEVICE).chunk(2)
    sei_per_shard = [torch.sort(h.repeat_interleave(2)).values for h in halves]

    union_active, union_buf, shard_into_union = shared_dequant_across_shards(
        experts, "gate_up_proj", sei_per_shard, E
    )
    assert union_active.numel() == E

    for i, sei in enumerate(sei_per_shard):
        active_i = get_active_experts(sei, E)
        reference = selective_expert_weights(experts, "gate_up_proj", active_i)
        shared_slice = union_buf.index_select(0, shard_into_union[i])
        assert torch.equal(shared_slice, reference)


def test_shared_dequant_single_shard_noop():
    """N=1 should reduce to the per-shard path: union == only active set."""
    E, N, K = 8, 32, 64
    mx = _make_mxfp4(E, N, K, seed=5)
    experts = _MockExperts(mx, E)

    sei = torch.tensor([0, 0, 2, 2, 5, 5], device=DEVICE, dtype=torch.long)
    union_active, union_buf, shard_into_union = shared_dequant_across_shards(
        experts, "gate_up_proj", [sei], E
    )
    active = get_active_experts(sei, E)
    reference = selective_expert_weights(experts, "gate_up_proj", active)
    assert torch.equal(union_active, active)
    assert torch.equal(union_buf, reference)
    assert torch.equal(union_buf.index_select(0, shard_into_union[0]), reference)
