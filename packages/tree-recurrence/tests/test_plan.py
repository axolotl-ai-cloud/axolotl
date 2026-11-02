# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""TreePlan from either TRL tree format, with or without collation padding."""

import numpy as np
import pytest
import torch
from tree_recurrence import TreePlan, tree_plan
from tree_testing import branching_sequences, pack


def _same(a: TreePlan, b: TreePlan) -> None:
    for name in ("cu_seqlens", "parents", "depths", "kv_index", "cu_seqlens_k"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_formats_describe_the_same_tree():
    packed = pack(branching_sequences())
    plans = [
        TreePlan.from_kwargs(packed.kwargs(fmt))
        for fmt in ("trl-1.16", "trl-7686", "parents")
    ]
    for plan in plans[1:]:
        _same(plans[0], plan)
    plan = plans[0]
    np.testing.assert_array_equal(plan.parents, packed.parents.numpy())
    np.testing.assert_array_equal(plan.depths, packed.depths.numpy())
    assert plan.num_levels >= 3 and plan.num_tokens == len(packed.input_ids)
    assert plan.path_tokens > plan.num_tokens


@pytest.mark.parametrize("pad", [-1, 0], ids=["trl-1.16", "trl-7686"])
def test_collation_padding_is_ignored(pad):
    packed = pack(branching_sequences())
    tree = packed.kwargs("trl-7686")
    padded = {
        key: torch.cat([value, value.new_full((5,), pad)])
        if isinstance(value, torch.Tensor)
        else value
        for key, value in tree.items()
    }
    _same(TreePlan.from_kwargs(padded), TreePlan.from_kwargs(tree))


def test_chains_are_a_tree_without_branches():
    plan = TreePlan.from_cu_seqlens(torch.tensor([0, 3, 5, 9]))
    assert plan.num_levels == 1 and plan.parents.tolist() == [-1, -1, -1]
    _same(
        plan,
        TreePlan.from_kwargs(
            pack([[1, 2, 3], [1, 2], [4, 5, 6, 7]], share=False).kwargs()
        ),
    )


def test_layouts():
    # [1 2 3] and [1 2 4]: the trie forks one token before they diverge, so the
    # row is a root "1" with children "2 3" and "2 4"
    plan = TreePlan.from_kwargs(pack([[1, 2, 3], [1, 2, 4]]).kwargs())
    assert plan.cu_seqlens.tolist() == [0, 1, 3, 5]
    assert plan.parents.tolist() == [-1, 0, 0]
    cpu = torch.device("cpu")
    gather, cu, keep = plan.path_layout(cpu)
    # the two root-to-leaf paths; the root keeps its output from the first
    assert gather.tolist() == [0, 1, 2, 0, 3, 4] and cu.tolist() == [0, 3, 6]
    assert keep.tolist() == [0, 1, 2, 4, 5]
    assert plan.path_tokens == 6
    gather, cu, keep = plan.conv_layout(3, cpu)
    # each segment preceded by two path tokens, or by zeros (index 5) before the root
    assert gather.tolist() == [5, 5, 0, 5, 0, 1, 2, 5, 0, 3, 4]
    assert cu.tolist() == [0, 3, 7, 11] and keep.tolist() == [2, 5, 6, 9, 10]
    order, inverse, levels = plan.level_layout(cpu)
    assert [(level.start, level.end) for level in levels] == [(0, 1), (1, 5)]
    assert levels[1].parents.tolist() == [0, 0]


def test_plan_is_built_once_per_row():
    tree = pack([[1, 2, 3], [1, 2, 4]]).kwargs()
    assert tree_plan(tree) is tree_plan(dict(tree))
    assert tree_plan({}) is None


def test_a_segment_must_follow_its_parent():
    with pytest.raises(ValueError, match="follow its parent"):
        TreePlan.from_kwargs(
            {
                "tree_cu_seqlens_q": torch.tensor([0, 1, 2]),
                "tree_segment_parents": torch.tensor([1, -1]),
            }
        )
