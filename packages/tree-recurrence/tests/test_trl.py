# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The TRL adapter against TRL's own packing: kwargs from collated batches in either
format, chain rows and batchers, and ``layer_types`` hidden from TRL's check."""

import numpy as np
import pytest
import torch
from tree_recurrence import TreePlan
from tree_recurrence.integrations.trl import (
    hidden_layer_types,
    sample_boundaries,
    tree_kwargs_from_batch,
)
from tree_testing import branching_sequences, pack

pytest.importorskip("trl.experimental.async_grpo")
from tree_recurrence.integrations.trl import (  # noqa: E402
    ChainFixedCountBatcher,
    ChainRow,
    ChainTokenBudgetBatcher,
)
from trl.experimental.async_grpo.packing import TreeRow  # noqa: E402


def samples(sequences):
    return [
        {
            "input_ids": np.asarray(ids),
            "completion_mask": np.asarray([0] * 4 + [1] * (len(ids) - 4)),
            "old_log_probs": np.zeros(len(ids), np.float32),
            "advantage": 1.0,
            "group_id": i // 4,
        }
        for i, ids in enumerate(sequences)
    ]


def _same_plan(a, b):
    for name in ("cu_seqlens", "parents", "depths", "kv_index", "cu_seqlens_k"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def _collated(rows, pad):
    """Rows padded to a batch as TRL's collator does, with ``pad`` for tree indices."""
    packed = [row.pack() for row in rows]
    batch = {}
    for key in packed[0]:
        if not key.startswith("tree_"):
            continue
        width = max(len(p[key]) for p in packed)
        batch[key] = torch.stack(
            [
                torch.cat([p[key], p[key].new_full((width - len(p[key]),), pad)])
                for p in packed
            ]
        )
    return packed, batch


def test_trl_rows_and_our_test_packer_agree():
    sequences = branching_sequences()
    row = TreeRow()
    for sample in samples(sequences):
        row.add(sample)
    packed = row.pack()
    ours = pack(sequences)
    assert packed["input_ids"].tolist() == ours.input_ids.tolist()
    trl_tree = {
        k: packed[k]
        for k in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
    }
    _same_plan(TreePlan.from_kwargs(trl_tree), TreePlan.from_kwargs(ours.kwargs()))


def test_batch_of_trl_1_16():
    sequences = branching_sequences()
    rows = [TreeRow(), TreeRow()]
    for sample in samples(sequences):
        rows[0].add(sample)
    for sample in samples(sequences[:2]):
        rows[1].add(sample)
    packed, batch = _collated(rows, pad=-1)
    for rank, row in enumerate(packed):
        tree = tree_kwargs_from_batch({k: v[rank : rank + 1] for k, v in batch.items()})
        for key in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k"):
            assert tree[key].tolist() == row[key].tolist()
        assert [tree["tree_max_seqlen_q"], tree["tree_max_seqlen_k"]] == row[
            "tree_max_seqlens"
        ].tolist()


def test_batch_of_trl_7686():
    """huggingface/trl#7686's collated tree: 0 padding, sizes in tree_sizes."""
    first, second = pack(branching_sequences()), pack(branching_sequences()[:2])
    batch = {}
    for name, attribute in (
        ("tree_kv_index", "kv_index"),
        ("tree_cu_seqlens_q", "cu_seqlens_q"),
        ("tree_cu_seqlens_k", "cu_seqlens_k"),
        ("tree_segment_parents", "parents"),
        ("tree_segment_depths", "depths"),
    ):
        a, b = getattr(first, attribute), getattr(second, attribute)
        b = torch.cat([b, b.new_zeros(len(a) - len(b))])
        batch[name] = torch.stack([a, b])
    sizes = []
    for packed in (first, second):
        lengths, depths = packed.cu_seqlens_q.diff(), packed.depths.tolist()
        sizes.append(
            [
                int(lengths.max()),
                int(packed.cu_seqlens_k.diff().max()),
                len(packed.kv_index),
                int(((lengths + 63) // 64).sum()),
            ]
            + [depths.count(d) for d in range(max(depths) + 1)]
        )
    width = max(map(len, sizes))
    batch["tree_sizes"] = torch.tensor([s + [0] * (width - len(s)) for s in sizes])
    for rank, packed in enumerate((first, second)):
        tree = tree_kwargs_from_batch({k: v[rank : rank + 1] for k, v in batch.items()})
        _same_plan(TreePlan.from_kwargs(tree), TreePlan.from_kwargs(packed.kwargs()))
        assert tree["tree_level_sizes"] == packed.kwargs("trl-7686")["tree_level_sizes"]


def test_chain_rows_share_nothing():
    row = ChainRow()
    for sample in samples(branching_sequences()):
        row.add(sample)
    packed = row.pack()
    tree = {
        k: packed[k]
        for k in ("tree_kv_index", "tree_cu_seqlens_q", "tree_cu_seqlens_k")
    }
    plan = TreePlan.from_kwargs(tree)
    assert plan.num_levels == 1
    assert plan.lengths.tolist() == [len(s) for s in branching_sequences()]
    boundaries = sample_boundaries(tree)
    assert boundaries["cu_seq_lens_q"] is tree["tree_cu_seqlens_q"]
    assert boundaries["max_length_q"] == max(plan.lengths)


@pytest.mark.parametrize("batcher", ["token_budget", "fixed_count"])
def test_chain_batchers(batcher):
    data = samples(branching_sequences())
    if batcher == "token_budget":
        batches = ChainTokenBudgetBatcher(
            data, num_processes=2, token_budget=40, metrics={}
        )
    else:
        batches = ChainFixedCountBatcher(data, num_processes=2, microbatch_size=6)
    rows = [row for micro_batch in batches for row in micro_batch]
    assert rows and all(isinstance(row, ChainRow) for row in rows)
    for row in rows:
        assert len(row.tokens) == sum(len(s["input_ids"]) for s in row.samples)


def test_hidden_layer_types():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"]
    )

    class Model:
        pass

    model = Model()
    model.config = config
    with hidden_layer_types(model):
        assert config.layer_types is None
    assert config.layer_types == ["linear_attention", "full_attention"]
