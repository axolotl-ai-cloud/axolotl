"""Collator-side context-parallel sharding (CPU, no distributed)."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.utils.collators import (
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)

pytest.importorskip("ringmaster")

from ringmaster import shard as rm_shard  # noqa: E402

from axolotl.integrations.context_parallel import (  # noqa: E402
    ContextParallelConfig,
    ContextParallelPlugin,
)
from axolotl.integrations.context_parallel.collator import (  # noqa: E402
    CP_SHARD_KEY,
    attach_collator_sharding,
    dispatches_batches,
)
from axolotl.integrations.context_parallel.trainer import (
    configure_trainer,  # noqa: E402
)
from axolotl.monkeypatch.ring_attn import set_ring_attn_group  # noqa: E402


class _Tokenizer:
    padding_side = "right"

    @staticmethod
    def pad(features, **kwargs):
        return {
            key: torch.tensor([feature[key] for feature in features])
            for key in features[0]
        }


def _runtime(rank=1, size=2, layout="contiguous"):
    return SimpleNamespace(cp_rank=rank, cp_size=size, shard_load_balance=layout)


def _packed_features():
    return [
        {
            "input_ids": [10, 11, 12, 13, 14, 15, 16, 17],
            "labels": [10, 11, 12, 13, 14, 15, 16, 17],
            "position_ids": [0, 1, 2, 0, 1, 2, 3, 4],
        }
    ]


def test_collator_hands_the_padded_batch_to_cp_shard_and_skips_varlen():
    seen = {}

    def fake_shard(batch):
        seen.update(batch)
        return {**batch, CP_SHARD_KEY: {"cp_rank": 0}}

    collator = DataCollatorForSeq2Seq(
        tokenizer=_Tokenizer(), emit_fa_varlen_kwargs=True, cp_shard=fake_shard
    )

    batch = collator(_packed_features())

    assert isinstance(seen["input_ids"], torch.Tensor)
    assert batch[CP_SHARD_KEY] == {"cp_rank": 0}
    assert "cu_seq_lens_q" not in batch and "cu_seq_lens_q" not in seen
    # without the hook the same collator emits the packed varlen kwargs
    collator.cp_shard = None
    assert "cu_seq_lens_q" in collator(_packed_features())


def test_batch_sampler_collator_shards_the_concatenated_batch():
    shapes = []

    def fake_shard(batch):
        shapes.append(tuple(batch["input_ids"].shape))
        return batch

    collator = V2BatchSamplerDataCollatorForSeq2Seq(
        tokenizer=_Tokenizer(), cp_shard=fake_shard
    )
    chunk = {"input_ids": [1, 2, 3, 4], "position_ids": [0, 1, 0, 1]}
    collator([[chunk, chunk]])

    assert shapes == [(1, 8)]


@pytest.mark.skipif(
    not hasattr(rm_shard, "shard_collated_batch"),
    reason="installed ringmaster predates collator-side sharding",
)
def test_real_shard_keeps_this_ranks_half():
    trainer = SimpleNamespace(
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=_Tokenizer(), emit_fa_varlen_kwargs=True
        ),
        eval_data_collator=None,
    )
    attach_collator_sharding(trainer, _runtime(rank=1, size=2))

    batch = trainer.data_collator(_packed_features())

    assert batch["input_ids"].tolist() == [[14, 15, 16, 17]]
    assert batch["shift_labels"].tolist() == [[15, 16, 17, -100]]
    assert batch[CP_SHARD_KEY]["cp_rank"] == 1
    assert batch[CP_SHARD_KEY]["varlen"][0] == [0, 3, 8]
    assert "cu_seq_lens_q" not in batch and "attention_mask" not in batch


def test_attach_binds_runtime_geometry_and_restores(monkeypatch):
    calls = []

    def recorder(batch, *, cp_rank, cp_size, load_balance):
        calls.append((cp_rank, cp_size, load_balance))
        return batch

    monkeypatch.setattr(rm_shard, "shard_collated_batch", recorder, raising=False)
    train = DataCollatorForSeq2Seq(tokenizer=_Tokenizer())
    evaluation = DataCollatorForSeq2Seq(tokenizer=_Tokenizer())
    trainer = SimpleNamespace(data_collator=train, eval_data_collator=evaluation)

    restore = attach_collator_sharding(
        trainer, _runtime(rank=1, size=4, layout="head_tail")
    )
    train(_packed_features())
    evaluation(_packed_features())

    assert calls == [(1, 4, "head_tail")] * 2
    restore()
    assert train.cp_shard is None and evaluation.cp_shard is None


def test_attach_skips_foreign_collators(monkeypatch):
    monkeypatch.setattr(
        rm_shard, "shard_collated_batch", lambda batch, **_: batch, raising=False
    )
    trainer = SimpleNamespace(data_collator=object(), eval_data_collator=None)

    attach_collator_sharding(trainer, _runtime())  # no attribute errors


def test_attach_skips_old_ringmaster(monkeypatch):
    monkeypatch.delattr(rm_shard, "shard_collated_batch", raising=False)
    trainer = SimpleNamespace(
        data_collator=DataCollatorForSeq2Seq(tokenizer=_Tokenizer()),
        eval_data_collator=None,
    )

    attach_collator_sharding(trainer, _runtime())

    assert trainer.data_collator.cp_shard is None


def test_attach_skips_unsupported_layout(monkeypatch):
    monkeypatch.setattr(
        rm_shard, "shard_collated_batch", lambda batch, **_: batch, raising=False
    )
    trainer = SimpleNamespace(
        data_collator=DataCollatorForSeq2Seq(tokenizer=_Tokenizer()),
        eval_data_collator=None,
    )

    attach_collator_sharding(trainer, _runtime(layout="zigzag"))

    assert trainer.data_collator.cp_shard is None


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({}, False),
        ({"streaming": True}, True),
        ({"pretraining_dataset": "x"}, True),
        ({"streaming": True, "accelerator_config": {"dispatch_batches": False}}, False),
    ],
)
def test_dispatched_batches_keep_pre_hook_sharding(cfg, expected):
    assert dispatches_batches(SimpleNamespace(**cfg)) is expected


def test_plugin_gate_respects_config_switch():
    cfg = SimpleNamespace(streaming=False, pretraining_dataset=None)
    assert ContextParallelPlugin._shards_in_collator(cfg, ContextParallelConfig(size=2))
    assert not ContextParallelPlugin._shards_in_collator(
        cfg, ContextParallelConfig(size=2, shard_in_collator=False)
    )
    assert not ContextParallelPlugin._shards_in_collator(
        SimpleNamespace(streaming=True, pretraining_dataset=None),
        ContextParallelConfig(size=2),
    )


def _trainer(average):
    return SimpleNamespace(
        accelerator=SimpleNamespace(
            parallelism_config=SimpleNamespace(cp_size=2, non_data_parallel_size=2),
            gather=lambda count: torch.stack([count, count]),
        ),
        model_accepts_loss_kwargs=True,
        compute_loss_func=None,
        label_smoother=None,
        _loss_shifts_labels=True,
        args=SimpleNamespace(average_tokens_across_devices=average),
    )


@pytest.mark.parametrize("average, expected", [(True, 8.0), (False, 4.0)])
def test_token_count_sums_pre_cut_shards_over_the_cp_group(
    monkeypatch, average, expected
):
    reduced = []

    def fake_all_reduce(tensor, group=None):
        reduced.append(group)
        tensor.mul_(2)  # the other CP rank holds the same number of targets

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    set_ring_attn_group("cp-group")
    try:
        trainer = _trainer(average)
        restore = configure_trainer(trainer)
        batches = [
            {
                "labels": torch.tensor([[1, 2, -100, 4]]),
                "shift_labels": torch.tensor([[2, -100, 4, -100]]),
                CP_SHARD_KEY: {"cp_rank": 0},
            },
            {
                "labels": torch.tensor([[1, 2, 3, 4]]),
                "shift_labels": torch.tensor([[2, 3, -100, -100]]),
                CP_SHARD_KEY: {"cp_rank": 0},
            },
        ]
        assert trainer._get_num_items_in_batch(batches, "cpu") == expected
        assert reduced == ["cp-group"]
        restore()
    finally:
        set_ring_attn_group(None)


def test_token_count_without_pre_cut_shards_does_not_reduce(monkeypatch):
    def boom(*_, **__):
        raise AssertionError("unexpected all_reduce")

    monkeypatch.setattr(torch.distributed, "all_reduce", boom)
    trainer = _trainer(average=False)
    restore = configure_trainer(trainer)
    batches = [{"labels": torch.tensor([[1, 2, -100, 4]])}]
    assert trainer._get_num_items_in_batch(batches, "cpu") == 1.0
    restore()
