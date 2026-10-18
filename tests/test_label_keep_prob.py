"""Tests for stochastic label dropout on configured token ids (``label_keep_prob``)."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.label_dropout import LabelDropout, check_label_keep_prob_ids

STRUCT_A, STRUCT_B = 7, 8


class _Tokenizer:
    padding_side = "right"
    pad_token_id = 0

    @staticmethod
    def pad(features, **kwargs):
        width = max(len(feature["input_ids"]) for feature in features)
        return {
            key: torch.tensor(
                [
                    list(feature[key]) + [0] * (width - len(feature[key]))
                    for feature in features
                ]
            )
            for key in features[0]
        }


def _batch():
    # row 0: user turn [1, 7, 2] untrained, assistant [7, 3, 8, 4, 7] trained
    # row 1: assistant-only, padded
    input_ids = torch.tensor(
        [[1, 7, 2, 7, 3, 8, 4, 7], [7, 5, 8, 8, 6, 0, 0, 0]], dtype=torch.long
    )
    labels = input_ids.clone()
    labels[0, :3] = -100
    labels[1, 5:] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": (labels != -100) | (input_ids != 0),
        "labels": labels,
    }


def _clone(batch):
    return {k: v.clone() for k, v in batch.items()}


def test_keep_prob_one_leaves_batch_identical():
    dropout = LabelDropout({STRUCT_A: 1.0, STRUCT_B: 1.0}, seed=42)
    batch = _batch()
    reference = _clone(batch)
    for step in range(50):
        dropout.apply(batch, dropout.generator(step))
    for key, value in reference.items():
        assert torch.equal(batch[key], value)
        assert batch[key].dtype == value.dtype


def test_keep_prob_zero_masks_only_matched_trained_positions():
    dropout = LabelDropout({STRUCT_A: 0.0}, seed=0)
    batch = _batch()
    reference = _clone(batch)
    dropout.apply(batch, dropout.generator(0))

    trained_match = reference["labels"] == STRUCT_A
    assert torch.all(batch["labels"][trained_match] == -100)
    assert torch.equal(
        batch["labels"][~trained_match], reference["labels"][~trained_match]
    )
    # untrained user-turn occurrence of the id stays untouched, as do inputs
    assert reference["input_ids"][0, 1] == STRUCT_A
    assert batch["labels"][0, 1] == -100 and reference["labels"][0, 1] == -100
    assert torch.equal(batch["input_ids"], reference["input_ids"])
    assert torch.equal(batch["attention_mask"], reference["attention_mask"])


def test_kept_fraction_matches_keep_prob():
    dropout = LabelDropout({STRUCT_A: 0.25, STRUCT_B: 0.75}, seed=3)
    kept = {STRUCT_A: 0, STRUCT_B: 0}
    total = {STRUCT_A: 0, STRUCT_B: 0}
    other_changed = False
    for step in range(4000):
        batch = _batch()
        reference = batch["labels"].clone()
        dropout.apply(batch, dropout.generator(step))
        for token_id in kept:
            sel = reference == token_id
            total[token_id] += int(sel.sum())
            kept[token_id] += int((batch["labels"][sel] == token_id).sum())
        others = (reference != STRUCT_A) & (reference != STRUCT_B)
        other_changed |= not torch.equal(batch["labels"][others], reference[others])
    assert abs(kept[STRUCT_A] / total[STRUCT_A] - 0.25) < 0.02
    assert abs(kept[STRUCT_B] / total[STRUCT_B] - 0.75) < 0.02
    assert not other_changed


def test_draws_are_reproducible_and_vary_by_step_and_rank():
    dropout = LabelDropout({STRUCT_A: 0.5, STRUCT_B: 0.5}, seed=11)

    def masked(step, rank=0, micro_step=0):
        batch = _batch()
        dropout.apply(batch, dropout.generator(step, rank, micro_step))
        return batch["labels"]

    assert torch.equal(masked(5), masked(5))
    masks = {
        tuple(masked(s, r, m).flatten().tolist())
        for s in range(4)
        for r in range(2)
        for m in range(2)
    }
    assert len(masks) > 1


@pytest.mark.parametrize(
    "collator_cls",
    [BatchSamplerDataCollatorForSeq2Seq, V2BatchSamplerDataCollatorForSeq2Seq],
)
def test_packed_batches(collator_cls):
    features = [
        {
            "input_ids": [1, 7, 2],
            "position_ids": [0, 1, 2],
            "attention_mask": [1, 1, 1],
            "labels": [-100, -100, 2],
            "length": 3,
        },
        {
            "input_ids": [7, 8, 7],
            "position_ids": [0, 1, 2],
            "attention_mask": [1, 1, 1],
            "labels": [7, 8, 7],
            "length": 3,
        },
    ]
    batch = collator_cls(tokenizer=_Tokenizer())([features])
    reference = _clone(batch)
    LabelDropout({STRUCT_A: 1.0}).apply(batch, torch.Generator().manual_seed(0))
    assert torch.equal(batch["labels"], reference["labels"])

    LabelDropout({STRUCT_A: 0.0}).apply(batch, torch.Generator().manual_seed(0))
    assert batch["labels"].tolist() == [[-100, -100, 2, -100, 8, -100]]
    assert torch.equal(batch["input_ids"], reference["input_ids"])
    assert torch.equal(batch["position_ids"], reference["position_ids"])


def test_padded_collator_batch():
    features = [
        {"input_ids": [7, 3], "labels": [7, 3]},
        {"input_ids": [8, 7, 7], "labels": [8, -100, 7]},
    ]
    batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer(), padding=False)(features)
    LabelDropout({STRUCT_A: 0.0}).apply(batch, torch.Generator().manual_seed(0))
    assert batch["labels"].tolist() == [[-100, 3, -100], [8, -100, -100]]


def _shifted_batch():
    batch = _batch()
    shift = torch.full_like(batch["labels"], -100)
    shift[:, :-1] = batch["labels"][:, 1:]
    batch["shift_labels"] = shift
    return batch


def test_pre_shifted_batch_masks_shift_labels():
    batch = _shifted_batch()
    reference = _clone(batch)
    LabelDropout({STRUCT_A: 0.0}).apply(batch, torch.Generator().manual_seed(0))

    matched = reference["shift_labels"] == STRUCT_A
    assert matched.any()
    assert torch.all(batch["shift_labels"][matched] == -100)
    assert torch.equal(
        batch["shift_labels"][~matched], reference["shift_labels"][~matched]
    )
    assert torch.equal(batch["labels"], reference["labels"])
    assert torch.equal(batch["input_ids"], reference["input_ids"])


def _trainer(normalize, keep_probs, step=0):
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(
        seed=0,
        process_index=0,
        label_keep_prob_normalize=normalize,
        average_tokens_across_devices=False,
        n_gpu=1,
        world_size=1,
    )
    trainer.state = SimpleNamespace(global_step=step)
    trainer.accelerator = SimpleNamespace(parallelism_config=None)
    trainer.model = SimpleNamespace()
    trainer.is_deepspeed_enabled = False
    trainer.model_accepts_loss_kwargs = True
    trainer.compute_loss_func = None
    trainer._loss_shifts_labels = False
    trainer._label_dropout = LabelDropout(keep_probs, seed=0)
    return trainer


@pytest.mark.parametrize(("normalize", "expected"), [("dropped", 8), ("original", 20)])
def test_num_items_in_batch_normalization(normalize, expected):
    trainer = _trainer(normalize, {STRUCT_A: 0.0, STRUCT_B: 0.0})
    batches, num_items = trainer.get_batch_samples(
        iter([_batch(), _batch(), _batch()]), 2, torch.device("cpu")
    )
    assert len(batches) == 2
    # 10 trained labels per batch, 6 of them configured ids; 2 batches counted
    assert int(num_items) == expected
    for batch in batches:
        assert not torch.isin(batch["labels"], torch.tensor([STRUCT_A, STRUCT_B])).any()


@pytest.mark.parametrize(("normalize", "expected"), [("dropped", 8), ("original", 18)])
def test_num_items_in_batch_counts_shift_labels(normalize, expected):
    trainer = _trainer(normalize, {STRUCT_A: 0.0, STRUCT_B: 0.0})
    batches, num_items = trainer.get_batch_samples(
        iter([_shifted_batch(), _shifted_batch()]), 2, torch.device("cpu")
    )
    # 9 trained shift_labels per batch, 5 of them configured ids
    assert int(num_items) == expected
    for batch in batches:
        assert not torch.isin(
            batch["shift_labels"], torch.tensor([STRUCT_A, STRUCT_B])
        ).any()


def test_get_batch_samples_stops_at_end_of_epoch():
    trainer = _trainer("dropped", {STRUCT_A: 0.5})
    batches, _ = trainer.get_batch_samples(iter([_batch()]), 4, torch.device("cpu"))
    assert len(batches) == 1


def test_get_batch_samples_is_reproducible_per_step():
    def labels_at(step):
        trainer = _trainer("dropped", {STRUCT_A: 0.5, STRUCT_B: 0.5}, step=step)
        batches, _ = trainer.get_batch_samples(
            iter([_batch(), _batch()]), 2, torch.device("cpu")
        )
        return [b["labels"] for b in batches]

    first, again = labels_at(3), labels_at(3)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


def test_vocab_check():
    class _Tok:
        def __len__(self):
            return 10

        @staticmethod
        def convert_ids_to_tokens(token_id):
            return f"<{token_id}>"

        @staticmethod
        def decode(ids):
            return f"<{ids[0]}>"

    check_label_keep_prob_ids({7: 0.5, 9: 0.1}, _Tok())
    with pytest.raises(ValueError, match=r"outside the tokenizer vocabulary"):
        check_label_keep_prob_ids({7: 0.5, 10: 0.1}, _Tok())


class TestLabelKeepProbConfig:
    def test_valid(self, min_base_cfg):
        cfg = validate_config(
            DictDefault(
                label_keep_prob={151672: 0.25, "151670": 0.5},
                label_keep_prob_normalize="original",
            )
            | min_base_cfg
        )
        assert cfg.label_keep_prob == {151672: 0.25, 151670: 0.5}
        assert cfg.label_keep_prob_normalize == "original"

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            ({1: 1.5}, r"must be in \[0, 1\]"),
            ({1: -0.1}, r"must be in \[0, 1\]"),
            ({-1: 0.5}, r"non-negative"),
            ({}, r"at least one token id"),
            ({"<|end|>": 0.5}, r"label_keep_prob"),
        ],
    )
    def test_invalid_values(self, min_base_cfg, value, match):
        with pytest.raises(ValueError, match=match):
            validate_config(DictDefault(label_keep_prob=value) | min_base_cfg)

    def test_invalid_normalize(self, min_base_cfg):
        with pytest.raises(ValueError):
            validate_config(
                DictDefault(label_keep_prob={1: 0.5}, label_keep_prob_normalize="x")
                | min_base_cfg
            )
        with pytest.raises(ValueError, match="requires label_keep_prob"):
            validate_config(
                DictDefault(label_keep_prob_normalize="original") | min_base_cfg
            )

    @pytest.mark.parametrize(
        "extra", [{"rl": "dpo"}, {"reward_model": True}, {"process_reward_model": True}]
    )
    def test_rejects_non_sft(self, min_base_cfg, extra):
        with pytest.raises(ValueError, match="only supported for SFT"):
            validate_config(
                DictDefault(label_keep_prob={1: 0.5}, **extra) | min_base_cfg
            )

    def test_rejects_tensor_parallel(self, min_base_cfg):
        with pytest.raises(ValueError, match="tensor_parallel_size > 1"):
            validate_config(
                DictDefault(label_keep_prob={1: 0.5}, tensor_parallel_size=2)
                | min_base_cfg
            )

    def test_tensor_parallel_without_label_keep_prob(self, min_base_cfg):
        cfg = validate_config(DictDefault(tensor_parallel_size=2) | min_base_cfg)
        assert cfg.tensor_parallel_size == 2
        assert cfg.label_keep_prob is None

    def test_label_keep_prob_without_tensor_parallel(self, min_base_cfg):
        cfg = validate_config(
            DictDefault(label_keep_prob={1: 0.5}, tensor_parallel_size=1) | min_base_cfg
        )
        assert cfg.label_keep_prob == {1: 0.5}
