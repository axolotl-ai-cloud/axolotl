"""Per-token SFT loss weights: loss, trainer routing, collation, and dataset plumbing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.monkeypatch.loss.weighted import (
    loss_weights_applied,
    weighted_causal_lm_loss,
)
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.data.shared import merge_datasets
from axolotl.utils.data.utils import truncate_long_seq
from axolotl.utils.dict import DictDefault

VOCAB = 64


@pytest.fixture(name="batch")
def fixture_batch():
    torch.manual_seed(0)
    logits = torch.randn(2, 12, VOCAB, requires_grad=True)
    labels = torch.randint(0, VOCAB, (2, 12))
    labels[0, :4] = -100
    labels[1, 7:] = -100
    weights = torch.rand(2, 12) * 3
    weights[labels == -100] = 0.0
    return logits, labels, weights


def _manual(logits, labels, weights, denom=None):
    shift_logits = logits[:, :-1].reshape(-1, VOCAB)
    shift_labels = labels[:, 1:].reshape(-1)
    shift_weights = weights[:, 1:].reshape(-1)
    mask = shift_labels != -100
    ce = nn.functional.cross_entropy(
        shift_logits[mask], shift_labels[mask], reduction="none"
    )
    total = (ce * shift_weights[mask]).sum()
    return total / (mask.sum() if denom is None else denom)


class TestWeightedLoss:
    def test_matches_manual(self, batch):
        logits, labels, weights = batch
        torch.testing.assert_close(
            weighted_causal_lm_loss(logits, labels, weights),
            _manual(logits, labels, weights),
        )
        torch.testing.assert_close(
            weighted_causal_lm_loss(logits, labels, weights, num_items_in_batch=37),
            _manual(logits, labels, weights, denom=37),
        )

    @pytest.mark.parametrize("num_items", [None, 23])
    def test_unit_weights_reproduce_unweighted(self, batch, num_items):
        logits, labels, _ = batch
        ones = (labels != -100).float()
        weighted = weighted_causal_lm_loss(
            logits, labels, ones, num_items_in_batch=num_items
        )
        reference = ForCausalLMLoss(logits, labels, VOCAB, num_items_in_batch=num_items)
        assert torch.equal(weighted, reference), (weighted, reference)

    def test_gradient_scales_per_token(self, batch):
        logits, labels, weights = batch
        grad = torch.autograd.grad(
            weighted_causal_lm_loss(logits, labels, weights, num_items_in_batch=1),
            logits,
        )[0]
        ref = torch.autograd.grad(_manual(logits, labels, weights, denom=1), logits)[0]
        torch.testing.assert_close(grad, ref)

    def test_rejects_preshifted_labels(self, batch):
        logits, labels, weights = batch
        with pytest.raises(ValueError, match="shift_labels"):
            weighted_causal_lm_loss(logits, labels, weights, shift_labels=labels[:, 1:])


@pytest.fixture(name="tiny_model")
def fixture_tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(name="model_inputs")
def fixture_model_inputs():
    torch.manual_seed(1)
    input_ids = torch.randint(0, VOCAB, (2, 10))
    labels = input_ids.clone()
    labels[:, :3] = -100
    weights = torch.where(labels != -100, 0.5 + torch.rand(2, 10), 0.0)
    return input_ids, labels, weights


class TestModelRouting:
    def test_unit_weights_match_model_loss(self, tiny_model, model_inputs):
        input_ids, labels, _ = model_inputs
        reference = tiny_model(input_ids=input_ids, labels=labels).loss
        with loss_weights_applied(tiny_model, (labels != -100).float()):
            weighted = tiny_model(input_ids=input_ids, labels=labels).loss
        assert torch.equal(weighted, reference), (weighted, reference)

    def test_weighted_model_loss_matches_manual(self, tiny_model, model_inputs):
        input_ids, labels, weights = model_inputs
        with loss_weights_applied(tiny_model, weights):
            out = tiny_model(input_ids=input_ids, labels=labels, num_items_in_batch=9)
        torch.testing.assert_close(
            out.loss, _manual(out.logits.float(), labels, weights, denom=9)
        )

    def test_outside_context_is_unchanged(self, tiny_model, model_inputs):
        input_ids, labels, weights = model_inputs
        before = tiny_model(input_ids=input_ids, labels=labels).loss
        with loss_weights_applied(tiny_model, weights):
            tiny_model(input_ids=input_ids, labels=labels)
        after = tiny_model(input_ids=input_ids, labels=labels).loss
        assert torch.equal(before, after)

    def test_peft_model(self, tiny_model, model_inputs):
        from peft import LoraConfig, get_peft_model

        input_ids, labels, weights = model_inputs
        peft_model = get_peft_model(
            tiny_model, LoraConfig(r=2, target_modules=["q_proj"])
        )
        with loss_weights_applied(peft_model, weights):
            out = peft_model(input_ids=input_ids, labels=labels)
        torch.testing.assert_close(
            out.loss, _manual(out.logits.float(), labels, weights)
        )

    def test_unconsumed_weights_raise(self, tiny_model, model_inputs):
        input_ids, _, weights = model_inputs
        with pytest.raises(RuntimeError, match="silently ignored"):
            with loss_weights_applied(tiny_model, weights):
                tiny_model(input_ids=input_ids)

    def test_compute_loss_pops_and_routes(self, tiny_model, model_inputs):
        input_ids, labels, weights = model_inputs
        seen = {}

        def _compute_loss(model, inputs, return_outputs, num_items_in_batch):
            seen["keys"] = set(inputs)
            return model(**inputs, num_items_in_batch=num_items_in_batch).loss

        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(unwrap_model=lambda m: m),
            _compute_sft_loss=_compute_loss,
        )
        inputs = {"input_ids": input_ids, "labels": labels, "loss_weights": weights}
        loss = AxolotlTrainer.compute_loss(
            trainer, tiny_model, inputs, num_items_in_batch=9
        )
        assert "loss_weights" not in seen["keys"]
        logits = tiny_model(input_ids=input_ids).logits
        torch.testing.assert_close(loss, _manual(logits, labels, weights, denom=9))


class _Tokenizer:
    def __init__(self, padding_side="right"):
        self.padding_side = padding_side

    def pad(self, features, pad_to_multiple_of=None, **kwargs):
        length = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of:
            length = -(-length // pad_to_multiple_of) * pad_to_multiple_of
        out = {}
        for key in features[0]:
            rows = []
            for f in features:
                value = list(f[key])
                if key == "input_ids":
                    pad = [0] * (length - len(value))
                    value = value + pad if self.padding_side == "right" else pad + value
                rows.append(value)
            out[key] = torch.tensor(np.array(rows))
        return out


class TestCollators:
    @pytest.mark.parametrize("side", ["right", "left"])
    def test_pads_with_zero_float(self, side):
        features = [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "loss_weights": [0.0, 0.5, 2.0],
            },
            {"input_ids": [4], "labels": [4], "loss_weights": [1.5]},
        ]
        batch = DataCollatorForSeq2Seq(
            tokenizer=_Tokenizer(side), pad_to_multiple_of=4
        )(features)
        assert batch["loss_weights"].dtype == torch.float32
        assert batch["loss_weights"].shape == batch["labels"].shape == (2, 4)
        expected_short = (
            [1.5, 0.0, 0.0, 0.0] if side == "right" else [0.0, 0.0, 0.0, 1.5]
        )
        assert batch["loss_weights"][1].tolist() == expected_short
        assert ((batch["labels"] == -100) <= (batch["loss_weights"] == 0)).all()

    def test_rows_missing_weights_get_unit_weights(self):
        features = [
            {"input_ids": [1, 2], "labels": [-100, 2], "loss_weights": [0.0, 0.25]},
            {"input_ids": [3, 4], "labels": [3, -100]},
        ]
        batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer())(features)
        assert batch["loss_weights"].tolist() == [[0.0, 0.25], [1.0, 0.0]]

    def test_absent_weights_stay_absent(self):
        batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer())(
            [{"input_ids": [1, 2], "labels": [1, 2]}]
        )
        assert "loss_weights" not in batch

    @pytest.mark.parametrize(
        "collator_cls",
        [BatchSamplerDataCollatorForSeq2Seq, V2BatchSamplerDataCollatorForSeq2Seq],
    )
    def test_packing_concatenates_weights(self, collator_cls):
        pack = [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "attention_mask": [1, 1, 1],
                "position_ids": [0, 1, 2],
                "loss_weights": [0.0, 0.5, 0.5],
            },
            {
                "input_ids": [4, 5],
                "labels": [-100, 5],
                "attention_mask": [1, 1],
                "position_ids": [0, 1],
            },
        ]
        batch = collator_cls(tokenizer=_Tokenizer(), pad_to_multiple_of=8)([pack])
        assert batch["loss_weights"].dtype == torch.float32
        assert batch["loss_weights"].tolist() == [
            [0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
        ]
        assert batch["labels"].shape == batch["loss_weights"].shape


class TestDatasetPlumbing:
    def test_merge_adds_unit_weights_to_unweighted_datasets(self):
        weighted = Dataset.from_list(
            [{"input_ids": [1, 2], "labels": [-100, 2], "loss_weights": [0.0, 0.5]}]
        )
        plain = Dataset.from_list([{"input_ids": [3, 4], "labels": [3, -100]}])
        merged = merge_datasets(
            [weighted, plain], DictDefault(shuffle_merged_datasets=False)
        )
        assert merged["loss_weights"] == [[0.0, 0.5], [1.0, 0.0]]

    def test_merge_without_weights_is_unchanged(self):
        plain = Dataset.from_list([{"input_ids": [3], "labels": [3]}])
        merged = merge_datasets(
            [plain, plain], DictDefault(shuffle_merged_datasets=False)
        )
        assert "loss_weights" not in merged.column_names

    def test_truncation_keeps_weights_aligned(self):
        sample = {
            "input_ids": [[1, 2, 3, 4]],
            "labels": [[1, 2, 3, 4]],
            "loss_weights": [[0.1, 0.2, 0.3, 0.4]],
        }
        out = truncate_long_seq(sample, sequence_len=2)
        assert out["loss_weights"] == [[0.1, 0.2]]
