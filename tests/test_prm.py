"""CPU regression coverage for the vendored PRM trainer."""

import numpy as np
import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from axolotl.core.trainers.prm.prm_trainer import PRMTrainer, compute_accuracy
from axolotl.core.trainers.trl import AxolotlPRMTrainer
from axolotl.core.training_args import AxolotlPRMConfig


@pytest.fixture
def prm_tokenizer():
    tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "question": 2, "good": 3, "bad": 4, "|": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    )


@pytest.mark.parametrize(
    "last_only,is_eval,labels",
    [
        (False, False, [-100, -100, 1, -100, 0]),
        (True, False, [-100, -100, -100, -100, 0]),
        (True, True, [-100, -100, 1, -100, 0]),
    ],
)
def test_step_labels(prm_tokenizer, last_only, is_eval, labels):
    row = PRMTrainer.tokenize_row(
        {"prompt": "question", "completions": ["good", "bad"], "labels": [True, False]},
        prm_tokenizer,
        step_separator="|",
        max_length=None,
        max_completion_length=None,
        train_on_last_step_only=last_only,
        is_eval=is_eval,
    )
    assert row == {"input_ids": [2, 3, 5, 4, 5], "labels": labels}


def test_accuracy_ignores_masked_tokens():
    logits = np.array([[[0, 1], [0, 1], [0, 1]]])
    labels = np.array([[-100, 1, 0]])
    assert compute_accuracy((logits, labels)) == {"accuracy": 0.5}


@pytest.mark.parametrize("pretokenized", [False, True])
def test_axolotl_prm_train_and_evaluate(tmp_path, prm_tokenizer, pretokenized):
    args = AxolotlPRMConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
        report_to="none",
        max_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        step_separator="|",
    )
    model = BertForTokenClassification(
        BertConfig(
            vocab_size=len(prm_tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=2,
        )
    )
    if pretokenized:
        dataset = Dataset.from_dict(
            {
                "input_ids": [[2, 3, 5], [2, 4, 5]],
                "labels": [[-100, -100, 1], [-100, -100, 0]],
            }
        )
    else:
        dataset = Dataset.from_dict(
            {
                "prompt": ["question"] * 2,
                "completions": [["good"], ["bad"]],
                "labels": [[True], [False]],
            }
        )
    trainer = AxolotlPRMTrainer(
        model=model,
        args=args,
        processing_class=prm_tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    assert isinstance(trainer, PRMTrainer)
    assert trainer.train_dataset[0]["labels"] == [-100, -100, 1]
    assert np.isfinite(trainer.train().training_loss)
    assert trainer.state.global_step == 1
    metrics = trainer.evaluate()
    assert np.isfinite(metrics["eval_loss"])
    assert 0 <= metrics["eval_accuracy"] <= 1
    assert (tmp_path / "checkpoint-1" / "model.safetensors").is_file()
    assert (tmp_path / "README.md").is_file()
