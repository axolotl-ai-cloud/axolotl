"""Checkpointing hooks reach the layers through the entry point the installed
transformers uses, without leaning on its deprecated-alias shims."""

import warnings

import pytest
import torch
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from axolotl.monkeypatch.selective_checkpointing import apply_selective_checkpointing
from axolotl.utils.activation_checkpointing import (
    CHECKPOINTING_ARG,
    CHECKPOINTING_KWARGS_ARG,
    disable_activation_checkpointing,
    enable_activation_checkpointing,
)


def _model():
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
        )
    )


def _layers_use_sac(model) -> bool:
    return all(
        layer.gradient_checkpointing
        and layer._gradient_checkpointing_func.keywords["use_reentrant"] is False
        and callable(layer._gradient_checkpointing_func.keywords.get("context_fn"))
        for layer in model.model.layers
    )


def _checkpointing_deprecations(caught):
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, FutureWarning) and "checkpointing" in str(w.message)
    ]


def test_trainer_enables_sac_without_deprecated_names(tmp_path):
    model = _model()
    apply_selective_checkpointing(model, save=[])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        args = TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=2,
            use_cpu=True,
            report_to="none",
            save_strategy="no",
            **{
                CHECKPOINTING_ARG: True,
                CHECKPOINTING_KWARGS_ARG: {"use_reentrant": True},
            },
        )
        ids = torch.randint(0, 64, (2, 8)).tolist()
        dataset = Dataset.from_list([{"input_ids": x, "labels": x} for x in ids])
        Trainer(model=model, args=args, train_dataset=dataset).train()

    assert getattr(args, CHECKPOINTING_ARG) is True
    assert _layers_use_sac(model)
    assert not _checkpointing_deprecations(caught)


def test_deprecated_enable_still_reaches_sac():
    """PEFT's kbit prep and TRL still call ``gradient_checkpointing_enable``."""
    model = _model()
    apply_selective_checkpointing(model, save=[])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
    assert _layers_use_sac(model)


@pytest.mark.parametrize("kwargs", [None, {"use_reentrant": False}])
def test_enable_disable_helpers(kwargs):
    model = _model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enable_activation_checkpointing(model, kwargs)
        assert all(layer.gradient_checkpointing for layer in model.model.layers)
        disable_activation_checkpointing(model)
    assert not any(layer.gradient_checkpointing for layer in model.model.layers)
    assert not _checkpointing_deprecations(caught)
