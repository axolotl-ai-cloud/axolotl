"""Tests for the cached Gemma4 mm_token_type_ids injection."""

import torch

from axolotl.core.trainers.base import AxolotlTrainer


def _trainer(inject: bool) -> AxolotlTrainer:
    trainer = object.__new__(AxolotlTrainer)
    trainer._inject_mm_token_type_ids = inject
    return trainer


def test_injects_zeros_for_gemma4():
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

    _trainer(True)._maybe_inject_mm_token_type_ids(inputs)

    assert torch.equal(inputs["mm_token_type_ids"], torch.zeros(1, 3, dtype=torch.long))


def test_keeps_collator_provided_values():
    provided = torch.tensor([[0, 1, 1]])
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "mm_token_type_ids": provided}

    _trainer(True)._maybe_inject_mm_token_type_ids(inputs)

    assert inputs["mm_token_type_ids"] is provided


def test_noop_for_other_models():
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

    _trainer(False)._maybe_inject_mm_token_type_ids(inputs)

    assert "mm_token_type_ids" not in inputs


def test_noop_without_input_ids():
    inputs = {"inputs_embeds": torch.zeros(1, 3, 4)}

    _trainer(True)._maybe_inject_mm_token_type_ids(inputs)

    assert "mm_token_type_ids" not in inputs
