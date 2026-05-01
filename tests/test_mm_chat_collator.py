"""
Regression tests for MultiModalChatDataCollator shape contracts.

Guard against the transformers 5.x breakage where apply_chat_template's
own `return_dict` parameter (default False) caused it to return the raw
input_ids tensor instead of the full BatchFeature dict, leading to
  IndexError: too many indices for tensor of dimension 2
when downstream code did batch["input_ids"] on the resulting tensor.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import BatchFeature


@pytest.fixture(name="mock_processor")
def fixture_mock_processor():
    """
    A mock processor whose apply_chat_template returns a BatchFeature
    when called with return_dict=True (the correct call convention),
    or a raw input_ids tensor when called without return_dict=True
    (the broken call convention that the bug introduced).
    """
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.image_token = "<|image|>"
    processor.tokenizer.convert_tokens_to_ids = MagicMock(return_value=128256)

    batch_size, seq_len = 2, 16
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    batch_feature = BatchFeature(
        data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    def _apply_chat_template(*args, **kwargs):
        if kwargs.get("return_dict", False):
            return batch_feature
        # Simulate transformers 5.x default behaviour: returns out["input_ids"]
        return input_ids

    processor.apply_chat_template = MagicMock(side_effect=_apply_chat_template)
    processor.chat_template = None
    return processor


@pytest.fixture(name="mock_processing_strategy")
def fixture_mock_processing_strategy(mock_processor):
    from axolotl.processing_strategies import ProcessingStrategy

    strategy = ProcessingStrategy(processor=mock_processor)
    return strategy


class TestMultiModalChatDataCollatorShapeContract:
    """
    Verify that MultiModalChatDataCollator.process_rows returns a dict with
    2-D input_ids and labels, not a raw tensor.  This is the shape contract
    that process_labels depends on.
    """

    def _make_collator(self, mock_processing_strategy):
        from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator

        tokenizer = mock_processing_strategy.processor.tokenizer
        return MultiModalChatDataCollator(
            tokenizer=tokenizer,
            processing_strategy=mock_processing_strategy,
        )

    def _make_examples(self):
        return [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]

    def test_process_rows_returns_dict(self, mock_processing_strategy):
        """batch must be a dict, not a raw tensor."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            batch = collator.process_rows(examples)

        assert isinstance(batch, dict), (
            "process_rows must return a dict (BatchFeature), not a raw tensor. "
            "If it returns a tensor, apply_chat_template was called without "
            "return_dict=True at the top level."
        )

    def test_process_rows_input_ids_shape(self, mock_processing_strategy):
        """batch['input_ids'] must be a 2-D tensor (batch, seq_len)."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            batch = collator.process_rows(examples)

        assert "input_ids" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["input_ids"].ndim == 2, (
            f"input_ids must be 2-D (batch, seq_len), got shape {batch['input_ids'].shape}"
        )

    def test_process_rows_labels_shape(self, mock_processing_strategy):
        """batch['labels'] must be a 2-D tensor matching input_ids shape."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            batch = collator.process_rows(examples)

        assert "labels" in batch
        assert isinstance(batch["labels"], torch.Tensor)
        assert batch["labels"].ndim == 2
        assert batch["labels"].shape == batch["input_ids"].shape

    def test_apply_chat_template_called_with_return_dict_true(
        self, mock_processing_strategy
    ):
        """apply_chat_template must be called with return_dict=True as a keyword arg."""
        collator = self._make_collator(mock_processing_strategy)
        examples = self._make_examples()

        with patch.object(
            mock_processing_strategy,
            "__call__",
            return_value=examples,
        ):
            collator.process_rows(examples)

        call_kwargs = (
            mock_processing_strategy.processor.apply_chat_template.call_args.kwargs
        )
        assert call_kwargs.get("return_dict") is True, (
            "apply_chat_template must be called with return_dict=True as a top-level "
            "keyword argument (not inside processor_kwargs). In transformers 5.x, "
            "apply_chat_template has its own return_dict param (default False) that "
            "controls whether it returns the full BatchFeature or just input_ids."
        )
