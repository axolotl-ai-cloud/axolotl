"""Tests for Mistral3Processor with transformers v5 ProcessorMixin integration"""

from unittest.mock import MagicMock

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from axolotl.utils.mistral.mistral3_processor import Mistral3Processor
from axolotl.utils.mistral.mistral_tokenizer import HFMistralTokenizer


@pytest.fixture()
def mock_tokenizer():
    """Create a mock HFMistralTokenizer that passes v5 ProcessorMixin isinstance checks."""
    return MagicMock(spec=HFMistralTokenizer)


@pytest.fixture()
def processor(mock_tokenizer):
    return Mistral3Processor(tokenizer=mock_tokenizer)


class TestMistral3ProcessorInit:
    def test_tokenizer_is_set(self, processor, mock_tokenizer):
        assert processor.tokenizer is mock_tokenizer

    def test_chat_template_is_none(self, processor):
        assert processor.chat_template is None

    def test_audio_tokenizer_is_none(self, processor):
        assert processor.audio_tokenizer is None


class TestApplyChatTemplateTokenized:
    """Test apply_chat_template with tokenize=True, return_dict=True"""

    @pytest.fixture()
    def batched_conversations(self):
        return [
            [
                {"role": "user", "content": "Describe this image."},
                {"role": "assistant", "content": "It is red."},
            ],
            [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "A cat."},
            ],
        ]

    def test_returns_batch_feature_with_pixel_values(
        self, processor, mock_tokenizer, batched_conversations
    ):
        pixel_values = torch.randn(2, 3, 224, 224, dtype=torch.float64)
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "pixel_values": pixel_values,
        }

        result = processor.apply_chat_template(
            batched_conversations, tokenize=True, return_dict=True
        )

        assert isinstance(result, BatchFeature)
        assert "pixel_values" in result
        assert "image_sizes" in result
        assert result["pixel_values"].dtype == torch.float32
        assert result["image_sizes"].shape == (2, 2)
        assert result["image_sizes"][0].tolist() == [224, 224]

    def test_returns_batch_feature_without_pixel_values(
        self, processor, mock_tokenizer, batched_conversations
    ):
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        result = processor.apply_chat_template(
            batched_conversations, tokenize=True, return_dict=True
        )

        assert isinstance(result, BatchFeature)
        assert "input_ids" in result
        assert "image_sizes" not in result


class TestApplyChatTemplateNotTokenized:
    def test_single_conversation_returns_unwrapped(self, processor, mock_tokenizer):
        """Single conversation (not batched) should return unwrapped result."""
        single_conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        mock_tokenizer.apply_chat_template.return_value = [
            "<s>[INST]Hello[/INST]Hi</s>"
        ]

        result = processor.apply_chat_template(
            single_conversation, tokenize=False, return_dict=False
        )

        assert result == "<s>[INST]Hello[/INST]Hi</s>"

    def test_batched_conversations_returns_list(self, processor, mock_tokenizer):
        batched = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            [
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Bye"},
            ],
        ]
        mock_tokenizer.apply_chat_template.return_value = ["text1", "text2"]

        result = processor.apply_chat_template(
            batched, tokenize=False, return_dict=False
        )

        assert result == ["text1", "text2"]


class TestCall:
    def test_delegates_to_tokenizer(self, processor, mock_tokenizer):
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }

        result = processor("Hello world")

        mock_tokenizer.assert_called_once()
        assert isinstance(result, BatchFeature)


class TestReturnTensorsValidation:
    def test_rejects_non_pt_return_tensors(self, processor):
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        with pytest.raises(ValueError, match="only supports.*return_tensors='pt'"):
            processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, return_tensors="np"
            )
