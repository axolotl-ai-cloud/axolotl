"""
Test for KD chat template strategies
"""

from unittest.mock import Mock

import pytest

from axolotl.integrations.kd.chat_template import ChatTemplateStrategyWithKDv2


class TestChatTemplateStrategyWithKDv2:
    """Test v2 strategy correctly handles target_token_ids"""

    @pytest.fixture
    def v2_strategy(self):
        """Create v2 strategy instance with mocked dependencies"""
        # Mock prompter
        mock_prompter = Mock()
        mock_prompter.roles = {"user": "user", "assistant": "assistant"}
        mock_prompter.chat_template_msg_variables = ["role", "content"]
        mock_prompter.chat_template = "{{ messages }}"

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.apply_chat_template = Mock(return_value=[1, 10, 20, 30, 2])
        mock_tokenizer.encode = Mock(return_value=[2])

        return ChatTemplateStrategyWithKDv2(
            prompter=mock_prompter,
            tokenizer=mock_tokenizer,
            train_on_inputs=False,
            sequence_len=512,
            logprobs_field="logprobs",
            gen_temperature=1.0,
            kd_temperature=1.0,
        )

    def test_v2_prepare_kd_fields_adds_target_token_ids(self, v2_strategy):
        """
        Test that v2's _prepare_kd_fields hook adds target_token_ids.

        Validates the Template Method pattern fix where v2 overrides
        the hook to add target_token_ids before transform.
        """
        tokenized = {"input_ids": [1, 10, 20, 30, 2], "labels": [1, 10, 20, 30, 2]}
        original = {"target_token_ids": [[10, 20], [30, 40]]}

        result = v2_strategy._prepare_kd_fields(tokenized, original)

        assert "target_token_ids" in result
        assert result["target_token_ids"] == [[10, 20], [30, 40]]

    def test_v2_prepare_kd_fields_handles_missing_field(self, v2_strategy):
        """Test hook handles missing target_token_ids gracefully"""
        tokenized = {"input_ids": [1, 10, 20, 30, 2], "labels": [1, 10, 20, 30, 2]}
        original = {}

        result = v2_strategy._prepare_kd_fields(tokenized, original)

        assert "target_token_ids" not in result

    def test_v2_transform_requires_target_token_ids(self, v2_strategy):
        """
        Test v2's transform fails without target_token_ids.

        Validates the bug fix - transform expects target_token_ids
        to be added by the hook.
        """
        sample = {
            "input_ids": [1, 10, 20, 30, 2],
            "labels": [1, 10, 20, 30, 2],
            "logprobs": [[-0.1, -0.2], [-0.3, -0.4]],
        }

        with pytest.raises(KeyError, match="target_token_ids"):
            v2_strategy.transform_logprobs(sample)
