"""
Tests for ChatTemplateStrategy._get_messages with str-encoded messages.

Verifies that when a JSON string decodes to a non-list value (e.g., a dict),
_get_messages raises AssertionError with the type of the decoded value, not a
NameError caused by referencing the loop variable `message` before it is bound.
"""

import json
from unittest.mock import MagicMock

import pytest

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)


@pytest.fixture
def strategy():
    """Minimal ChatTemplateStrategy with a mocked tokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    tokenizer.encode.return_value = [2]

    prompter = MagicMock(spec=ChatTemplatePrompter)
    prompter.field_messages = "messages"
    prompter.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    prompter.chat_template_msg_variables = set()

    strat = ChatTemplateStrategy.__new__(ChatTemplateStrategy)
    strat.prompter = prompter
    strat.tokenizer = tokenizer
    strat.train_on_inputs = False
    strat.sequence_len = 512
    strat.roles_to_train = []
    strat.train_on_eos = None
    strat.train_on_eot = None
    strat.eot_tokens = []
    strat._eot_token_ids = set()
    strat.split_thinking = False
    strat.images = "images"
    return strat


class TestGetMessagesStrFormat:
    """Tests for _get_messages when messages is stored as a JSON string."""

    def test_valid_str_messages_returned(self, strategy):
        """A valid JSON list of dicts is decoded and returned."""
        turns = [{"role": "user", "content": "hello"}]
        prompt = {"messages": json.dumps(turns)}
        result = strategy._get_messages(prompt)
        assert result == turns

    def test_non_list_json_raises_assertion_error_with_type(self, strategy):
        """When JSON decodes to a non-list, AssertionError is raised with the
        actual decoded type — NOT a NameError from `type(message)`."""
        # A JSON-encoded dict (not a list) — common mistake
        prompt = {"messages": '{"role": "user", "content": "hi"}'}

        with pytest.raises(AssertionError) as exc_info:
            strategy._get_messages(prompt)

        # The error message should describe the actual type (dict), not raise NameError
        assert "dict" in str(exc_info.value), (
            "AssertionError message should name the actual type; "
            f"got: {exc_info.value}"
        )

    def test_non_dict_turn_raises_assertion_error(self, strategy):
        """When a turn inside the decoded list is not a dict, AssertionError fires."""
        prompt = {"messages": json.dumps(["not_a_dict"])}

        with pytest.raises(AssertionError) as exc_info:
            strategy._get_messages(prompt)

        assert "str" in str(exc_info.value)

    def test_invalid_json_raises_json_decode_error(self, strategy):
        """Malformed JSON raises JSONDecodeError."""
        prompt = {"messages": "not-valid-json"}

        with pytest.raises(json.JSONDecodeError):
            strategy._get_messages(prompt)

    def test_none_messages_raises_value_error(self, strategy):
        """Missing field raises ValueError."""
        with pytest.raises(ValueError, match="null"):
            strategy._get_messages({})
