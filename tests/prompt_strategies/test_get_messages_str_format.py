"""
Tests for ChatTemplateStrategy._get_messages with str-encoded messages.

When a JSON string decodes to a non-list, the assertion must report the decoded type
instead of raising UnboundLocalError on the not-yet-bound loop variable.
"""

import json
from types import SimpleNamespace

import pytest

from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy


@pytest.fixture
def strategy():
    # _get_messages only reads self.prompter.field_messages; skip __init__ so we
    # test the real method without a tokenizer/model.
    strat = ChatTemplateStrategy.__new__(ChatTemplateStrategy)
    strat.prompter = SimpleNamespace(field_messages="messages")
    return strat


class TestGetMessagesStrFormat:
    def test_valid_str_messages_returned(self, strategy):
        turns = [{"role": "user", "content": "hello"}]
        assert strategy._get_messages({"messages": json.dumps(turns)}) == turns

    def test_non_list_json_reports_decoded_type(self, strategy):
        # JSON object instead of a list
        with pytest.raises(AssertionError, match=r"got <class 'dict'>"):
            strategy._get_messages({"messages": '{"role": "user", "content": "hi"}'})

    def test_non_dict_turn_reports_turn_type(self, strategy):
        with pytest.raises(AssertionError, match=r"got <class 'str'> for the turn 0"):
            strategy._get_messages({"messages": json.dumps(["not_a_dict"])})

    def test_invalid_json_raises_json_decode_error(self, strategy):
        with pytest.raises(json.JSONDecodeError):
            strategy._get_messages({"messages": "not-valid-json"})

    def test_none_messages_raises_value_error(self, strategy):
        with pytest.raises(ValueError, match="null"):
            strategy._get_messages({})
