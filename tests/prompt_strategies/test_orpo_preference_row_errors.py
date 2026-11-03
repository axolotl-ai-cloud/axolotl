"""ORPO preference rows must fail with a named error, not a raw exception."""

import pytest

from axolotl.prompt_strategies.orpo.chat_template import ORPODatasetParsingStrategy


strategy = ORPODatasetParsingStrategy()


def test_missing_chosen_field_is_named():
    with pytest.raises(ValueError, match="missing the `chosen` field"):
        strategy.get_chosen_conversation_thread({"rejected": [{"content": "no"}]})


def test_chosen_as_plain_strings_is_named():
    row = {"chosen": "a good answer", "rejected": "a bad answer"}

    with pytest.raises(ValueError, match="must be a non-empty list of message dicts"):
        strategy.get_chosen_conversation_thread(row)


def test_single_turn_thread_is_named():
    row = {"chosen": [{"role": "user", "content": "question"}]}

    with pytest.raises(ValueError, match="needs at least a user and an assistant turn"):
        strategy.get_chosen_conversation_thread(row)


def test_odd_number_of_turns_is_named():
    row = {"chosen": [{"content": "question"}]}

    with pytest.raises(ValueError, match="odd number of turns"):
        strategy.get_prompt(row)


def test_valid_row_still_parses():
    row = {
        "chosen": [{"content": "question"}, {"content": "good"}],
        "rejected": [{"content": "question"}, {"content": "bad"}],
    }

    chosen = strategy.get_chosen_conversation_thread(row)
    rejected = strategy.get_rejected_conversation_thread(row)

    assert [message.content for message in chosen.messages] == ["question", "good"]
    assert chosen.messages[-1].label is True
    assert [message.content for message in rejected.messages] == ["question", "bad"]
