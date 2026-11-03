"""Unknown or missing roles must raise a clear ValueError, not a raw KeyError."""

import pytest

from axolotl.core.datasets.transforms.chat_builder import chat_message_transform_builder


@pytest.fixture
def transform():
    return chat_message_transform_builder(
        train_on_inputs=False,
        conversations_field="messages",
        message_field_role="role",
        message_field_content="content",
    )


def test_unknown_role_raises_value_error_naming_role(transform):
    sample = {"messages": [{"role": "user", "content": "hi"}, {"role": "bot", "content": "hello"}]}

    with pytest.raises(ValueError, match=r"role 'bot'.*messages") as error:
        transform(sample)

    assert "assistant" in str(error.value)


def test_missing_role_field_in_later_message(transform):
    sample = {"messages": [{"role": "user", "content": "hi"}, {"content": "no role"}]}

    with pytest.raises(ValueError, match="role"):
        transform(sample)


def test_known_aliases_still_map(transform):
    sample = {"messages": [{"role": "human", "content": "hi"}, {"role": "gpt", "content": "hello"}]}

    result = transform(sample)

    assert [m["role"] for m in result["conversation"]] == ["user", "assistant"]
    assert [m["weight"] for m in result["conversation"]] == [0, 1]
