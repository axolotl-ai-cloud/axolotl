from typing import Any

import pytest

from axolotl.integrations.diffusion_decision import prompts


class RecordingTokenizer:
    def __init__(self, output: Any):
        self.output = output
        self.messages: Any = None
        self.kwargs: dict[str, Any] | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return self.output


def test_serializes_state_like_pinned_djev():
    assert prompts.serialize_state("raw °") == "raw °"
    assert prompts.serialize_state({"temperature": "20°"}) == '{"temperature": "20°"}'
    with pytest.raises(ValueError, match="state is required"):
        prompts.serialize_state(None)


def test_structured_content_and_dict_tokenizer_output():
    tokenizer = RecordingTokenizer({"input_ids": [[4, 5, 6]]})

    ids = prompts.decision_prompt_ids(
        tokenizer,
        "system",
        {"value": "20°"},
        thinking=True,
        structured_content=True,
    )

    assert ids == (4, 5, 6)
    assert tokenizer.messages == [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": '{"value": "20°"}'}],
        },
    ]
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


def test_defaults_to_source_faithful_string_content():
    tokenizer = RecordingTokenizer([7, 8])

    assert prompts.decision_prompt_ids(tokenizer, "system", "raw state") == (7, 8)
    assert tokenizer.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "raw state"},
    ]


def test_rejects_multiple_prompts_or_non_token_output():
    with pytest.raises(ValueError, match="more than one prompt"):
        prompts.decision_prompt_ids(
            RecordingTokenizer([[1], [2]]), "system", "state", structured_content=False
        )
    with pytest.raises(TypeError, match="token ids"):
        prompts.decision_prompt_ids(
            RecordingTokenizer(object()), "system", "state", structured_content=False
        )
    with pytest.raises(TypeError, match="integer token ids"):
        prompts.decision_prompt_ids(
            RecordingTokenizer([1.5]), "system", "state", structured_content=False
        )
