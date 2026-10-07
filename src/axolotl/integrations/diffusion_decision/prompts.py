"""Tokenize decision prompts with the loaded tokenizer's chat template."""

import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any


def serialize_state(state: Any) -> str:
    """Match djev's text-state serialization."""
    if state is None:
        raise ValueError("state is required")
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _content(text: str, structured_content: bool) -> str | list[dict[str, str]]:
    if structured_content:
        return [{"type": "text", "text": text}]
    return text


def _input_ids(output: Any) -> tuple[int, ...]:
    value = output["input_ids"] if isinstance(output, Mapping) else output
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("apply_chat_template must return token ids")
    if (
        value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        if len(value) != 1:
            raise ValueError("apply_chat_template returned more than one prompt")
        value = value[0]
    if any(
        isinstance(token, bool) or not isinstance(token, Integral) for token in value
    ):
        raise TypeError("apply_chat_template must return integer token ids")
    return tuple(int(token) for token in value)


def decision_prompt_ids(
    tokenizer: Any,
    system_text: str,
    state: Any,
    *,
    thinking: bool = False,
    structured_content: bool = False,
) -> tuple[int, ...]:
    """Return the source-compatible chat prompt ending at the model turn.

    Plain text is the pinned serving contract. Callers with a tokenizer template
    that requires OpenAI-style text parts must explicitly set
    ``structured_content=True``.
    """
    state_text = serialize_state(state)
    messages = [
        {
            "role": "system",
            "content": _content(system_text, structured_content),
        },
        {
            "role": "user",
            "content": _content(state_text, structured_content),
        },
    ]
    output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return _input_ids(output)
