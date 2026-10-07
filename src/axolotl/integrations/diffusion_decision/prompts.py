"""Tokenize decision prompts with the loaded tokenizer's chat template."""

import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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


def resolve_image_refs(
    images: Sequence[str], base_dir: Path | str | None = None
) -> tuple[str, ...]:
    """Keep URLs and absolute paths; anchor relative paths to ``base_dir`` (default cwd)."""
    root = Path.cwd() if base_dir is None else Path(base_dir)
    resolved = []
    for ref in images:
        if not isinstance(ref, str) or not ref:
            raise ValueError("image references must be nonempty strings")
        if urlparse(ref).scheme in {"http", "https"} or Path(ref).is_absolute():
            resolved.append(ref)
        else:
            resolved.append(str((root / ref).resolve()))
    return tuple(resolved)


def decision_prompt(
    tokenizer: Any,
    system_text: str,
    state: Any,
    *,
    thinking: bool = False,
    structured_content: bool = False,
    images: Sequence[str] = (),
    max_image_size: int = 1400,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Return the chat prompt ids and, for image prompts, the resized image sizes.

    Images are placed before the state text in the user turn; their
    ``<|image_start|>`` placeholders are expanded from the real image sizes.
    """
    state_text = serialize_state(state)
    if images:
        from axolotl.processing_strategies import (
            NemotronDiffusionVLMProcessingStrategy,
        )

        adapter = NemotronDiffusionVLMProcessingStrategy(
            tokenizer, max_image_size=max_image_size
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _content(system_text, structured_content)},
            {
                "role": "user",
                "content": [
                    *(
                        {"type": "image_url", "image_url": {"url": ref}}
                        for ref in images
                    ),
                    {"type": "text", "text": state_text},
                ],
            },
        ]
        encoded = adapter.encode(
            messages, add_generation_prompt=True, enable_thinking=thinking
        )
        sizes = tuple(
            (int(height), int(width)) for height, width in encoded["image_sizes"]
        )
        return _input_ids(encoded["input_ids"]), sizes
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
    return _input_ids(output), ()


def decision_prompt_ids(
    tokenizer: Any,
    system_text: str,
    state: Any,
    *,
    thinking: bool = False,
    structured_content: bool = False,
    images: Sequence[str] = (),
    max_image_size: int = 1400,
) -> tuple[int, ...]:
    """Return the source-compatible chat prompt ending at the model turn.

    Plain text is the pinned serving contract. Callers with a tokenizer template
    that requires OpenAI-style text parts must explicitly set
    ``structured_content=True``.
    """
    return decision_prompt(
        tokenizer,
        system_text,
        state,
        thinking=thinking,
        structured_content=structured_content,
        images=images,
        max_image_size=max_image_size,
    )[0]
