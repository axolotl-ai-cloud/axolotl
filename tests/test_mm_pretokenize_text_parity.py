"""Regression tests: the pre-tokenize MM path must preserve turn text under
list-only chat templates (SmolVLM/Idefics3-family), which render plain-string
content as an empty turn. Uses the cached processor, no model weights."""

from __future__ import annotations

import pytest
from PIL import Image

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)

MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
ANSWER = "A solid brown square on a plain background."
FOLLOWUP = "Are you sure about the color?"


@pytest.fixture(scope="module")
def smolvlm_processor():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    except OSError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"Could not load cached {MODEL} processor: {exc!r}")


def _make_strategy(processor, *, train_on_inputs=False):
    prompter = ChatTemplatePrompter(
        processor.tokenizer,
        chat_template=processor.tokenizer.chat_template,
        processor=processor,
        max_length=4096,
    )
    return ChatTemplateStrategy(
        prompter,
        tokenizer=processor.tokenizer,
        train_on_inputs=train_on_inputs,
        sequence_len=4096,
        roles_to_train=["assistant"],
        train_on_eos="turn",
        chat_template_type="tokenizer_default",
    )


def _conversation(content_style: str):
    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    if content_style == "parts":
        assistant_content = [{"type": "text", "text": ANSWER}]
        followup_content = [{"type": "text", "text": FOLLOWUP}]
    else:
        assistant_content = ANSWER
        followup_content = FOLLOWUP
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": followup_content},
        ],
        "images": [img],
    }


@pytest.mark.parametrize("content_style", ["parts", "string"])
def test_non_media_turn_text_survives_pretokenize(smolvlm_processor, content_style):
    """Assistant/follow-up text must appear in input_ids whether the source rows
    carry content-part lists or plain strings."""
    strategy = _make_strategy(smolvlm_processor)
    result = strategy._tokenize_single_prompt(_conversation(content_style))

    decoded = smolvlm_processor.tokenizer.decode(result["input_ids"])
    assert ANSWER in decoded, f"assistant text dropped ({content_style} content)"
    assert FOLLOWUP in decoded, f"follow-up user text dropped ({content_style} content)"


def test_string_and_parts_content_tokenize_identically(smolvlm_processor):
    strategy = _make_strategy(smolvlm_processor)
    ids_parts = strategy._tokenize_single_prompt(_conversation("parts"))["input_ids"]
    ids_string = strategy._tokenize_single_prompt(_conversation("string"))["input_ids"]
    assert ids_parts == ids_string
