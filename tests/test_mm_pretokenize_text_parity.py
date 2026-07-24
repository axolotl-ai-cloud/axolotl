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
    assistant_content: str | list[dict[str, str]] = ANSWER
    followup_content: str | list[dict[str, str]] = FOLLOWUP
    if content_style == "parts":
        assistant_content = [{"type": "text", "text": ANSWER}]
        followup_content = [{"type": "text", "text": FOLLOWUP}]
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


def test_image_size_resize_matches_eager(smolvlm_processor):
    """cfg.image_size must produce the same tokens pre-tokenized as the eager
    collator path's square-pad resize."""
    from axolotl.processing_strategies import get_processing_strategy
    from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator

    conversation = _conversation("parts")
    conversation["images"] = [Image.new("RGB", (640, 444), color=(120, 80, 40))]

    strategy = _make_strategy(smolvlm_processor)
    strategy.image_size = 512
    resized_ids = strategy._tokenize_single_prompt(dict(conversation))["input_ids"]

    eager = get_processing_strategy(
        smolvlm_processor, None, "tokenizer_default", image_size=512
    )
    collator = MultiModalChatDataCollator(
        tokenizer=smolvlm_processor.tokenizer, processing_strategy=eager
    )
    eager_ids = collator.torch_call([dict(conversation)])["input_ids"][0].tolist()

    strategy_no_resize = _make_strategy(smolvlm_processor)
    native_ids = strategy_no_resize._tokenize_single_prompt(dict(conversation))[
        "input_ids"
    ]

    # A 640x444 source padded to a 512 square hits a larger tile grid than its
    # native aspect ratio, so resize must change the token count and match the
    # eager path exactly.
    assert resized_ids == eager_ids
    assert len(resized_ids) != len(native_ids)
