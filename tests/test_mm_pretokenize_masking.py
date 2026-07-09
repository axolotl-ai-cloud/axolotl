"""Parity tests: the pre-tokenize MM label mask must match the eager token-level
ProcessingStrategy scanner. Uses the cached processor, no model weights."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from axolotl.processing_strategies import (
    Qwen2VLProcessingStrategy,
    get_processing_strategy,
)
from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.utils.chat_templates.base import get_chat_template

MODEL = "Qwen/Qwen2-VL-7B-Instruct"


@pytest.fixture(scope="module")
def qwen2vl_processor():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(MODEL)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"Could not load cached {MODEL} processor: {exc!r}")


@pytest.fixture(scope="module")
def qwen2vl_template():
    return get_chat_template("qwen2_vl")


def _conversation():
    img = Image.new("RGB", (56, 56), color=(120, 80, 40))
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "A solid brown square."}],
            },
        ],
        "images": [img],
    }


def _make_strategy(processor, template, *, train_on_inputs=False):
    prompter = ChatTemplatePrompter(
        processor.tokenizer,
        chat_template=template,
        processor=processor,
        max_length=2048,
    )
    return ChatTemplateStrategy(
        prompter,
        tokenizer=processor.tokenizer,
        train_on_inputs=train_on_inputs,
        sequence_len=2048,
        roles_to_train=["assistant"],
        train_on_eos="turn",
        chat_template_type="qwen2_vl",
    )


def _trainable_positions(labels) -> set[int]:
    return {i for i, label in enumerate(labels) if label != -100}


def test_pretokenize_matches_eager_masking(qwen2vl_processor, qwen2vl_template):
    """Pre-tokenize labels == eager scanner labels, and both are nonzero."""
    strategy = _make_strategy(qwen2vl_processor, qwen2vl_template)
    result = strategy._tokenize_single_prompt(dict(_conversation()))

    input_ids = result["input_ids"]
    assert result["length"] == len(input_ids)

    pre_positions = _trainable_positions(result["labels"])

    eager = get_processing_strategy(
        qwen2vl_processor,
        qwen2vl_template,
        "qwen2_vl",
        train_on_inputs=False,
        roles_to_train=["assistant"],
        train_on_eos="turn",
    )
    assert isinstance(eager, Qwen2VLProcessingStrategy)
    eager_labels = eager.process_labels(torch.tensor([input_ids]))[0].tolist()
    eager_positions = _trainable_positions(eager_labels)

    # Nonzero: masking actually fired (the old find_turn path yielded 0 tokens).
    assert pre_positions, (
        "pre-tokenize path produced an all-masked (0 trainable) label set"
    )
    assert eager_positions
    assert pre_positions == eager_positions
    for idx in pre_positions:
        assert result["labels"][idx] == input_ids[idx]


def test_train_on_inputs_does_not_raise_image_token_error(
    qwen2vl_processor, qwen2vl_template
):
    """train_on_inputs=True through the pre-tokenize path must not raise the
    '<image> tokens ... Found 0 <image> tokens and 1 images' ValueError."""
    strategy = _make_strategy(qwen2vl_processor, qwen2vl_template, train_on_inputs=True)
    result = strategy._tokenize_single_prompt(dict(_conversation()))

    # Everything except pad/image-pad tokens is trainable; still nonzero.
    assert _trainable_positions(result["labels"])
