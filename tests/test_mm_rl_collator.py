"""Tests for on-the-fly image loading in the multimodal RL (DPO/KTO/GRPO) path."""

import base64
import io

import pytest
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor
from trl.data_utils import prepare_multimodal_messages

from axolotl.utils.collators.mm_rl import (
    AxolotlVisionPreferenceCollator,
    MultimodalRLExampleNormalizer,
)

from tests.hf_offline_utils import enable_hf_offline

MM_TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.content is string %}{{ m.content }}{% else %}"
    "{% for c in m.content %}{% if c.type == 'image' %}<image>{% else %}{{ c.text }}"
    "{% endif %}{% endfor %}{% endif %}<end>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _b64_png(size, color) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_normalizer_hoists_inline_and_column_images_in_order(tmp_path):
    path = tmp_path / "inline.png"
    Image.new("RGB", (40, 20), "red").save(path)
    column_image = Image.new("RGB", (12, 12), "blue")
    # Arrow schema merging fills absent struct keys with None.
    example = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(path), "text": None},
                    {"type": "text", "text": "Compare", "path": None},
                    {"type": "image", "path": None, "text": None},
                ],
            }
        ],
        "chosen": [{"role": "assistant", "content": "red then blue"}],
        "images": [column_image],
    }

    out = MultimodalRLExampleNormalizer()([example])[0]

    assert [img.size for img in out["images"]] == [(40, 20), (12, 12)]
    assert out["prompt"][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "Compare"},
        {"type": "image"},
    ]
    # TRL's own placeholder check must accept the result.
    prepare_multimodal_messages(out["prompt"], images=out["images"])


def test_normalizer_multi_turn_prepends_leftover_images_and_resizes():
    example = {
        "prompt": [
            {"role": "user", "content": "What?"},
            {"role": "assistant", "content": "A square."},
            {"role": "user", "content": "Which color?"},
        ],
        "image": _b64_png((40, 20), "red"),
    }

    out = MultimodalRLExampleNormalizer(image_size=32)([example])[0]

    assert "image" not in out
    assert [img.size for img in out["images"]] == [(32, 32)]
    assert out["prompt"][0]["content"][0] == {"type": "image"}
    # TRL must not add placeholders to the later string user turn.
    prepare_multimodal_messages(out["prompt"], images=out["images"])


def test_normalizer_uses_image_when_merged_images_is_none():
    example = {
        "prompt": [{"role": "user", "content": "What?"}],
        "image": Image.new("RGB", (8, 8), "red"),
        "images": None,
    }
    out = MultimodalRLExampleNormalizer()([example])[0]
    assert [img.size for img in out["images"]] == [(8, 8)]


@enable_hf_offline
def test_vision_preference_collator_produces_pixel_values():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    processor = LlavaProcessor(
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
        ),
        tokenizer=tokenizer,
        patch_size=16,
        vision_feature_select_strategy="full",
        chat_template=MM_TEMPLATE,
    )
    examples = [
        {
            "prompt": [{"role": "user", "content": "Describe."}],
            "chosen": [{"role": "assistant", "content": "A red square."}],
            "rejected": [{"role": "assistant", "content": "A cat."}],
            "images": [_b64_png((40, 20), "red")],
        }
    ]
    collator = AxolotlVisionPreferenceCollator(
        processor=processor,
        max_length=512,
        normalizer=MultimodalRLExampleNormalizer(image_size=32),
    )

    batch = collator(examples)

    # chosen + rejected rows each carry the prompt image
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    assert (batch["input_ids"] == image_token_id).sum() == 2 * (32 // 16) ** 2
    assert batch["completion_mask"].sum() > 0


@pytest.mark.parametrize("images", [[], None])
def test_normalizer_text_only_rows(images):
    example = {"prompt": [{"role": "user", "content": "hi"}], "images": images}
    out = MultimodalRLExampleNormalizer()([example])[0]
    assert out["images"] == []
    assert out["prompt"] == [{"role": "user", "content": "hi"}]
