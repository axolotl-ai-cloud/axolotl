"""Tests that cfg.image_size is honored on the pre-tokenization path."""

import pytest
import torch
from PIL import Image

from axolotl.prompt_strategies.chat_template import ChatTemplatePrompter, load
from axolotl.utils.dict import DictDefault
from axolotl.utils.images import load_and_resize_image

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{{ eos_token }}{% endfor %}"
)


class FakeTokenizer:
    """Minimal tokenizer stub."""

    eos_token = "</s>"
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [2]


class RecordingProcessor:
    """Processor stub that records the images it receives."""

    def __init__(self):
        self.received_images = None

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):  # noqa: ARG002
        return "user: hello"

    def __call__(self, text=None, images=None, return_tensors="pt"):  # noqa: ARG002
        self.received_images = images
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class TestLoadAndResizeImage:
    """Shared helper semantics: tuple = exact, int = aspect-preserving pad."""

    def test_tuple_resizes_exact(self):
        image = Image.new("RGB", (20, 10))
        out = load_and_resize_image(image, (8, 6))
        assert out.size == (8, 6)

    def test_int_pads_to_square(self):
        image = Image.new("RGB", (20, 10), color=(255, 0, 0))
        out = load_and_resize_image(image, 16)
        assert out.size == (16, 16)
        # aspect preserved: content is 16x8, the rest black padding
        assert out.getpixel((0, 0)) == (0, 0, 0)
        assert out.getpixel((8, 8)) == (255, 0, 0)

    def test_none_returns_loaded_image_unresized(self):
        image = Image.new("RGB", (20, 10))
        out = load_and_resize_image(image, None)
        assert out.size == (20, 10)


class TestBuildPromptImageSize:
    """build_prompt must resize images before the processor sees them."""

    def build(self, image_size):
        processor = RecordingProcessor()
        prompter = ChatTemplatePrompter(
            FakeTokenizer(),
            chat_template=CHAT_TEMPLATE,
            processor=processor,
            image_size=image_size,
        )
        prompter.build_prompt(
            [{"role": "user", "content": "hello"}],
            images=[Image.new("RGB", (20, 10))],
        )
        return processor.received_images

    def test_image_size_applied(self):
        received = self.build(image_size=(8, 6))
        assert received[0].size == (8, 6)

    def test_no_image_size_passes_through(self):
        received = self.build(image_size=None)
        assert received[0].size == (20, 10)


class TestStrategyLoaderPlumbing:
    """cfg.image_size / cfg.image_resize_algorithm must reach the prompter."""

    def test_cfg_image_size_reaches_prompter(self):
        cfg = DictDefault(
            {
                "sequence_len": 512,
                "chat_template": "jinja",
                "chat_template_jinja": CHAT_TEMPLATE,
                "image_size": 256,
                "image_resize_algorithm": Image.Resampling.BICUBIC,
            }
        )
        strategy = load(FakeTokenizer(), cfg, ds_cfg={}, processor=RecordingProcessor())
        assert strategy.prompter.image_size == 256
        assert strategy.prompter.image_resize_algorithm == Image.Resampling.BICUBIC


if __name__ == "__main__":
    pytest.main([__file__])
