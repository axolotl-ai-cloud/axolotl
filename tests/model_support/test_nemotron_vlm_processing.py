"""Unit contracts for the source-backed Nemotron Diffusion VLM adapter."""

from types import SimpleNamespace

import numpy as np
import torch
import transformers

from axolotl.model_support import get_model_support, resolve_model_support
from axolotl.model_support.nemotron_diffusion import _processor_class, processing


class _Tokenizer:
    def __init__(self):
        self.messages = None
        self.prompt = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "<image><image>state"

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        assert kwargs == {"return_tensors": "pt"}
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))


class _ImageProcessor:
    IMG_START_TOKEN = "<image>"
    IMG_START_ID = 18
    IMG_PAD_ID = 19
    IMG_BREAK_ID = 20
    IMG_END_ID = 21
    DEFAULT_PATCH_SIZE = 2
    DEFAULT_SPATIAL_MERGE_SIZE = 2

    @staticmethod
    def load_image(source):
        return source

    @staticmethod
    def encode_image(source):
        if source == "first":
            return 1, 2, np.zeros((3, 4, 2), dtype=np.float32)
        return 2, 1, np.ones((3, 2, 4), dtype=np.float32)

    @staticmethod
    def build_image_token_str(width, height):
        return f"<{width}x{height}>"


def test_processor_expands_images_in_message_order_and_keeps_ragged_pixels(monkeypatch):
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        processing, "_image_processing_module", lambda *args: _ImageProcessor
    )

    result = processing.prepare_decision_prompt(
        tokenizer,
        "system",
        {"value": 1},
        ["first", "second"],
        model_source="nvidia/Nemotron-Labs-Diffusion-VLM-8B",
    )

    assert tokenizer.messages[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "first"}},
        {"type": "image_url", "image_url": {"url": "second"}},
        {"type": "text", "text": '{"value": 1}'},
    ]
    assert tokenizer.prompt == "<1x2><2x1>state"
    assert result["input_ids"] == [1, 2, 3]
    assert [pixels.shape for pixels in result["pixel_values"]] == [
        (3, 4, 2),
        (3, 2, 4),
    ]
    torch.testing.assert_close(result["image_sizes"], torch.tensor([[8, 4], [4, 8]]))
    assert result["image_token_ids"] == (18, 19, 20, 21)


def test_vlm_descriptor_is_multimodal_and_provides_native_processor():
    support = get_model_support("nemotron_labs_diffusion_vlm")
    assert support is not None
    resolved = resolve_model_support(support)
    assert resolved.is_multimodal is True
    assert resolved.strategies.auto_processor_cls is _processor_class
    assert _processor_class().__name__ == "NemotronVLMProcessor"


def test_processor_accepts_unbatched_input_ids_without_tensor_return(monkeypatch):
    class ListTokenizer(_Tokenizer):
        def __call__(self, prompt, **kwargs):
            self.prompt = prompt
            assert kwargs == {"return_tensors": None}
            return SimpleNamespace(input_ids=[1, 2, 3])

    monkeypatch.setattr(
        processing, "_image_processing_module", lambda *args: _ImageProcessor
    )

    result = processing.prepare_messages(
        ListTokenizer(),
        [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "first"}},
                    {"type": "image_url", "image_url": {"url": "second"}},
                    {"type": "text", "text": "state"},
                ],
            },
        ],
        model_source="example/vlm",
        return_tensors=None,
    )

    assert result["input_ids"] == [1, 2, 3]


def test_saved_processor_loads_its_saved_tokenizer(monkeypatch, tmp_path):
    saved = tmp_path / "processor"
    saved.mkdir()
    (saved / "nemotron_vlm_processor.json").write_text(
        '{"model_source": "example/original", "revision": "commit"}'
    )
    calls = []
    created = {}

    def from_pretrained(source, **kwargs):
        calls.append((source, kwargs))
        return object()

    def init(self, tokenizer, *, model_source, revision=None, max_image_size=None):
        created.update(
            tokenizer=tokenizer,
            model_source=model_source,
            revision=revision,
            max_image_size=max_image_size,
        )

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(processing.NemotronVLMProcessor, "__init__", init)

    processing.NemotronVLMProcessor.from_pretrained(saved, trust_remote_code=True)

    assert calls == [(saved, {"revision": "commit", "trust_remote_code": True})]
    assert created["model_source"] == "example/original"
