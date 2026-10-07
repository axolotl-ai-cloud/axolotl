"""Nemotron Diffusion VLM media adapter: grid math, expansion, pixels (CPU, no model)."""

import re

import pytest
import torch
from PIL import Image

from axolotl.processing_strategies import (
    NEMOTRON_VLM_IMAGE_PAD_ID,
    NemotronDiffusionVLMProcessingStrategy,
    pad_image_batch,
)

_SPECIAL = {
    "<|image_start|>": 18,
    "<|image_pad|>": 19,
    "<|image_break|>": 20,
    "<|image_end|>": 21,
}


class PlaceholderTokenizer:
    """Renders one ``<|image_start|>`` per image block and tokenizes specials to ids 18-21."""

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert not tokenize
        self.kwargs = kwargs
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
                continue
            for block in content:
                parts.append(
                    "<|image_start|>" if block["type"] != "text" else block["text"]
                )
        return "|".join(parts)

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        ids = []
        for piece in re.split(r"(<\|image_[a-z]+\|>)", text):
            if piece in _SPECIAL:
                ids.append(_SPECIAL[piece])
            else:
                ids.extend(ord(character) for character in piece)
        return ids


@pytest.mark.parametrize(
    "size,grid",
    [
        ((1400, 1400), (50, 50)),
        ((1540, 1540), (50, 50)),
        ((28, 28), (1, 1)),
        ((29, 28), (2, 1)),
        ((50, 100), (2, 4)),
        ((1500, 3000), (25, 50)),
        ((1401, 1401), (50, 50)),
        ((3080, 770), (50, 13)),
    ],
)
def test_token_grid_matches_remote_ceiling_division(size, grid):
    adapter = NemotronDiffusionVLMProcessingStrategy()
    assert adapter.image_token_grid(*size) == grid
    assert adapter.image_size(*size) == (grid[0] * 28, grid[1] * 28)


def test_smaller_max_image_size_shrinks_the_grid():
    adapter = NemotronDiffusionVLMProcessingStrategy(max_image_size=784)
    assert adapter.image_token_grid(1540, 1540) == (28, 28)
    assert adapter.image_token_count(1540, 1540) == 28 * 28 + 27 + 2
    with pytest.raises(ValueError, match="multiple of 28"):
        NemotronDiffusionVLMProcessingStrategy(max_image_size=800)


def test_token_count_adds_breaks_start_and_end():
    adapter = NemotronDiffusionVLMProcessingStrategy()
    assert adapter.image_token_count(28, 28) == 3
    assert adapter.image_token_count(28, 56) == 4
    assert adapter.image_token_count(56, 28) == 5
    assert adapter.image_token_count(1540, 1540) == 2500 + 49 + 2
    assert adapter.merged_patch_count((56, 84)) == 6
    with pytest.raises(ValueError, match="multiples of 28"):
        adapter.merged_patch_count((30, 28))


def test_expansion_string_matches_remote_layout():
    adapter = NemotronDiffusionVLMProcessingStrategy()
    assert (
        adapter.image_token_string((56, 56))
        == "<|image_start|><|image_pad|><|image_pad|><|image_break|>"
        "<|image_pad|><|image_pad|><|image_end|>"
    )
    expanded = adapter.expand_image_placeholders(
        "a<|image_start|>b<|image_start|>c", [(28, 28), (28, 56)]
    )
    assert expanded == (
        "a<|image_start|><|image_pad|><|image_end|>"
        "b<|image_start|><|image_pad|><|image_pad|><|image_end|>c"
    )
    with pytest.raises(ValueError, match="placeholders"):
        adapter.expand_image_placeholders("a<|image_start|>", [])


def test_transform_normalizes_like_the_remote_processor():
    adapter = NemotronDiffusionVLMProcessingStrategy()
    pixels = adapter.transform_image(Image.new("RGB", (56, 28), (255, 0, 128)))
    assert pixels.shape == (3, 28, 56)
    assert pixels.dtype == torch.float32
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073))
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711))
    expected = (torch.tensor((255.0, 0.0, 128.0)) / 255.0 - mean) / std
    assert torch.allclose(pixels[:, 0, 0], expected)
    assert torch.allclose(pixels, expected[:, None, None].expand_as(pixels))


def test_transform_resizes_to_the_token_grid_and_flattens_alpha():
    adapter = NemotronDiffusionVLMProcessingStrategy(max_image_size=56)
    pixels = adapter.transform_image(Image.new("RGBA", (300, 100), (0, 0, 0, 0)))
    assert pixels.shape == (3, 28, 56)
    white = (torch.ones(3) - torch.tensor((0.48145466, 0.4578275, 0.40821073))) / (
        torch.tensor((0.26862954, 0.26130258, 0.27577711))
    )
    assert torch.allclose(pixels[:, 5, 5], white, atol=1e-5)


def test_pad_image_batch_pads_bottom_right():
    batch = pad_image_batch([torch.ones(3, 28, 56), torch.full((3, 28, 28), 2.0)])
    assert batch.shape == (2, 3, 28, 56)
    assert torch.all(batch[1, :, :, :28] == 2.0)
    assert torch.all(batch[1, :, :, 28:] == 0.0)


def test_encode_expands_placeholders_from_real_sizes(tmp_path):
    first = tmp_path / "wide.png"
    Image.new("RGB", (56, 28), (10, 20, 30)).save(first)
    second = tmp_path / "square.png"
    Image.new("RGB", (28, 28), (40, 50, 60)).save(second)
    tokenizer = PlaceholderTokenizer()
    adapter = NemotronDiffusionVLMProcessingStrategy(tokenizer)
    encoded = adapter.encode(
        [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": str(first)}},
                    {"type": "image", "image": str(second)},
                    {"type": "text", "text": "state"},
                ],
            },
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert tokenizer.kwargs == {"add_generation_prompt": True, "enable_thinking": False}
    assert encoded["input_ids"].count(NEMOTRON_VLM_IMAGE_PAD_ID) == 3
    assert encoded["input_ids"].count(18) == 2 and encoded["input_ids"].count(21) == 2
    assert encoded["pixel_values"].shape == (2, 3, 28, 56)
    assert encoded["image_sizes"].tolist() == [[28, 56], [28, 28]]
    assert torch.all(encoded["pixel_values"][1, :, :, 28:] == 0)

    text_only = adapter.encode([{"role": "user", "content": "hi"}])
    assert text_only == {"input_ids": [ord("h"), ord("i")]}


def _encode_one(path):
    adapter = NemotronDiffusionVLMProcessingStrategy(PlaceholderTokenizer())
    return adapter.encode(
        [{"role": "user", "content": [{"type": "image", "image": str(path)}]}]
    )


def test_transparent_png_file_composites_onto_white(tmp_path):
    path = tmp_path / "clear.png"
    Image.new("RGBA", (28, 28), (0, 0, 0, 0)).save(path)
    pixels = _encode_one(path)["pixel_values"][0]
    white = (torch.ones(3) - torch.tensor((0.48145466, 0.4578275, 0.40821073))) / (
        torch.tensor((0.26862954, 0.26130258, 0.27577711))
    )
    assert torch.allclose(pixels[:, 5, 5], white, atol=1e-5)


def test_exif_orientation_is_not_applied_like_the_remote_loader(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (56, 28), (200, 10, 10)).save(path, exif=exif.tobytes())
    with Image.open(path) as raw:
        width, height = raw.size
    assert _encode_one(path)["image_sizes"].tolist() == [[height, width]] == [[28, 56]]


def _gradient(width, height):
    import numpy as np

    y, x = np.mgrid[0:height, 0:width]
    noise = np.random.default_rng(0).integers(-20, 20, (height, width, 3))
    pixels = np.stack([x * 255 / width, y * 255 / height, (x + y) % 256], -1)
    return Image.fromarray((pixels + noise).clip(0, 255).astype(np.uint8))


@pytest.mark.parametrize("size", [(300, 100), (3000, 1500)])
def test_resampling_matches_the_remote_cv2_bicubic(size):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from axolotl.processing_strategies import (
        _NEMOTRON_VLM_IMAGE_MEAN,
        _NEMOTRON_VLM_IMAGE_STD,
    )

    image = _gradient(*size)
    adapter = NemotronDiffusionVLMProcessingStrategy()
    ours = adapter.transform_image(image)
    height, width = ours.shape[-2:]
    assert (height, width) != (size[1], size[0])
    remote = cv2.resize(
        np.asarray(image, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    remote = (remote / 255.0 - np.array(_NEMOTRON_VLM_IMAGE_MEAN, np.float32)) / (
        np.array(_NEMOTRON_VLM_IMAGE_STD, np.float32)
    )
    torch.testing.assert_close(
        ours, torch.from_numpy(remote.transpose(2, 0, 1)), atol=2e-3, rtol=0
    )


def test_huge_jpegs_decode_at_a_bounded_draft_size(tmp_path):
    path = tmp_path / "huge.jpg"
    _gradient(6000, 3000).save(path, format="JPEG")
    adapter = NemotronDiffusionVLMProcessingStrategy(max_image_size=280)
    image = adapter.load_image(path)
    pixels = adapter.transform_image(image)
    assert tuple(pixels.shape) == (3, 140, 280)
    assert image.size[0] < 6000 and image.size[0] >= 560


@pytest.mark.parametrize("fmt", ["TIFF", "PPM"])
def test_load_image_rejects_formats_outside_the_allowlist(tmp_path, fmt):
    path = tmp_path / f"x.{fmt.lower()}"
    Image.new("RGB", (4, 4)).save(path, format=fmt)
    with pytest.raises(ValueError, match="is not one of PNG, JPEG"):
        NemotronDiffusionVLMProcessingStrategy.load_image(path)


class _Response:
    def __init__(self, *, status=200, headers=None, chunks=(b"",), redirect=False):
        self.status_code = status
        self.headers = headers or {"Content-Type": "image/png"}
        self.chunks = chunks
        self.is_redirect = redirect
        self.is_permanent_redirect = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, chunk_size):
        yield from self.chunks


def _png_bytes():
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "response,match",
    [
        (_Response(redirect=True, status=302), "redirects"),
        (_Response(headers={"Content-Type": "text/html"}), "Content-Type"),
        (_Response(chunks=(b"x" * (17 << 20),) * 2), "exceeds"),
    ],
)
def test_url_fetch_guards(monkeypatch, response, match):
    from axolotl import processing_strategies

    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(processing_strategies.requests, "get", get)
    with pytest.raises(ValueError, match=match):
        NemotronDiffusionVLMProcessingStrategy.load_image("https://x/y.png")
    assert calls == [{"stream": True, "timeout": 30, "allow_redirects": False}]


def test_url_fetch_enforces_a_total_deadline_and_loads_images(monkeypatch):
    from axolotl import processing_strategies

    data = _png_bytes()
    monkeypatch.setattr(
        processing_strategies.requests,
        "get",
        lambda url, **kwargs: _Response(chunks=(data[:10], data[10:])),
    )
    assert NemotronDiffusionVLMProcessingStrategy.load_image(
        "https://x/y.png"
    ).size == (
        4,
        4,
    )
    clock = iter([0.0, 0.0, 1000.0])
    monkeypatch.setattr(processing_strategies.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="to download"):
        NemotronDiffusionVLMProcessingStrategy.load_image("https://x/y.png")
