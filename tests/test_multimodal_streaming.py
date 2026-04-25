"""Multimodal CPT streaming encoder + collator tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from axolotl.prompt_strategies.multimodal_pretrain import build_image_token_spec
from axolotl.utils.collators.mm_pretrain import MultiModalPretrainDataCollator
from axolotl.utils.data.streaming import (
    encode_streaming_multimodal,
    wrap_streaming_dataset,
)
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


@pytest.fixture(scope="module", name="smolvlm_processor")
@enable_hf_offline
def fixture_smolvlm_processor(
    download_smolvlm_500m_instruct_model,  # pylint: disable=unused-argument
):
    return AutoProcessor.from_pretrained(_SMOLVLM)


@pytest.fixture(scope="module", name="two_tiny_images")
def fixture_two_tiny_images(tmp_path_factory) -> list[Path]:
    d = tmp_path_factory.mktemp("mm_stream_imgs")
    out = []
    for i in range(2):
        p = d / f"dummy_{i}.png"
        arr = np.random.default_rng(i).integers(0, 255, (64, 64, 3)).astype("uint8")
        Image.fromarray(arr).save(p)
        out.append(p)
    return out


# ---- encode_streaming_multimodal ------------------------------------------


def test_encode_preserves_images_and_text(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    examples = {
        "text": [
            f"{spec.image_token}\nrow one",
            f"{spec.image_token}\nrow two slightly longer",
        ],
        "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
    }
    out = encode_streaming_multimodal(
        examples,
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    assert set(out) >= {"input_ids", "labels", "attention_mask", "images", "_mm_text"}
    assert len(out["input_ids"]) == 2
    assert out["images"] == [[str(two_tiny_images[0])], [str(two_tiny_images[1])]]
    # EOS appended -> input_ids len equals attention_mask len and > text
    for ids, mask in zip(out["input_ids"], out["attention_mask"], strict=True):
        assert len(ids) == len(mask) and len(ids) > 0
    # CPT: labels == input_ids pre-masking.
    for ids, lbls in zip(out["input_ids"], out["labels"], strict=True):
        assert ids == lbls


def test_encode_rejects_mismatch(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    examples = {
        "text": [f"{spec.image_token}{spec.image_token}\ntwo placeholders one image"],
        "images": [[str(two_tiny_images[0])]],
    }
    with pytest.raises(ValueError, match="occurrence"):
        encode_streaming_multimodal(
            examples,
            tokenizer=smolvlm_processor.tokenizer,
            max_tokens=2048,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
        )


def test_encode_rejects_row_without_list(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    with pytest.raises(ValueError, match="list"):
        encode_streaming_multimodal(
            {
                "text": [f"{spec.image_token}\nrow one"],
                "images": [str(two_tiny_images[0])],  # scalar, not a list
            },
            tokenizer=smolvlm_processor.tokenizer,
            max_tokens=2048,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
        )


def test_encode_counts_placeholders_on_full_text(smolvlm_processor, two_tiny_images):
    # All 3 placeholders must be counted even when text would have been truncated.
    spec = build_image_token_spec(smolvlm_processor)
    long_filler = "lorem ipsum " * 20
    text = f"{spec.image_token} {long_filler} {spec.image_token} {long_filler} {spec.image_token}"
    examples = {
        "text": [text],
        "images": [[str(two_tiny_images[0])] * 3],
    }
    out = encode_streaming_multimodal(
        examples,
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=4096,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    assert sum(1 for t in out["input_ids"][0] if t == spec.image_token_id) == 3


def test_encode_rejects_row_exceeding_max_tokens(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    huge = "word " * 5000
    examples = {
        "text": [f"{spec.image_token} {huge}"],
        "images": [[str(two_tiny_images[0])]],
    }
    with pytest.raises(ValueError, match="exceeds sequence_len"):
        encode_streaming_multimodal(
            examples,
            tokenizer=smolvlm_processor.tokenizer,
            max_tokens=512,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
        )


# ---- wrap_streaming_dataset routing --------------------------------------


def test_wrap_streaming_dataset_uses_pretraining_config_arg(
    smolvlm_processor, monkeypatch
):
    # Eval path passes a per-entry config that may differ from cfg.pretraining_dataset[0].
    # The MM-CPT branch must read from that arg, not re-resolve from cfg.
    captured = {}

    def fake_partial(fn, **kwargs):
        captured["encode_fn"] = fn
        captured["kwargs"] = kwargs
        return lambda batch: batch

    monkeypatch.setattr("axolotl.utils.data.streaming.functools.partial", fake_partial)

    class _Dataset:
        features = {"text": None, "images": None}

        def shuffle(self, **_):
            return self

        def map(self, *_args, **_kwargs):
            return self

    cfg = DictDefault(
        {
            "sample_packing": False,
            "pretraining_dataset": [
                {
                    "path": "train/ds",
                    "type": "multimodal_pretrain",
                    "text_column": "wrong_train_col",
                    "image_column": "wrong_train_imgs",
                }
            ],
            "sequence_len": 256,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 1000,
            "seed": 42,
        }
    )
    eval_entry = DictDefault(
        {
            "path": "test/ds",
            "type": "multimodal_pretrain",
            "text_column": "eval_text",
            "image_column": "eval_imgs",
        }
    )

    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=eval_entry,
    )

    assert captured["encode_fn"] is encode_streaming_multimodal
    assert captured["kwargs"]["text_column"] == "eval_text"
    assert captured["kwargs"]["image_column"] == "eval_imgs"


# ---- MultiModalPretrainDataCollator ---------------------------------------


def test_collator_builds_batch_and_masks_labels(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    encoded = encode_streaming_multimodal(
        {
            "text": [
                f"{spec.image_token}\nrow one",
                f"{spec.image_token}\nrow two slightly longer",
            ],
            "images": [[str(two_tiny_images[0])], [str(two_tiny_images[1])]],
        },
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    rows = [
        {
            k: encoded[k][i]
            for k in ("input_ids", "labels", "attention_mask", "images", "_mm_text")
        }
        for i in range(2)
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    batch = collator.torch_call(rows)
    # Expected keys
    for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
        assert k in batch, f"missing batch key {k}"
    assert isinstance(batch["input_ids"], torch.Tensor)
    # Label masking check: no image-family ids remaining as valid labels.
    for tid in spec.image_family_token_ids:
        assert int((batch["labels"] == tid).sum().item()) == 0, (
            f"label masking left id={tid} in labels"
        )
    # Pad is also masked.
    pad_id = smolvlm_processor.tokenizer.pad_token_id
    if pad_id is not None:
        assert int((batch["labels"] == pad_id).sum().item()) == 0


def test_collator_raises_on_missing_columns(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    with pytest.raises(KeyError, match="encode_streaming_multimodal"):
        collator.torch_call([{"input_ids": [1, 2, 3]}])  # no _mm_text / images


# ---- security gates -------------------------------------------------------


def test_collator_rejects_path_traversal_with_base_dir(
    smolvlm_processor, two_tiny_images, tmp_path
):
    spec = build_image_token_spec(smolvlm_processor)
    base = tmp_path / "images"
    base.mkdir()
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        image_base_dir=str(base),
    )
    # Absolute path rejection
    with pytest.raises(RuntimeError) as exc:
        collator._load_images_for_row([str(two_tiny_images[0])], row_index=0)
    assert isinstance(exc.value.__cause__, ValueError)
    assert "Absolute image path" in str(exc.value.__cause__)
    # Containment-escape rejection
    with pytest.raises(RuntimeError) as exc:
        collator._load_images_for_row(["../../../etc/passwd"], row_index=0)
    assert isinstance(exc.value.__cause__, ValueError)
    assert "outside" in str(exc.value.__cause__)


def test_collator_rejects_remote_urls(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    for url in (
        "http://example.com/a.png",
        "https://x/y.jpg",
        "file:///etc/passwd",
        "ftp://x/y.png",
        "data:image/png;base64,xxx",
        # Case-variant bypass attempts.
        "HTTP://evil.com/x.png",
        "Https://x/y.jpg",
        "FILE:///etc/passwd",
        "DATA:image/png;base64,xxx",
    ):
        with pytest.raises(RuntimeError) as exc:
            collator._load_images_for_row([url], row_index=0)
        assert isinstance(exc.value.__cause__, ValueError)
        assert "Non-local image path scheme" in str(exc.value.__cause__)


def test_collator_rejects_nul_byte_paths(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    with pytest.raises(RuntimeError) as exc:
        collator._load_images_for_row(["bad\x00path.png"], row_index=0)
    assert "NUL byte" in str(exc.value.__cause__)


def test_collator_rejects_non_string_image_entries(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    rows = [
        {
            "_mm_text": f"{spec.image_token}\nrow",
            "images": [None],  # type: ignore[list-item]
        }
    ]
    with pytest.raises(TypeError, match="path must be str"):
        collator.torch_call(rows)


def test_collator_rejects_bytes_mm_text(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    rows = [
        {
            "_mm_text": f"{spec.image_token}\nrow".encode(),
            "images": [str(two_tiny_images[0])],
        }
    ]
    with pytest.raises(TypeError, match="`_mm_text` must be str"):
        collator.torch_call(rows)


def test_collator_sanitizes_error_message(smolvlm_processor, tmp_path):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    missing = tmp_path / "subdir_with_secret_name" / "nope.png"
    with pytest.raises(RuntimeError) as exc:
        collator._load_images_for_row([str(missing)], row_index=3)
    # basename appears, full directory path does NOT
    assert "nope.png" in str(exc.value)
    assert "subdir_with_secret_name" not in str(exc.value)
    assert "Row 3" in str(exc.value)


def test_collator_rejects_too_many_images(smolvlm_processor, two_tiny_images):
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        max_images_per_row=2,
    )
    paths = [str(two_tiny_images[0])] * 3
    with pytest.raises(ValueError, match="max_images_per_row"):
        collator._load_images_for_row(paths, row_index=0)


# ---- mixed / all-text batches --------------------------------------------


def test_collator_all_text_batch_uses_tokenizer_fallback(smolvlm_processor):
    """A batch where every row has images=[] tokenizes via the tokenizer; no pixel_values."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    rows = [
        {"_mm_text": "first text-only row", "images": []},
        {"_mm_text": "second text-only row, slightly longer", "images": []},
    ]
    batch = collator.torch_call(rows)
    for k in ("input_ids", "attention_mask", "labels"):
        assert k in batch, f"missing batch key {k}"
    assert "pixel_values" not in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    pad_id = smolvlm_processor.tokenizer.pad_token_id
    if pad_id is not None:
        assert int((batch["labels"] == pad_id).sum().item()) == 0


def test_collator_mixed_batch_still_succeeds(smolvlm_processor, two_tiny_images):
    """A batch with one imaged row and one text-only row still produces pixel_values."""
    spec = build_image_token_spec(smolvlm_processor)
    encoded = encode_streaming_multimodal(
        {
            "text": [
                f"{spec.image_token}\nimaged row",
                "text-only row",
            ],
            "images": [[str(two_tiny_images[0])], []],
        },
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=2048,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    rows = [
        {
            k: encoded[k][i]
            for k in ("input_ids", "labels", "attention_mask", "images", "_mm_text")
        }
        for i in range(2)
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    batch = collator.torch_call(rows)
    for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
        assert k in batch, f"missing batch key {k}"
