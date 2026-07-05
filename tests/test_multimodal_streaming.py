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
    # The last placeholder must remain countable even when it's hundreds of
    # tokens deep — guards against a regression that adds tokenizer
    # truncation and silently drops trailing placeholders.
    spec = build_image_token_spec(smolvlm_processor)
    long_filler = "lorem ipsum " * 400
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
    ids = out["input_ids"][0]
    # Sanity: the input is genuinely long, so a truncating regression would
    # have to cut into it to drop the last placeholder.
    assert len(ids) > 2000
    assert sum(1 for t in ids if t == spec.image_token_id) == 3


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


# ---- build_image_token_spec autodetection --------------------------------


class _StubTokenizer:
    """Minimal tokenizer stub for autodetection tests."""

    def __init__(self, vocab: dict[str, int], unk_id: int = 0):
        self._vocab = vocab
        self.unk_token_id = unk_id
        self.all_special_tokens = list(vocab.keys())
        self.additional_special_tokens: list[str] = []

    def get_added_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, self.unk_token_id)


class _StubProcessor:
    def __init__(self, tokenizer, image_token=None, boi_token=None):
        self.tokenizer = tokenizer
        if image_token is not None:
            self.image_token = image_token
        if boi_token is not None:
            self.boi_token = boi_token


def test_build_image_token_spec_gemma4_uses_image_token_not_boi():
    """Gemma-4: `image_token` is the user-facing placeholder; don't swap to boi_token."""
    tok = _StubTokenizer({"<|image|>": 258880, "<|image>": 255999})
    proc = _StubProcessor(tok, image_token="<|image|>", boi_token="<|image>")
    spec = build_image_token_spec(proc)
    assert spec.image_token == "<|image|>"
    assert spec.image_token_id == 258880


def test_build_image_token_spec_gemma3_swaps_to_boi_token():
    """Gemma-3: `image_token` is the post-expansion soft token; placeholder is `boi_token`."""
    tok = _StubTokenizer({"<image_soft_token>": 262144, "<start_of_image>": 255999})
    proc = _StubProcessor(
        tok, image_token="<image_soft_token>", boi_token="<start_of_image>"
    )
    spec = build_image_token_spec(proc)
    assert spec.image_token == "<start_of_image>"
    assert spec.image_token_id == 255999


def test_build_image_token_spec_override_not_special_rejected():
    """Override that isn't a registered special token is rejected (would BPE-tokenize)."""
    tok = _StubTokenizer({"<|image|>": 258880})
    proc = _StubProcessor(tok, image_token="<|image|>")
    with pytest.raises(ValueError, match="not a registered special token"):
        build_image_token_spec(proc, override="not_a_real_token")


def test_build_image_token_spec_override_resolves_to_unk_rejected():
    """Override that resolves to unk is rejected with a clear error."""
    tok = _StubTokenizer({"<|image|>": 258880, "<|fake|>": 0}, unk_id=0)
    proc = _StubProcessor(tok, image_token="<|image|>")
    with pytest.raises(ValueError, match="did not resolve"):
        build_image_token_spec(proc, override="<|fake|>")


def test_build_image_token_spec_no_candidates_raises():
    """If neither processor attrs nor any known candidate resolve, raise a clear error."""
    tok = _StubTokenizer({})  # nothing registered
    proc = _StubProcessor(tok)  # no image_token, no boi_token
    with pytest.raises(ValueError, match="Could not autodetect"):
        build_image_token_spec(proc)


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


def test_wrap_streaming_dataset_eval_honors_eval_sequence_len(
    smolvlm_processor, monkeypatch
):
    """is_eval=True with cfg.eval_sequence_len set caps encoder at eval_sequence_len."""
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
                {"path": "train/ds", "type": "multimodal_pretrain"}
            ],
            "sequence_len": 4096,
            "eval_sequence_len": 1024,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 1000,
            "seed": 42,
        }
    )

    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=DictDefault(
            {"path": "test/ds", "type": "multimodal_pretrain"}
        ),
        is_eval=True,
    )
    assert captured["kwargs"]["max_tokens"] == 1024

    captured.clear()
    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=DictDefault(
            {"path": "train/ds", "type": "multimodal_pretrain"}
        ),
        is_eval=False,
    )
    assert captured["kwargs"]["max_tokens"] == 4096

    # eval_sequence_len unset -> eval falls back to sequence_len.
    captured.clear()
    cfg_no_eval = DictDefault(
        {
            "sample_packing": False,
            "pretraining_dataset": [
                {"path": "train/ds", "type": "multimodal_pretrain"}
            ],
            "sequence_len": 4096,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 1000,
            "seed": 42,
        }
    )
    wrap_streaming_dataset(
        _Dataset(),
        smolvlm_processor.tokenizer,
        cfg_no_eval,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=DictDefault(
            {"path": "test/ds", "type": "multimodal_pretrain"}
        ),
        is_eval=True,
    )
    assert captured["kwargs"]["max_tokens"] == 4096


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


# ---- input validation -----------------------------------------------------


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


def test_collator_skip_bad_images_drops_row_and_continues(
    smolvlm_processor, two_tiny_images, tmp_path
):
    """skip_bad_images=True: bad row drops, batch survives on remaining rows."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        skip_bad_images=True,
    )
    rows = [
        {
            "_mm_text": f"{spec.image_token}\ngood row",
            "images": [str(two_tiny_images[0])],
        },
        {
            "_mm_text": f"{spec.image_token}\nbad row",
            "images": [str(tmp_path / "missing.png")],
        },
    ]
    batch = collator.torch_call(rows)
    # Surviving row produced a batch with pixel_values from the good image.
    assert "input_ids" in batch and "pixel_values" in batch
    assert batch["input_ids"].shape[0] == 1


def test_collator_all_rows_dropped_raises(smolvlm_processor, tmp_path):
    """skip_bad_images=True with every row failing surfaces a RuntimeError."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        skip_bad_images=True,
    )
    rows = [
        {
            "_mm_text": f"{spec.image_token}\nrow",
            "images": [str(tmp_path / f"missing_{i}.png")],
        }
        for i in range(2)
    ]
    with pytest.raises(RuntimeError, match="All rows in the batch were dropped"):
        collator.torch_call(rows)


# ---- mixed / all-text batches --------------------------------------------


def test_collator_warns_when_tokenizer_diverges_from_processor_tokenizer(
    smolvlm_processor, caplog, monkeypatch
):
    """Construct-time warning when self.tokenizer is not processor.tokenizer."""
    import logging as _logging

    # `axolotl` logger has propagate=False (logging_config.py); flip it so
    # caplog's root handler receives the record.
    monkeypatch.setattr(_logging.getLogger("axolotl"), "propagate", True)
    spec = build_image_token_spec(smolvlm_processor)

    # Same tokenizer: no warning.
    with caplog.at_level(
        _logging.WARNING, logger="axolotl.utils.collators.mm_pretrain"
    ):
        MultiModalPretrainDataCollator(
            tokenizer=smolvlm_processor.tokenizer,
            processor=smolvlm_processor,
            image_token_spec=spec,
        )
    assert not any("tokenize inconsistently" in r.getMessage() for r in caplog.records)

    caplog.clear()

    # Different tokenizer instance (a stand-in object): warning fires.
    class _OtherTokenizer:
        pad_token_id = None

    with caplog.at_level(
        _logging.WARNING, logger="axolotl.utils.collators.mm_pretrain"
    ):
        MultiModalPretrainDataCollator(
            tokenizer=_OtherTokenizer(),
            processor=smolvlm_processor,
            image_token_spec=spec,
        )
    assert any("tokenize inconsistently" in r.getMessage() for r in caplog.records)


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


# ---- padding / EOS label masking ------------------------------------------


def test_build_labels_masks_padding_structurally_not_by_pad_id(smolvlm_processor):
    """Padding is masked via attention_mask, not by pad-token-id value.

    Regression for the pad_token_id == eos_token_id case (and for a processor
    that pads with a different id than self.tokenizer): a real, attended EOS
    that shares the pad id must survive as a label; only the unattended pad
    positions are masked.
    """
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    eos_id = smolvlm_processor.tokenizer.eos_token_id
    # index 2 is the real end-of-sequence token (attended); indices 3-4 are
    # padding that happens to reuse the same id (pad == eos).
    input_ids = torch.tensor([[10, 11, eos_id, eos_id, eos_id]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    labels = collator._build_labels(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    assert labels[0, 2].item() == eos_id, "real EOS was masked out of labels"
    assert labels[0, 3].item() == -100
    assert labels[0, 4].item() == -100


def test_collator_all_text_preserves_real_eos_when_pad_equals_eos(
    smolvlm_processor, monkeypatch
):
    """End-to-end all-text batch with pad_token == eos_token keeps the real EOS.

    Axolotl falls back to `pad_token = eos_token` for tokenizers without a pad
    token (see loaders/tokenizer.py). Masking padding by value would delete the
    trailing EOS the collator appends, leaving CPT with no stop supervision.
    """
    tok = smolvlm_processor.tokenizer
    monkeypatch.setattr(tok, "pad_token", tok.eos_token)
    assert tok.pad_token_id == tok.eos_token_id
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=tok, processor=smolvlm_processor, image_token_spec=spec
    )
    rows = [
        {"_mm_text": "short", "images": []},
        {
            "_mm_text": "a considerably longer row that forces the first to pad",
            "images": [],
        },
    ]
    batch = collator.torch_call(rows)
    labels, attn, ids = batch["labels"], batch["attention_mask"], batch["input_ids"]
    eos_id = tok.eos_token_id
    # Every unattended (padding) position is masked...
    assert int((labels[attn == 0] != -100).sum().item()) == 0
    # ...and the real, attended EOS survives as a training target.
    real_eos = (attn == 1) & (ids == eos_id)
    assert int(real_eos.sum().item()) >= 1, "expected a real EOS token in the batch"
    assert int((labels[real_eos] != eos_id).sum().item()) == 0, (
        "real EOS was masked out of labels (pad==eos value-masking regression)"
    )


# ---- encoder EOS / bad-row handling ---------------------------------------


class _FakeEncTok:
    """Minimal tokenizer stub for encoder EOS-handling tests."""

    def __init__(self, eos_id, already_appends=False):
        self.eos_token_id = eos_id
        self._already = already_appends

    def __call__(self, text, add_special_tokens=True):
        ids = [5, 6, 7]
        if self._already and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_encode_no_none_when_tokenizer_lacks_eos():
    """A tokenizer with eos_token_id=None must not append a None id (L1)."""
    out = encode_streaming_multimodal(
        {"text": ["hello"], "images": [[]]},
        tokenizer=_FakeEncTok(eos_id=None),
        max_tokens=64,
        image_token="<image>",
        image_token_id=999,
    )
    ids = out["input_ids"][0]
    assert None not in ids
    assert ids == [5, 6, 7]
    assert len(out["attention_mask"][0]) == len(ids)


def test_encode_no_double_eos_when_tokenizer_appends_eos():
    """When the tokenizer already appends EOS, the encoder must not add another (M2)."""
    out = encode_streaming_multimodal(
        {"text": ["hello"], "images": [[]]},
        tokenizer=_FakeEncTok(eos_id=9, already_appends=True),
        max_tokens=64,
        image_token="<image>",
        image_token_id=999,
    )
    ids = out["input_ids"][0]
    assert ids == [5, 6, 7, 9]
    assert ids.count(9) == 1


def test_encode_skip_bad_rows_drops_mismatch_and_oversize(smolvlm_processor):
    """skip_bad_rows warns-and-drops malformed rows instead of aborting the map (#4)."""
    spec = build_image_token_spec(smolvlm_processor)
    good = f"{spec.image_token}\ngood"
    examples = {
        "text": [
            good,
            f"{spec.image_token}{spec.image_token}\ntwo placeholders one image",
            "word " * 5000,
        ],
        "images": [["a.png"], ["b.png"], []],
    }
    out = encode_streaming_multimodal(
        examples,
        tokenizer=smolvlm_processor.tokenizer,
        max_tokens=128,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        skip_bad_rows=True,
    )
    assert out["_mm_text"] == [good]
    assert out["images"] == [["a.png"]]


def test_encode_bad_row_still_raises_by_default(smolvlm_processor):
    """Default (skip_bad_rows=False) keeps the loud-failure contract."""
    spec = build_image_token_spec(smolvlm_processor)
    examples = {
        "text": [f"{spec.image_token}{spec.image_token}\nmismatch"],
        "images": [["a.png"]],
    }
    with pytest.raises(ValueError, match="occurrence"):
        encode_streaming_multimodal(
            examples,
            tokenizer=smolvlm_processor.tokenizer,
            max_tokens=128,
            image_token=spec.image_token,
            image_token_id=spec.image_token_id,
        )


# ---- image-family masking of structural tokens (#2) -----------------------


def test_spec_family_includes_structural_image_tokens(smolvlm_processor):
    """SmolVLM tile/grid/global markers join the family set (were leaking)."""
    spec = build_image_token_spec(smolvlm_processor)
    tok = smolvlm_processor.tokenizer
    for surface in ("<fake_token_around_image>", "<global-img>", "<row_1_col_1>"):
        tid = tok.convert_tokens_to_ids(surface)
        assert tid is not None and tid != tok.unk_token_id, surface
        assert tid in spec.image_family_token_ids, f"{surface} not masked"
    # bos/eos are never swept into the family.
    assert tok.eos_token_id not in spec.image_family_token_ids


def test_collator_masks_structural_image_tokens(smolvlm_processor, two_tiny_images):
    """No structural image token survives as a valid label after collation."""
    spec = build_image_token_spec(smolvlm_processor)
    tok = smolvlm_processor.tokenizer
    struct_ids = [
        tok.convert_tokens_to_ids(t)
        for t in ("<fake_token_around_image>", "<global-img>", "<row_1_col_1>")
    ]
    encoded = encode_streaming_multimodal(
        {"text": [f"{spec.image_token}\nrow"], "images": [[str(two_tiny_images[0])]]},
        tokenizer=tok,
        max_tokens=4096,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
    )
    rows = [
        {
            k: encoded[k][0]
            for k in ("input_ids", "labels", "attention_mask", "images", "_mm_text")
        }
    ]
    collator = MultiModalPretrainDataCollator(
        tokenizer=tok, processor=smolvlm_processor, image_token_spec=spec
    )
    batch = collator.torch_call(rows)
    for tid in struct_ids:
        assert int((batch["labels"] == tid).sum().item()) == 0, (
            f"structural image id {tid} leaked into labels"
        )


# ---- remote-URL / path-traversal image rejection (#3 / M1) -----------------


def test_collator_rejects_remote_image_source(smolvlm_processor):
    """A URL-scheme image source is rejected (SSRF guard) — never fetched."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
    )
    with pytest.raises(RuntimeError, match="remote image source"):
        collator._load_images_for_row(
            ["http://169.254.169.254/latest/meta-data/"], row_index=0
        )


def test_collator_remote_skipped_with_skip_bad_images(smolvlm_processor):
    """skip_bad_images drops a remote source instead of raising."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        skip_bad_images=True,
    )
    assert collator._load_images_for_row(["https://evil.example/x.png"], 0) == []


def test_collator_allows_remote_when_opted_in(smolvlm_processor):
    """allow_remote_images=True stops the scheme short-circuit (fetch still up to load_image)."""
    spec = build_image_token_spec(smolvlm_processor)
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        allow_remote_images=True,
    )
    assert collator._reject_remote("http://example.com/x.png") is None


def test_collator_rejects_path_traversal(smolvlm_processor, tmp_path):
    """A relative path escaping image_base_dir is rejected with a clear message."""
    spec = build_image_token_spec(smolvlm_processor)
    base = tmp_path / "imgs"
    base.mkdir()
    collator = MultiModalPretrainDataCollator(
        tokenizer=smolvlm_processor.tokenizer,
        processor=smolvlm_processor,
        image_token_spec=spec,
        image_base_dir=str(base),
    )
    with pytest.raises(RuntimeError, match="escapes image_base_dir"):
        collator._load_images_for_row(["../../../../etc/passwd"], row_index=0)


def test_collator_double_eos_guard_respects_add_eos_token(
    smolvlm_processor, monkeypatch
):
    """With tokenizer.add_eos_token=True the collator must not append its own EOS (M2).

    SmolVLM honors add_eos_token, so the tokenizer already emits exactly one
    EOS. Without the guard the collator would append the EOS string on top,
    yielding a doubled `<eos><eos>` tail (count == 2).
    """
    spec = build_image_token_spec(smolvlm_processor)
    tok = smolvlm_processor.tokenizer
    monkeypatch.setattr(tok, "add_eos_token", True, raising=False)
    collator = MultiModalPretrainDataCollator(
        tokenizer=tok, processor=smolvlm_processor, image_token_spec=spec
    )
    batch = collator.torch_call([{"_mm_text": "no eos please", "images": []}])
    assert int((batch["input_ids"] == tok.eos_token_id).sum().item()) == 1
