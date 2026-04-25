"""Multimodal CPT eval-path tests."""

from __future__ import annotations

import pytest

from axolotl.utils.data.sft import (
    _create_placeholder_dataset,
    _prepare_streaming_dataset,
)
from axolotl.utils.dict import DictDefault

# ---- placeholder dataset for dispatch_batches ----------------------------


def test_placeholder_text_only_keeps_existing_shape():
    """Without an MM config, the placeholder is a single-column text dataset."""
    ds = _create_placeholder_dataset()
    row = next(iter(ds))
    assert "text" in row
    assert "images" not in row


def test_placeholder_mm_emits_image_column():
    """MM placeholder rows carry the configured image column as an empty list."""
    pt_cfg = DictDefault(
        {
            "type": "multimodal_pretrain",
            "text_column": "text",
            "image_column": "images",
            "multimodal": True,
        }
    )
    ds = _create_placeholder_dataset(pt_cfg)
    row = next(iter(ds))
    assert "text" in row
    assert "images" in row
    assert row["images"] == []


def test_placeholder_mm_honors_custom_columns():
    """Custom text_column / image_column on the MM config are reflected in the placeholder row."""
    pt_cfg = DictDefault(
        {
            "type": "multimodal_pretrain",
            "text_column": "doc",
            "image_column": "imgs",
        }
    )
    ds = _create_placeholder_dataset(pt_cfg)
    row = next(iter(ds))
    assert "doc" in row
    assert "imgs" in row
    assert row["imgs"] == []


# ---- multiple MM eval datasets are loaded --------------------------------


def test_mm_eval_iterates_all_test_datasets(monkeypatch):
    """All MM entries in test_datasets are loaded and concatenated into the eval stream."""
    cfg = DictDefault(
        {
            "streaming": True,
            "pretraining_dataset": [
                {"path": "train/ds", "type": "multimodal_pretrain"}
            ],
            "test_datasets": [
                {"path": "eval/a", "type": "multimodal_pretrain"},
                {"path": "eval/b", "type": "multimodal_pretrain"},
                {"path": "eval/c", "type": "multimodal_pretrain"},
            ],
            "max_steps": 10,
        }
    )

    seen_eval_paths: list[str] = []

    def fake_load_streaming(pretraining_config, *_a, **_kw):
        path = pretraining_config["path"]
        if path.startswith("eval/"):
            seen_eval_paths.append(path)
        return f"<stream:{path}>"

    def fake_concat(streams):
        return tuple(streams)

    monkeypatch.setattr(
        "axolotl.utils.data.sft._load_streaming_dataset", fake_load_streaming
    )
    monkeypatch.setattr("axolotl.utils.data.sft.concatenate_datasets", fake_concat)

    train, eval_ds, _, _ = _prepare_streaming_dataset(
        cfg, tokenizer=None, processor=None
    )

    assert seen_eval_paths == ["eval/a", "eval/b", "eval/c"]
    assert eval_ds == ("<stream:eval/a>", "<stream:eval/b>", "<stream:eval/c>")


def test_mm_eval_rejects_mixed_mm_and_non_mm_test_datasets(monkeypatch):
    """MM CPT runs require every test_datasets entry to be MM; mixed lists raise."""
    cfg = DictDefault(
        {
            "streaming": True,
            "pretraining_dataset": [
                {"path": "train/ds", "type": "multimodal_pretrain"}
            ],
            "test_datasets": [
                {"path": "eval/a", "type": "multimodal_pretrain"},
                # Plain text eval entry — not allowed alongside MM eval.
                {"path": "eval/b", "type": "pretrain"},
            ],
            "max_steps": 10,
        }
    )
    monkeypatch.setattr(
        "axolotl.utils.data.sft._load_streaming_dataset",
        lambda *_a, **_kw: "<stream>",
    )
    with pytest.raises(ValueError, match="multimodal and non-multimodal"):
        _prepare_streaming_dataset(cfg, tokenizer=None, processor=None)


# ---- eval collator pulls image settings from test_datasets ---------------


def test_eval_collator_uses_eval_image_settings(monkeypatch):
    """Eval collator pulls image_base_dir / image_token from test_datasets[0]; train collator from pretraining_dataset[0]."""
    from axolotl.core.builders.causal import HFCausalTrainerBuilder

    captured = {}

    class _FakeSpec:
        image_token = "<img>"
        image_token_id = 7
        image_family_token_ids = (7,)

    def fake_build_image_token_spec(processor, override=None):
        captured["override"] = override
        return _FakeSpec()

    monkeypatch.setattr(
        "axolotl.prompt_strategies.multimodal_pretrain.build_image_token_spec",
        fake_build_image_token_spec,
    )

    class _FakeCollator:
        def __init__(self, **kw):
            captured["kwargs"] = kw

    monkeypatch.setattr(
        "axolotl.core.builders.causal.MultiModalPretrainDataCollator", _FakeCollator
    )

    builder = HFCausalTrainerBuilder.__new__(HFCausalTrainerBuilder)
    builder.tokenizer = object()
    builder.processor = object()
    builder.cfg = DictDefault(
        {
            "pretraining_dataset": [
                {
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/train_images",
                    "image_token": "<train_img>",
                }
            ],
            "test_datasets": [
                {
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/eval_images",
                    "image_token": "<eval_img>",
                }
            ],
            "sequence_len": 2048,
        }
    )

    builder._build_mm_pretrain_collator(is_eval=True)
    assert captured["override"] == "<eval_img>"
    assert captured["kwargs"]["image_base_dir"] == "/eval_images"

    captured.clear()
    builder._build_mm_pretrain_collator(is_eval=False)
    assert captured["override"] == "<train_img>"
    assert captured["kwargs"]["image_base_dir"] == "/train_images"
