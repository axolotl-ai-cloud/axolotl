"""Multimodal CPT eval-path tests."""

from __future__ import annotations

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


def test_pretraining_config_from_entry_drops_trust_remote_code():
    """trust_remote_code is dead in datasets>=4.x — it must not be forwarded."""
    from axolotl.utils.data.sft import _pretraining_config_from_entry

    cfg = _pretraining_config_from_entry(
        {"path": "ds", "type": "multimodal_pretrain", "trust_remote_code": True}
    )
    assert "trust_remote_code" not in cfg


def test_pretraining_config_from_entry_preserves_max_images_per_row():
    """max_images_per_row on the dataset entry survives normalization (default 32)."""
    from axolotl.utils.data.sft import _pretraining_config_from_entry

    cfg = _pretraining_config_from_entry(
        {"path": "ds", "type": "multimodal_pretrain", "max_images_per_row": 8}
    )
    assert cfg["max_images_per_row"] == 8

    cfg = _pretraining_config_from_entry({"path": "ds", "type": "multimodal_pretrain"})
    assert cfg["max_images_per_row"] == 32


def test_pretraining_config_from_entry_preserves_ds_type():
    """ds_type on the dataset entry survives normalization."""
    from axolotl.utils.data.sft import _pretraining_config_from_entry

    cfg = _pretraining_config_from_entry(
        {"path": "/data/*.jsonl", "type": "multimodal_pretrain", "ds_type": "json"}
    )
    assert cfg["ds_type"] == "json"

    cfg = _pretraining_config_from_entry({"path": "ds", "type": "multimodal_pretrain"})
    assert cfg["ds_type"] is None


def _capture_load_dataset(monkeypatch):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        class _Stub:
            def skip(self, *_a, **_kw):
                return self

        return _Stub()

    class _StubFormat:
        def with_format(self, *_a, **_kw):
            return self

    monkeypatch.setattr("axolotl.utils.data.sft.load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        "axolotl.utils.data.sft.wrap_streaming_dataset",
        lambda *a, **kw: _StubFormat(),
    )
    return captured


def _streaming_pretraining_config(**overrides):
    base = {
        "path": None,
        "name": None,
        "skip": 0,
        "split": "train",
        "data_files": None,
        "ds_type": None,
        "type": "multimodal_pretrain",
        "text_column": "text",
        "multimodal": True,
        "image_column": "images",
        "image_base_dir": None,
        "image_token": None,
    }
    base.update(overrides)
    return DictDefault(base)


def test_load_streaming_dataset_routes_ds_type_to_loader(monkeypatch, tmp_path):
    """ds_type + a local glob path routes to the packaged loader with path as
    data_files."""
    from axolotl.utils.data.sft import _load_streaming_dataset

    captured = _capture_load_dataset(monkeypatch)
    shard = tmp_path / "shard-0000.jsonl"
    shard.write_text('{"text": "x", "images": []}\n')
    glob_path = str(tmp_path / "*.jsonl")

    pretraining_config = _streaming_pretraining_config(path=glob_path, ds_type="json")
    cfg = DictDefault({"sequence_len": 2048, "accelerator_config": None})

    _load_streaming_dataset(pretraining_config, cfg, tokenizer=None, processor=None)

    assert captured["args"] == ("json",)
    assert captured["kwargs"]["data_files"] == glob_path
    assert captured["kwargs"]["split"] == "train"


def test_load_streaming_dataset_hub_path_keeps_path_first_load(monkeypatch):
    """A Hub repo path with ds_type + data_files keeps the historical path-first
    load — ds_type must not redirect it to the local-file loader."""
    from axolotl.utils.data.sft import _load_streaming_dataset

    captured = _capture_load_dataset(monkeypatch)
    pretraining_config = _streaming_pretraining_config(
        path="someorg/some-hub-repo",
        ds_type="json",
        data_files=["data/train-00000.jsonl"],
    )
    cfg = DictDefault({"sequence_len": 2048, "accelerator_config": None})

    _load_streaming_dataset(pretraining_config, cfg, tokenizer=None, processor=None)

    assert captured["args"] == ("someorg/some-hub-repo",)
    assert captured["kwargs"]["data_files"] == ["data/train-00000.jsonl"]


def test_load_streaming_dataset_never_forwards_trust_remote_code(monkeypatch, tmp_path):
    """datasets>=4.x logs an alarming error on trust_remote_code — never forward it."""
    from axolotl.utils.data.sft import _load_streaming_dataset

    captured = _capture_load_dataset(monkeypatch)
    pretraining_config = _streaming_pretraining_config(path="someorg/some-hub-repo")
    cfg = DictDefault({"sequence_len": 2048, "accelerator_config": None})

    _load_streaming_dataset(pretraining_config, cfg, tokenizer=None, processor=None)

    assert "trust_remote_code" not in captured["kwargs"]


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

    _train, eval_ds, _, _ = _prepare_streaming_dataset(
        cfg, tokenizer=None, processor=None
    )

    assert seen_eval_paths == ["eval/a", "eval/b", "eval/c"]
    assert eval_ds == ("<stream:eval/a>", "<stream:eval/b>", "<stream:eval/c>")


# Mixed MM / non-MM test_datasets is rejected at config-load time by
# check_multimodal_cpt (see tests/utils/schemas/validation/test_multimodal_cpt.py).


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


def test_collator_forwards_skip_and_remote_flags(monkeypatch):
    """`skip_bad_images` / `allow_remote_images` on the dataset entry reach the collator."""
    from axolotl.core.builders.causal import HFCausalTrainerBuilder

    captured = {}

    class _FakeSpec:
        image_token = "<img>"
        image_token_id = 7
        image_family_token_ids = (7,)

    monkeypatch.setattr(
        "axolotl.prompt_strategies.multimodal_pretrain.build_image_token_spec",
        lambda processor, override=None: _FakeSpec(),
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
                    "skip_bad_images": True,
                    "allow_remote_images": True,
                }
            ],
            "sequence_len": 2048,
        }
    )
    builder._build_mm_pretrain_collator(is_eval=False)
    assert captured["kwargs"]["skip_bad_images"] is True
    assert captured["kwargs"]["allow_remote_images"] is True

    # Unset -> secure defaults (False), never None.
    captured.clear()
    builder.cfg = DictDefault(
        {
            "pretraining_dataset": [{"type": "multimodal_pretrain"}],
            "sequence_len": 2048,
        }
    )
    builder._build_mm_pretrain_collator(is_eval=False)
    assert captured["kwargs"]["skip_bad_images"] is False
    assert captured["kwargs"]["allow_remote_images"] is False


def test_eval_collator_honors_eval_sequence_len(monkeypatch):
    """Eval collator uses cfg.eval_sequence_len when set; train collator uses cfg.sequence_len."""
    from axolotl.core.builders.causal import HFCausalTrainerBuilder

    captured = {}

    class _FakeSpec:
        image_token = "<img>"
        image_token_id = 7
        image_family_token_ids = (7,)

    monkeypatch.setattr(
        "axolotl.prompt_strategies.multimodal_pretrain.build_image_token_spec",
        lambda processor, override=None: _FakeSpec(),
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
            "pretraining_dataset": [{"type": "multimodal_pretrain"}],
            "test_datasets": [{"type": "multimodal_pretrain"}],
            "sequence_len": 4096,
            "eval_sequence_len": 1024,
        }
    )

    builder._build_mm_pretrain_collator(is_eval=True)
    assert captured["kwargs"]["max_length"] == 1024

    captured.clear()
    builder._build_mm_pretrain_collator(is_eval=False)
    assert captured["kwargs"]["max_length"] == 4096

    # eval_sequence_len unset -> eval falls back to sequence_len
    builder.cfg = DictDefault(
        {
            "pretraining_dataset": [{"type": "multimodal_pretrain"}],
            "test_datasets": [{"type": "multimodal_pretrain"}],
            "sequence_len": 4096,
        }
    )
    captured.clear()
    builder._build_mm_pretrain_collator(is_eval=True)
    assert captured["kwargs"]["max_length"] == 4096
