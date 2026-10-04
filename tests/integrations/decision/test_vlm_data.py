"""VLM conditioning contracts for prepared decision data."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from axolotl.integrations.decision import datasets
from axolotl.integrations.decision.grouping import group_records
from axolotl.integrations.decision.prepared_cache import _image_identity
from axolotl.integrations.decision.preprocessing import build_decision_canvas
from axolotl.integrations.decision.records import DecisionCanvas
from axolotl.integrations.decision.row_codec import row_from_arrow, row_to_arrow


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def record(*, images=None):
    value = {
        "id": "one",
        "source": "synthetic",
        "group": "group",
        "state": {"weather": "clear"},
        "questions": {"q": {"type": "choice", "options": ["no", "yes"]}},
        "labels": {"q": {"kind": "hard", "gold_idx": 1}},
    }
    if images is not None:
        value["images"] = images
    return value


def _processor(monkeypatch, marker_ids=(18,)):
    module = types.ModuleType("axolotl.model_support.nemotron_diffusion.processing")

    def prepare(*args, **kwargs):
        assert kwargs["model_source"] == "example/vlm"
        return {
            "input_ids": [4, *marker_ids, 5],
            "pixel_values": [[[[0.25, 0.5], [0.75, 1.0]]]],
            "image_sizes": [[2, 2]],
            "image_token_ids": marker_ids,
        }

    module.prepare_decision_prompt = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _canvas(**kwargs):
    model_type = kwargs.pop("model_type", "nemotron_labs_diffusion_vlm")
    return build_decision_canvas(
        Tokenizer(),
        record(images=["image.png"]),
        scaffold_ids=(),
        turn_close_id=4,
        pad_id=0,
        vocab_size=256,
        model_source="example/vlm",
        model_type=model_type,
        **kwargs,
    )


def test_vlm_canvas_expands_prompt_and_preserves_typed_media(monkeypatch):
    _processor(monkeypatch)
    canvas = _canvas()

    assert canvas.prompt_ids == (4, 18, 5)
    assert canvas.model_inputs == {
        "pixel_values": [[[[0.25, 0.5], [0.75, 1.0]]]],
        "image_sizes": [[2, 2]],
    }


def test_vlm_marker_cannot_be_an_answer_token(monkeypatch):
    _processor(monkeypatch)
    canvas = _canvas()
    _processor(monkeypatch, marker_ids=(canvas.allowed_ids[0][0],))

    with pytest.raises(ValueError, match="image marker IDs"):
        _canvas()


def test_arrow_roundtrip_keeps_media_typed_not_json():
    canvas = DecisionCanvas(
        prompt_ids=(1, 2),
        canvas_ids=(3, 4),
        label_positions=(0,),
        allowed_ids=((5, 6),),
        question_ids=("q",),
        targets=({"kind": "hard", "gold_idx": 0},),
        pinned_mask=(False, False),
        semantic_mask=(True, True),
        slot_mask=(False, False),
        template_length=1,
        model_inputs={
            "pixel_values": [[[[0.25, 0.5], [0.75, 1.0]]]],
            "image_sizes": [[2, 2]],
        },
    )
    encoded = row_to_arrow({"canvas": canvas, "record": record(images=["a.png"])})

    assert "pixel_values" not in encoded["extras_json"]
    assert encoded["model_input_image_sizes"] == [[2, 2]]
    assert row_from_arrow(encoded)["canvas"].model_inputs == canvas.model_inputs


def test_grouping_keeps_different_image_contexts_separate():
    grouped = group_records(
        [record(images=["first.png"]), {**record(images=["second.png"]), "id": "two"}],
        max_questions=4,
    )

    assert [value["id"] for value in grouped] == ["one", "two"]


def test_image_cache_identity_hashes_local_bytes_and_bypasses_remote(tmp_path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"first")
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"images": [str(image)]})
        + "\n"
        + json.dumps({"state": "text only"})
        + "\n"
    )

    first = _image_identity([source])
    image.write_bytes(b"second")

    assert first is not None
    assert first != _image_identity([source])
    source.write_text('{"images": ["https://example.test/image.png"]}\n')
    assert _image_identity([source]) is None


def test_images_cannot_use_an_unexpanded_explicit_prompt(monkeypatch):
    _processor(monkeypatch)
    with pytest.raises(ValueError, match="processor-expanded"):
        _canvas(prompt_ids=(1, 2))


def test_images_require_the_nemotron_vlm_model_type(monkeypatch):
    _processor(monkeypatch)
    with pytest.raises(ValueError, match="nemotron_labs_diffusion_vlm"):
        _canvas(model_type="nemotron_labs_diffusion")


def test_canvas_forwards_image_resolution_settings(monkeypatch):
    _processor(monkeypatch)
    module = sys.modules["axolotl.model_support.nemotron_diffusion.processing"]
    prepare = module.prepare_decision_prompt
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_decision_prompt", capture)
    _canvas(processor_kwargs={"max_image_size": 560})
    assert seen["max_image_size"] == 560


def test_processor_cache_identity_tracks_resolution_and_local_source(tmp_path):
    from axolotl.integrations.decision.prepared_cache import _processor_identity

    source = tmp_path / "image_processing.py"
    source.write_text("version = 1")
    cfg = {
        "base_model": str(tmp_path),
        "revision_of_model": "main",
        "model_config": {"_commit_hash": "resolved-commit"},
        "processor_kwargs": {"max_image_size": 560},
    }
    first = _processor_identity(cfg)
    assert first["model_revision"] == "resolved-commit"
    cfg["processor_kwargs"] = {"max_image_size": 280}
    assert _processor_identity(cfg) != first
    cfg["processor_kwargs"] = {"max_image_size": 560}
    source.write_text("version = 2")
    assert _processor_identity(cfg) != first


def test_dataset_uses_canonical_model_type_before_model_config_is_available(
    monkeypatch,
):
    captured = {}

    def build(_tokenizer, _record, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(prompt_ids=(1,), canvas_ids=(100,) * 128)

    monkeypatch.setattr(datasets, "build_decision_canvas", build)
    monkeypatch.setattr(datasets, "permute_record", lambda value, **kwargs: value)
    monkeypatch.setattr(
        datasets, "decision_example_from_canvas", lambda *args, **kwargs: None
    )
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = 1
    tokenizer.pad_token_id = 1
    datasets._canvas_row(
        tokenizer,
        {"source": "synthetic"},
        {
            "base_model": "example/vlm",
            "model_config": {"vocab_size": 256},
            "model_config_type": "nemotron_labs_diffusion_vlm",
            "diffusion": {"mask_token_id": 100},
            "decision": {},
        },
        1.0,
        spec=SimpleNamespace(max_canvas=128, noise=datasets.DiffusionNoise.ABSORBING),
    )
    assert captured["model_type"] == "nemotron_labs_diffusion_vlm"


def test_prepared_cache_uses_canonical_model_type_for_image_identity():
    from axolotl.integrations.decision.prepared_cache import _uses_vlm

    assert _uses_vlm(
        {"model_config": {}, "model_config_type": "nemotron_labs_diffusion_vlm"}
    )
