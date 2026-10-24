"""Release fixes: read-time manifest check, honest read-protocol fields, read-overflow canvas width."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from axolotl.integrations.diffusion_decision import datasets
from axolotl.integrations.diffusion_decision.args import DiffusionDecisionConfig
from axolotl.integrations.diffusion_decision.manifest import (
    MANIFEST_FILENAME,
    DecisionManifest,
    NoiseReadProtocol,
    build_decision_manifest,
    decision_read_passes,
    manifest_mismatches,
    read_manifest_path,
    validate_read_manifest,
)
from axolotl.integrations.diffusion_decision.plugin import DiffusionDecisionPlugin
from axolotl.model_support import DiffusionLayout
from axolotl.utils.dict import DictDefault

from tests.integrations.diffusion_decision.helpers import (
    ChatCharacterTokenizer,
    make_cfg,
    make_record,
    make_spec,
)

MODEL = SimpleNamespace(config=SimpleNamespace(mask_token_id=100, _commit_hash="rev"))


def _cfg(**decision) -> DictDefault:
    labels = {"codebook": decision.pop("codebook", "expanded255")}
    k_max = decision.pop("k_max", 2)
    return DictDefault(
        {
            "base_model": "nvidia/Nemotron-Labs-Diffusion-8B",
            "model_config_type": "nemotron_labs_diffusion",
            "diffusion_lm": {
                "canvas_width": decision.pop("canvas_width", 128),
                "mask_token_id": 100,
                "unroll": {"k_max": k_max},
            },
            "diffusion_decision": {
                "max_questions_per_canvas": 10,
                "labels": labels,
                **decision,
            },
        }
    )


def _legacy_manifest_json(cfg) -> dict:
    """A manifest as written by feat/decision-codebook-255 (run 2h): no max_questions_per_canvas,
    canvas_prefix_ids or read_passes, and `diffusion_steps` holding num_diffusion_steps."""
    data = build_decision_manifest(cfg, model=MODEL).model_dump(mode="json")
    data.pop("canvas_prefix_ids")
    data.pop("max_questions_per_canvas")
    protocol = data["noise_read_protocol"]
    for key in (
        "read_passes",
        "unroll_k_max",
        "unroll_eval_steps",
        "generation_diffusion_steps",
    ):
        protocol.pop(key)
    protocol["diffusion_steps"] = 128
    return data


# ------------------------------------------------------------------ manifest fields


def test_manifest_records_read_passes_not_generation_steps():
    manifest = build_decision_manifest(_cfg(), model=MODEL)
    written = manifest.model_dump(mode="json")

    protocol = written["noise_read_protocol"]
    assert "diffusion_steps" not in protocol
    # Full-sequence layout, no free slots: the k_max=2 unroll re-reads an unchanged canvas.
    assert protocol["read_passes"] == 1
    assert protocol["unroll_k_max"] == 2
    assert protocol["unroll_eval_steps"] is None
    assert written["max_questions_per_canvas"] == 10
    assert written["canvas_prefix_ids"] == []
    assert written["label_codebook"] == "expanded255"


def test_legacy_diffusion_steps_parses_as_generation_steps():
    legacy = _legacy_manifest_json(_cfg())
    restored = DecisionManifest.model_validate_json(json.dumps(legacy))

    assert restored.noise_read_protocol.generation_diffusion_steps == 128
    assert restored.noise_read_protocol.read_passes is None
    assert restored.max_questions_per_canvas is None
    assert restored.canvas_prefix_ids == []
    dumped = restored.model_dump(mode="json")["noise_read_protocol"]
    assert "diffusion_steps" not in dumped


def test_unroll_eval_steps_cannot_exceed_k_max():
    with pytest.raises(ValueError, match="unroll_eval_steps"):
        NoiseReadProtocol(
            read_fraction=1.0, carry_enabled=False, unroll_k_max=1, unroll_eval_steps=2
        )


def test_read_passes_by_layout_and_unroll():
    cfg = _cfg()
    full = make_spec(layout=DiffusionLayout.FULL_SEQUENCE, self_conditioning=False)
    encoder = make_spec(layout=DiffusionLayout.ENCODER_CANVAS, self_conditioning=True)
    assert decision_read_passes(cfg, full) == 1
    # Encoder-canvas self-conditioning: step 2 reads the pilot, so the count matters.
    assert decision_read_passes(cfg, encoder) is None
    assert decision_read_passes(_cfg(k_max=1), encoder) == 1
    pinned = _cfg()
    pinned["diffusion_lm"]["unroll"]["eval_steps"] = 2
    assert decision_read_passes(pinned, encoder) == 2


# ------------------------------------------------------------------ read-time manifest check


def _write(directory, data) -> None:
    (directory / MANIFEST_FILENAME).write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"codebook": "expanded128"}, "label_codebook"),
        ({"canvas_width": 192}, "canvas_width"),
        ({"max_questions_per_canvas": 20}, "max_questions_per_canvas"),
        ({"layout": "prompt_slots"}, "decision_layout"),
    ],
)
def test_read_refuses_a_config_that_disagrees_with_the_adapter(tmp_path, change, field):
    build_decision_manifest(_cfg(), model=MODEL).write_to(tmp_path)
    read = _cfg(**change)
    read["lora_model_dir"] = str(tmp_path)

    with pytest.raises(ValueError, match=field):
        validate_read_manifest(read, model=MODEL)
    with pytest.raises(ValueError, match="does not match the manifest"):
        DiffusionDecisionPlugin().pre_lora_load(read, MODEL)


def test_read_accepts_a_matching_config_and_legacy_manifest(tmp_path):
    cfg = _cfg()
    build_decision_manifest(cfg, model=MODEL).write_to(tmp_path)
    read = _cfg()
    read["lora_model_dir"] = str(tmp_path)
    assert validate_read_manifest(read, model=MODEL) == tmp_path / MANIFEST_FILENAME

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    _write(legacy_dir, _legacy_manifest_json(cfg))
    # A legacy manifest does not record max_questions_per_canvas or read passes:
    # those are not compared, everything it does record still is.
    loose = _cfg(max_questions_per_canvas=20, k_max=1)
    loose["lora_model_dir"] = str(legacy_dir)
    assert validate_read_manifest(loose, model=MODEL) is not None
    wrong = _cfg(codebook="expanded128")
    wrong["lora_model_dir"] = str(legacy_dir)
    with pytest.raises(ValueError, match="label_codebook"):
        validate_read_manifest(wrong, model=MODEL)


def test_canvas_prefix_from_another_branch_is_refused(tmp_path):
    data = build_decision_manifest(_cfg(), model=MODEL).model_dump(mode="json")
    data["canvas_prefix_ids"] = [70, 71]
    _write(tmp_path, data)
    read = _cfg()
    read["lora_model_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="canvas_prefix_ids"):
        validate_read_manifest(read, model=MODEL)


def test_read_passes_mismatch_is_refused():
    saved = build_decision_manifest(_cfg(), model=MODEL)
    sampled = saved.model_copy(
        update={
            "noise_read_protocol": saved.noise_read_protocol.model_copy(
                update={"read_passes": 2}
            )
        }
    )
    problems = manifest_mismatches(sampled, saved)
    assert problems and "read_passes" in problems[0]


def test_merged_checkpoint_manifest_is_checked(tmp_path):
    build_decision_manifest(_cfg(), model=MODEL).write_to(tmp_path)
    read = _cfg(codebook="expanded128")
    read["base_model"] = str(tmp_path)  # merged checkpoint dir, no adapter
    assert read_manifest_path(read) == tmp_path / MANIFEST_FILENAME
    with pytest.raises(ValueError, match="label_codebook"):
        validate_read_manifest(read, model=MODEL)


def test_no_manifest_means_no_check(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    build_decision_manifest(_cfg(), model=MODEL).write_to(tmp_path)
    read = _cfg(codebook="expanded128")
    read["lora_model_dir"] = str(adapter)
    # the adapter carries no manifest; the base model's dir is not consulted for it
    read["base_model"] = str(tmp_path)
    assert read_manifest_path(read) is None
    assert validate_read_manifest(read, model=MODEL) is None
    hub = _cfg(codebook="expanded128")
    assert validate_read_manifest(hub, model=MODEL) is None


# ------------------------------------------------------------------ read-overflow canvas width

LONG = "a_rather_long_question_identifier"


def _overflow_cfg(width=None):
    cfg = make_cfg(
        diffusion_lm={"canvas_width": 24, "mask_token_id": 9},
        diffusion_decision=(
            {} if width is None else {"read_overflow_canvas_width": width}
        ),
    )
    cfg["diffusion_lm"]["unroll"] = {"k_max": 2}
    return cfg


def _rows(cfg, record, *, train=False, layout=DiffusionLayout.FULL_SEQUENCE):
    spec = make_spec(layout=layout, self_conditioning=False)
    return datasets._canvas_rows(
        ChatCharacterTokenizer(), [record], cfg, {}, spec, train=train
    )


def test_overflow_width_default_drops_the_canvas():
    rows, dropped = _rows(_overflow_cfg(), make_record(question=LONG))
    assert (rows, dropped) == ([], 1)


def test_overflow_width_rebuilds_only_overflowing_read_canvases():
    cfg = _overflow_cfg(64)
    rows, dropped = _rows(cfg, make_record(question=LONG))
    assert dropped == 0
    canvas = rows[0]["canvas"]
    assert len(canvas.canvas_ids) == 64
    assert canvas.template_length + 1 <= 64

    # A canvas that fits keeps the trained width and is identical to a read without the option.
    fits, _ = _rows(cfg, make_record(question="q"))
    plain, _ = _rows(_overflow_cfg(), make_record(question="q"))
    assert len(fits[0]["canvas"].canvas_ids) == 24
    assert fits[0]["canvas"] == plain[0]["canvas"]


def test_overflow_width_never_applies_to_training_rows():
    rows, dropped = _rows(_overflow_cfg(64), make_record(question=LONG), train=True)
    assert (rows, dropped) == ([], 1)


def test_overflow_width_still_drops_what_does_not_fit():
    rows, dropped = _rows(_overflow_cfg(30), make_record(question=LONG))
    assert (rows, dropped) == ([], 1)


def test_overflow_width_requires_full_sequence_layout():
    with pytest.raises(ValueError, match="full_sequence"):
        _rows(
            _overflow_cfg(64),
            make_record(question=LONG),
            layout=DiffusionLayout.ENCODER_CANVAS,
        )


def test_overflow_width_validation():
    with pytest.raises(ValueError):
        DiffusionDecisionConfig.model_validate({"read_overflow_canvas_width": 1})
    assert DiffusionDecisionConfig.model_validate({}).read_overflow_canvas_width is None


def test_allow_manifest_mismatch_warns_instead(tmp_path, caplog):
    build_decision_manifest(_cfg(), model=MODEL).write_to(tmp_path)
    read = _cfg(codebook="expanded128", allow_manifest_mismatch=True)
    read["lora_model_dir"] = str(tmp_path)
    with caplog.at_level("WARNING"):
        assert validate_read_manifest(read, model=MODEL) is not None
    assert "label_codebook" in caplog.text
