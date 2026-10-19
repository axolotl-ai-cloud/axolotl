"""answer_format and width-aware grouping: more than 10 questions per canvas in the lines format."""

from __future__ import annotations

import hashlib
import json

import pytest

from axolotl.integrations.diffusion_decision import datasets, prepared_cache
from axolotl.integrations.diffusion_decision.args import DiffusionDecisionConfig
from axolotl.integrations.diffusion_decision.grouping import group_records
from axolotl.integrations.diffusion_decision.manifest import (
    MANIFEST_FILENAME,
    DecisionManifest,
    build_decision_manifest,
    effective_answer_format,
    validate_read_manifest,
)
from axolotl.integrations.diffusion_decision.preprocessing import (
    build_decision_canvas,
    template_fits,
)
from axolotl.integrations.diffusion_decision.template import SchemaError
from axolotl.utils.dict import DictDefault

from .helpers import ChatCharacterTokenizer, make_cfg, make_spec

NEMOTRON = ("nvidia/Nemotron-Labs-Diffusion-3B", "nvidia/Nemotron-Labs-Diffusion-8B")


def _nemotron():
    transformers = pytest.importorskip("transformers")
    for name in NEMOTRON:
        try:
            return transformers.AutoTokenizer.from_pretrained(
                name, local_files_only=True, trust_remote_code=True
            )
        except Exception:  # noqa: BLE001 - not cached
            continue
    pytest.skip("no cached Nemotron-Labs-Diffusion tokenizer")


def _openjev_questions(count, *, noul=True):
    """Open-Jev style ids: noul, choice, score with _2, _3, ... suffixes. ``noul=False`` keeps the ids but makes
    the noul questions choices (the character tokenizer has no single-token yes/no)."""
    questions, labels = {}, {}
    kinds = ("noul", "choice", "score")
    for index in range(count):
        kind = kinds[index % 3]
        ordinal = index // 3 + 1
        qid = kind if ordinal == 1 else f"{kind}_{ordinal}"
        if kind == "noul" and noul:
            questions[qid] = {"type": "noul", "instructions": f"Is {index} true?"}
            labels[qid] = {"kind": "hard", "gold_idx": index % 2}
        elif kind in ("choice", "noul"):
            questions[qid] = {
                "type": "choice",
                "instructions": f"Pick {index}.",
                "options": ["red", "green", "blue"],
            }
            labels[qid] = {"kind": "hard", "gold_idx": index % 3}
        else:
            questions[qid] = {
                "type": "score",
                "instructions": f"Rate {index}.",
                "levels": ["low", "mid", "high"],
            }
            labels[qid] = {"kind": "hard", "gold_idx": index % 3}
    return questions, labels


def _record(rid, count, *, group="g", state="state", noul=False):
    questions, labels = _openjev_questions(count, noul=noul)
    return {
        "id": rid,
        "source": "openjev.fixture",
        "group": group,
        "state": state,
        "questions": questions,
        "labels": labels,
    }


def _build(tokenizer, record, *, answer_format="auto", width=128):
    return build_decision_canvas(
        tokenizer,
        record,
        prompt_ids=[1],
        scaffold_ids=(),
        turn_close_id=11,
        pad_id=0,
        vocab_size=len(tokenizer) if hasattr(tokenizer, "__len__") else 256,
        width=width,
        answer_format=answer_format,
    )


# --- byte-identical canvases for existing configs ---------------------------------------------------------------


def _digest_records():
    records = [_record(f"solo{i}", 1 + i % 9, state=f"s{i}") for i in range(12)]
    records += [_record("big", 13, state="big")]  # split 10 + 3 by count
    records += [_record(f"m{i}", 4, group="merged", state="shared") for i in range(4)]
    return records


def _legacy_canvas_json(canvas):
    # Image fields (VLM support) postdate the legacy digest; text canvases leave
    # them empty, so drop them to compare the rest of the canvas byte for byte.
    value = prepared_cache._canvas_to_json(canvas)
    for name in ("image_refs", "image_sizes"):
        if not value.get(name):
            value.pop(name, None)
    return value


def _canvas_digest(cfg):
    spec = make_spec()
    tokenizer = ChatCharacterTokenizer()
    grouped = group_records(
        [json.loads(json.dumps(r)) for r in _digest_records()],
        max_questions=int(cfg["diffusion_decision"]["max_questions_per_canvas"]),
        fits=datasets._grouping_fits(tokenizer, cfg, spec),
    )
    rows, dropped = datasets._canvas_rows(tokenizer, grouped, cfg, {}, spec)
    payload = [
        [row["record"]["id"], _legacy_canvas_json(row["canvas"])] for row in rows
    ]
    text = json.dumps([payload, dropped], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


# Computed with feat/decision-release-fixes @ebe3a7e4d (before answer_format existed).
LEGACY_DIGEST_MAX10 = "2402aabb46a90747e480c7112aa08f1214f6fbb96cf5fe200a382c7642c71deb"


def test_existing_configs_prepare_byte_identical_canvases():
    cfg = make_cfg(diffusion_decision={"max_questions_per_canvas": 10})
    assert _canvas_digest(cfg) == LEGACY_DIGEST_MAX10
    explicit = make_cfg(
        diffusion_decision={"max_questions_per_canvas": 10, "answer_format": "auto"}
    )
    assert _canvas_digest(explicit) == LEGACY_DIGEST_MAX10


def test_auto_is_the_default_and_groups_by_count_only():
    assert DiffusionDecisionConfig().answer_format == "auto"
    assert (
        datasets._grouping_fits(ChatCharacterTokenizer(), make_cfg(), make_spec())
        is None
    )


def test_lines_at_ten_matches_auto_canvases_for_records_that_fit():
    tokenizer = ChatCharacterTokenizer()
    for count in (1, 5, 10):
        record = _record("r", count)
        assert _build(tokenizer, record, answer_format="lines") == _build(
            tokenizer, record
        )


# --- lines format above 10 questions -------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [11, 14, 18, 25])
def test_lines_format_reads_more_than_ten_openjev_questions(count):
    tokenizer = _nemotron()
    record = _record("r", count, noul=True)
    canvas = _build(tokenizer, record, answer_format="lines", width=256)
    assert len(canvas.question_ids) == count
    assert len(set(canvas.label_positions)) == count
    assert all(len(set(ids)) == len(ids) for ids in canvas.allowed_ids)
    # Each label slot is one token: swapping a label changes exactly that position.
    text = tokenizer.decode(canvas.canvas_ids[: canvas.template_length])
    assert text.count("\n") == count - 1


def test_auto_and_indexed_break_on_openjev_noul_ids_above_ten():
    tokenizer = _nemotron()
    record = _record("r", 12, noul=True)
    with pytest.raises(SchemaError):
        _build(tokenizer, record, width=256)
    with pytest.raises(SchemaError):
        _build(tokenizer, record, answer_format="indexed", width=256)


def test_lines_template_measure_matches_the_built_canvas():
    tokenizer = _nemotron()
    for count in range(1, 30):
        record = _record("r", count, noul=True)
        fits = template_fits(
            tokenizer,
            record["questions"],
            head=(),
            pad_slots=0,
            width=128,
            answer_format="lines",
        )
        try:
            canvas = _build(tokenizer, record, answer_format="lines")
        except SchemaError as error:
            assert datasets._is_canvas_overflow(error)
            assert not fits
        else:
            assert fits
            assert canvas.template_length + 1 <= 128


# --- width-aware grouping ------------------------------------------------------------------------------------------


def _lines_cfg(max_questions, width=128):
    return make_cfg(
        diffusion_lm={"canvas_width": width, "mask_token_id": 9},
        diffusion_decision={
            "max_questions_per_canvas": max_questions,
            "answer_format": "lines",
        },
    )


def test_overflowing_group_is_split_into_canvases_that_fit():
    tokenizer = ChatCharacterTokenizer()
    cfg = _lines_cfg(64, width=64)
    spec = make_spec()
    fits = datasets._grouping_fits(tokenizer, cfg, spec)
    records = [_record("big", 30)]
    grouped = group_records(records, max_questions=64, fits=fits)
    assert len(grouped) > 1
    assert sum(len(r["questions"]) for r in grouped) == 30
    assert all(fits(r["questions"]) for r in grouped)
    # Packed greedily: adding the next canvas's first question would overflow.
    for left, right in zip(grouped, grouped[1:], strict=False):
        first = next(iter(right["questions"].items()))
        assert not fits({**left["questions"], "x" * len(first[0]): first[1]})
    rows, dropped = datasets._canvas_rows(tokenizer, grouped, cfg, {}, spec)
    assert dropped == 0 and len(rows) == len(grouped)


def test_count_cap_still_applies_when_everything_fits():
    tokenizer = ChatCharacterTokenizer()
    cfg = _lines_cfg(4)
    fits = datasets._grouping_fits(tokenizer, cfg, make_spec())
    grouped = group_records([_record("big", 10)], max_questions=4, fits=fits)
    assert [len(r["questions"]) for r in grouped] == [4, 4, 2]


def test_merged_records_pack_to_width_and_keep_metadata():
    tokenizer = ChatCharacterTokenizer()
    cfg = _lines_cfg(64, width=96)
    fits = datasets._grouping_fits(tokenizer, cfg, make_spec())
    records = [_record(f"m{i}", 5, group="merged", state="shared") for i in range(6)]
    grouped = group_records(records, max_questions=64, fits=fits)
    assert sum(len(r["questions"]) for r in grouped) == 30
    assert all(fits(r["questions"]) for r in grouped)
    for chunk in grouped:
        for qid in chunk["questions"]:
            meta = chunk["grouped_question_metadata"][qid]
            assert meta["record_id"].startswith("m")


def test_record_that_fits_is_returned_unchanged():
    tokenizer = ChatCharacterTokenizer()
    fits = datasets._grouping_fits(tokenizer, _lines_cfg(24), make_spec())
    record = _record("r", 12)
    (grouped,) = group_records([record], max_questions=24, fits=fits)
    assert grouped == record


def test_a_question_too_long_alone_still_gets_its_own_canvas():
    tokenizer = ChatCharacterTokenizer()
    cfg = _lines_cfg(64, width=16)
    fits = datasets._grouping_fits(tokenizer, cfg, make_spec())
    record = {
        **_record("r", 2),
        "questions": {
            "q": {"type": "choice", "options": ["a", "b"]},
            "a_question_id_longer_than_the_canvas": {
                "type": "choice",
                "options": ["a", "b"],
            },
        },
        "labels": {
            "q": {"kind": "hard", "gold_idx": 0},
            "a_question_id_longer_than_the_canvas": {"kind": "hard", "gold_idx": 1},
        },
    }
    grouped = group_records([record], max_questions=64, fits=fits)
    assert [list(r["questions"]) for r in grouped] == [
        ["q"],
        ["a_question_id_longer_than_the_canvas"],
    ]
    rows, dropped = datasets._canvas_rows(tokenizer, grouped, cfg, {}, make_spec())
    assert (len(rows), dropped) == (1, 1)


def test_dependencies_do_not_break_the_width_measure():
    tokenizer = ChatCharacterTokenizer()
    fits = datasets._grouping_fits(tokenizer, _lines_cfg(24), make_spec())
    assert fits({"b": {"type": "noul", "depends_on": ["a"]}})


def test_answer_format_changes_the_prepared_cache_identity():
    auto = prepared_cache._semantic_config(make_cfg())
    lines = prepared_cache._semantic_config(
        make_cfg(diffusion_decision={"answer_format": "lines"})
    )
    assert auto != lines


# --- manifest ------------------------------------------------------------------------------------------------------

MODEL = None


def _manifest_cfg(**decision):
    return DictDefault(
        {
            "base_model": "nvidia/Nemotron-Labs-Diffusion-8B",
            "model_config_type": "nemotron_labs_diffusion",
            "diffusion_lm": {
                "canvas_width": 128,
                "mask_token_id": 100,
                "unroll": {"k_max": 2},
            },
            "diffusion_decision": {"max_questions_per_canvas": 10, **decision},
        }
    )


def test_manifest_round_trips_answer_format(tmp_path):
    cfg = _manifest_cfg(max_questions_per_canvas=24, answer_format="lines")
    manifest = build_decision_manifest(cfg)
    assert manifest.answer_format == "lines"
    path = manifest.write_to(tmp_path)
    assert DecisionManifest.from_path(path) == manifest
    assert json.loads(path.read_text())["answer_format"] == "lines"


def test_effective_answer_format():
    assert effective_answer_format("auto", 10) == "lines"
    assert effective_answer_format("auto", 20) == "auto"
    assert effective_answer_format(None, 8) == "lines"
    assert effective_answer_format("indexed", 8) == "indexed"


def test_read_check_compares_the_effective_answer_format(tmp_path):
    build_decision_manifest(_manifest_cfg()).write_to(tmp_path)  # auto, max 10
    same = _manifest_cfg(answer_format="lines")
    same["lora_model_dir"] = str(tmp_path)
    assert validate_read_manifest(same) == tmp_path / MANIFEST_FILENAME

    other = tmp_path / "lines24"
    other.mkdir()
    build_decision_manifest(
        _manifest_cfg(max_questions_per_canvas=24, answer_format="lines")
    ).write_to(other)
    wrong = _manifest_cfg(max_questions_per_canvas=24, answer_format="indexed")
    wrong["lora_model_dir"] = str(other)
    with pytest.raises(ValueError, match="answer_format"):
        validate_read_manifest(wrong)
    allowed = _manifest_cfg(allow_manifest_mismatch=True)
    allowed["lora_model_dir"] = str(other)
    assert validate_read_manifest(allowed) is not None


def test_legacy_manifest_without_answer_format_is_not_compared(tmp_path):
    data = build_decision_manifest(_manifest_cfg()).model_dump(mode="json")
    del data["answer_format"]
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    read = _manifest_cfg(answer_format="indexed")
    read["lora_model_dir"] = str(tmp_path)
    assert validate_read_manifest(read) is not None
