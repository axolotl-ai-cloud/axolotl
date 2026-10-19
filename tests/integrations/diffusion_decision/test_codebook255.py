"""The expanded255 label codebook: tokenizer verification, round trips, and per-question label shuffling."""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from axolotl.integrations.diffusion_decision import datasets
from axolotl.integrations.diffusion_decision.adapters.jsonl import normalize_jsonl
from axolotl.integrations.diffusion_decision.args import DiffusionDecisionConfig
from axolotl.integrations.diffusion_decision.budgets import (
    decision_budget,
    exceeds_budget,
)
from axolotl.integrations.diffusion_decision.manifest import DecisionManifest
from axolotl.integrations.diffusion_decision.permutation import (
    permute_record,
    shuffle_labels,
)
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.template import (
    CODEBOOK_SIZES,
    EXPANDED128,
    EXPANDED255,
    SchemaError,
    codebook_alphabet,
    parse_decision_schema,
    resolve_template,
    system_text,
)
from axolotl.integrations.diffusion_decision.vendored.djev_template import parse_schema
from axolotl.model_support import DiffusionLayout

from .helpers import ChatCharacterTokenizer

ADDED = EXPANDED255[128:]


def _record(count, gold, *, rid="r"):
    return {
        "source": "fixture",
        "group": "group",
        "id": rid,
        "state": {"seen": "objects"},
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Pick one.",
                "options": [
                    {"name": f"o{i}", "description": f"option {i}"}
                    for i in range(count)
                ],
            }
        },
        "labels": {"choice": {"kind": "hard", "gold_idx": gold}},
    }


def test_expanded255_extends_expanded128_in_order():
    assert EXPANDED255[:128] == EXPANDED128
    assert len(EXPANDED255) == 255 == CODEBOOK_SIZES["expanded255"]
    assert len(set(EXPANDED255)) == 255
    assert codebook_alphabet("expanded255") is EXPANDED255
    for label in ADDED:
        assert len(label) == 1 and not label.isspace()
        # letters only (Georgian Mkhedruli is Ll since Unicode 11): no marks, digits or symbols
        assert unicodedata.category(label) in {"Lo", "Ll"}
        assert unicodedata.normalize("NFC", label) == label


def test_expanded255_has_no_ascii_or_greek_or_cyrillic_lookalikes():
    scripts = {unicodedata.name(label).split()[0] for label in ADDED}
    assert scripts == {
        "HIRAGANA",
        "KATAKANA",
        "HANGUL",
        "HEBREW",
        "DEVANAGARI",
        "THAI",
        "GEORGIAN",
    }
    # dakuten kana and Hebrew final forms are look-alikes of kept letters
    for label in ADDED:
        assert "VOICED" not in unicodedata.name(label)
        assert "FINAL" not in unicodedata.name(label)


@pytest.mark.parametrize("count", [129, 200, 255])
def test_round_trip_large_choice_questions(count):
    gold = count - 2
    original = _record(count, gold)
    with pytest.raises(SchemaError, match="at most 128 alternatives"):
        normalize_jsonl(original, codebook="expanded128")
    normalized = normalize_jsonl(original, codebook="expanded255")
    assert normalized["labels"] == original["labels"]
    permuted = permute_record(original, seed=7, codebook="expanded255")
    options = permuted["questions"]["choice"]["options"]
    assert sorted(o["name"] for o in options) == sorted(
        o["name"] for o in original["questions"]["choice"]["options"]
    )
    assert options[permuted["labels"]["choice"]["gold_idx"]]["name"] == f"o{gold}"
    schema = parse_decision_schema(
        {"questions": [{**q, "id": k} for k, q in permuted["questions"].items()]},
        codebook="expanded255",
    )
    assert schema["questions"][0]["labels"] == list(EXPANDED255[:count])
    text = system_text(schema)
    assert f"  {EXPANDED255[count - 1]}: " in text


def test_256_options_are_refused():
    with pytest.raises(SchemaError, match="at most 255 alternatives"):
        normalize_jsonl(_record(256, 0), codebook="expanded255")


@pytest.mark.parametrize("count", [129, 255])
def test_canvas_and_budget_for_large_questions(count):
    tokenizer = ChatCharacterTokenizer()
    record = permute_record(_record(count, 5), seed=1, codebook="expanded255")
    canvas = build_decision_canvas(
        tokenizer,
        record,
        list(
            range(3000)
        ),  # a long prompt: the option list lives in the prompt, not the canvas
        scaffold_ids=(),
        turn_close_id=11,
        pad_id=0,
        vocab_size=70000,
        width=128,
        codebook="expanded255",
    )
    assert len(canvas.canvas_ids) == 128
    assert len(canvas.allowed_ids) == 1 and len(canvas.allowed_ids[0]) == count
    assert canvas.allowed_ids[0] == tuple(ord(label) for label in EXPANDED255[:count])
    budget = decision_budget(
        prompt_tokens=len(canvas.prompt_ids),
        canvas_tokens=len(canvas.canvas_ids),
        layout=DiffusionLayout.FULL_SEQUENCE,
    )
    assert exceeds_budget(budget, logical_limit=4096, physical_limit=None) is None
    assert exceeds_budget(budget, logical_limit=3000, physical_limit=None) == "logical"


def test_config_and_manifest_accept_expanded255():
    config = DiffusionDecisionConfig.model_validate(
        {"labels": {"codebook": "expanded255", "label_shuffle_prob": 0.2}}
    )
    assert config.labels.codebook == "expanded255"
    assert config.labels.label_shuffle_prob == 0.2
    assert DiffusionDecisionConfig.model_validate({}).labels.label_shuffle_prob == 0.0
    with pytest.raises(ValueError):
        DiffusionDecisionConfig.model_validate({"labels": {"label_shuffle_prob": 1.5}})
    with pytest.raises(ValueError):
        DiffusionDecisionConfig.model_validate({"labels": {"codebook": "expanded256"}})
    field = DecisionManifest.model_fields["label_codebook"]
    assert TypeAdapter(field.annotation).validate_python("expanded255") == "expanded255"


def test_label_indices_override_and_validation():
    question = {
        "id": "q",
        "type": "choice",
        "options": ["a", "b", "c"],
        "label_indices": [3, 130, 254],
    }
    schema = parse_decision_schema({"questions": [question]}, codebook="expanded255")
    assert schema["questions"][0]["labels"] == [
        EXPANDED255[3],
        EXPANDED255[130],
        EXPANDED255[254],
    ]
    for bad in ([1, 1, 2], [0, 1], [0, 1, 255], [0, 1, True]):
        with pytest.raises(SchemaError, match="label_indices"):
            parse_decision_schema(
                {"questions": [{**question, "label_indices": bad}]},
                codebook="expanded255",
            )
    with pytest.raises(SchemaError, match="choice question"):
        parse_decision_schema(
            {"questions": [{"id": "n", "type": "noul", "label_indices": [0, 1]}]},
            codebook="expanded255",
        )


def test_shuffle_labels_is_deterministic_and_off_by_default():
    record = normalize_jsonl(_record(12, 4), codebook="expanded255")
    assert (
        shuffle_labels(record, seed=1, probability=0.0, codebook="expanded255")
        == record
    )
    first = shuffle_labels(record, seed=1, probability=1.0, codebook="expanded255")
    again = shuffle_labels(record, seed=1, probability=1.0, codebook="expanded255")
    assert first == again
    indices = first["questions"]["choice"]["label_indices"]
    assert len(indices) == 12 and indices == sorted(set(indices))
    assert "label_indices" not in record["questions"]["choice"]  # input untouched
    other = shuffle_labels(record, seed=2, probability=1.0, codebook="expanded255")
    assert other["questions"]["choice"]["label_indices"] != indices
    # full-width questions have nothing to shuffle
    full = normalize_jsonl(_record(255, 0), codebook="expanded255")
    assert (
        "label_indices"
        not in shuffle_labels(full, seed=1, probability=1.0, codebook="expanded255")[
            "questions"
        ]["choice"]
    )


def test_shuffle_labels_rate_and_coverage():
    shuffled, high = 0, set()
    for index in range(2000):
        record = normalize_jsonl(_record(4, 0, rid=f"r{index}"), codebook="expanded255")
        result = shuffle_labels(record, seed=0, probability=0.2, codebook="expanded255")
        indices = result["questions"]["choice"].get("label_indices")
        if indices is not None:
            shuffled += 1
            high.update(i for i in indices if i >= 128)
    assert 330 < shuffled < 470
    assert len(high) > 100  # labels 129-255 are reached on four-option questions


def test_canvas_row_shuffles_training_rows_only():
    cfg = {
        "seed": 3,
        "model_config": {"vocab_size": 70000},
        "diffusion_lm": {"mask_token_id": 100},
        "diffusion_decision": {
            "labels": {"codebook": "expanded255", "label_shuffle_prob": 1.0}
        },
    }
    spec = SimpleNamespace(max_canvas=128, noise=datasets.DiffusionNoise.ABSORBING)
    record = _record(6, 2)
    tokenizer = ChatCharacterTokenizer()
    plain = datasets._canvas_row(tokenizer, record, cfg, 1.0, spec=spec)
    assert plain["canvas"].allowed_ids[0] == tuple(ord(c) for c in EXPANDED255[:6])
    trained = datasets._canvas_row(tokenizer, record, cfg, 1.0, spec=spec, train=True)
    indices = trained["record"]["questions"]["choice"]["label_indices"]
    assert trained["canvas"].allowed_ids[0] == tuple(
        ord(EXPANDED255[i]) for i in indices
    )
    assert trained["canvas"].targets == plain["canvas"].targets
    again = datasets._canvas_row(tokenizer, record, cfg, 1.0, spec=spec, train=True)
    assert again["canvas"] == trained["canvas"]


# --- real tokenizers -------------------------------------------------------------------------------------------

TOKENIZERS = {
    "nemotron-diffusion": [
        "nvidia/Nemotron-Labs-Diffusion-8B",
        "nvidia/Nemotron-Labs-Diffusion-3B",
    ],
    "qwen3.5": [f"Qwen/Qwen3.5-{size}" for size in ("4B", "9B", "2B", "0.8B", "27B")],
    "gemma3": ["google/gemma-3-4b-it", "google/gemma-3-1b-it", "unsloth/gemma-3-1b-it"],
    "diffusiongemma": ["google/diffusiongemma-26B-A4B-it"],
}
_TABLE: dict[str, str] = {}


def _cached_tokenizer(candidates):
    transformers = pytest.importorskip("transformers")
    for name in candidates:
        try:
            return name, transformers.AutoTokenizer.from_pretrained(
                name, local_files_only=True, trust_remote_code=True
            )
        except Exception:  # noqa: BLE001 - not cached (or not loadable offline)
            continue
    return None, None


def _contexts():
    """Answer-template contexts the label must survive: the lines format with short and long ids, alone and between
    neighbouring questions, with and without a lead join; and the indexed format (> 10 questions) with digit-ending
    ids, the label first, mid and last."""
    out = []
    for ids in (
        ["q"],
        ["intent"],
        ["a1"],
        ["Q_2"],
        ["x", "q", "r"],
        ["p0", "p1", "p2"],
    ):
        for target in range(len(ids)):
            for lead in ("", "\n"):
                out.append(("lines", ids, target, lead))
    for ids in ([f"q{i}" for i in range(12)], [f"opt_{i}" for i in range(11)]):
        for target in (0, len(ids) // 2, len(ids) - 1):
            for lead in ("", " "):
                out.append(("indexed", ids, target, lead))
    return out


def _check(tokenizer, alphabet):
    """resolve_template over every context with a 255-label question: one token per label, one shared slot,
    distinct ids. Returns {format: contexts passed} and the failures."""
    passed, failures = {"lines": 0, "indexed": 0}, []
    for fmt, ids, target, lead in _contexts():
        questions = []
        for index, qid in enumerate(ids):
            options = (
                [f"o{i}" for i in range(len(alphabet))]
                if index == target
                else ["x", "y"]
            )
            questions.append({"id": qid, "type": "choice", "options": options})
        schema = parse_schema({"questions": questions}, alphabet=alphabet)
        assert schema["format"] == fmt
        try:
            _, slots = resolve_template(
                tokenizer, schema["questions"], lead=lead, fmt=fmt, width=128
            )
        except SchemaError as error:
            failures.append(f"{fmt} {ids[target]!r} lead={lead!r}: {error}")
            continue
        assert len(set(slots[target]["label_ids"])) == len(alphabet)
        passed[fmt] += 1
    return passed, failures


@pytest.mark.parametrize("family", sorted(TOKENIZERS))
def test_expanded255_labels_are_single_distinct_tokens(family):
    name, tokenizer = _cached_tokenizer(TOKENIZERS[family])
    if tokenizer is None:
        pytest.skip(f"no cached {family} tokenizer")
    lines_total = sum(1 for c in _contexts() if c[0] == "lines")
    indexed_total = len(_contexts()) - lines_total
    # base label A (first label) and, reversed, a new label as the base the others are compared with
    forward, forward_failures = _check(tokenizer, EXPANDED255)
    backward, backward_failures = _check(tokenizer, EXPANDED255[::-1])
    # the added labels alone (label 0 is a new label); indexed contexts only need digit-ending ids
    added, added_failures = _check(tokenizer, ADDED)
    row = (
        f"{family:20s} {name:36s} lines {forward['lines']}/{lines_total} "
        f"rev {backward['lines']}/{lines_total} added {added['lines']}/{lines_total} | "
        f"indexed {forward['indexed']}/{indexed_total} rev {backward['indexed']}/{indexed_total} "
        f"added {added['indexed']}/{indexed_total}"
    )
    _TABLE[family] = row
    print("\n" + row)
    # the lines format is the contract EXPANDED128 was verified against; it must hold for all 255 labels
    assert forward["lines"] == lines_total, forward_failures[:3]
    assert backward["lines"] == lines_total, backward_failures[:3]
    # the added labels must also hold in the indexed format (EXPANDED128's Θ merges with a preceding digit for
    # Nemotron, so the full alphabet is only reported there)
    assert added["indexed"] == indexed_total and added["lines"] == lines_total, (
        added_failures[:3]
    )


def test_print_tokenizer_table():
    if _TABLE:
        print("\nexpanded255 verification (contexts passed):")
        for family in sorted(_TABLE):
            print("  " + _TABLE[family])
