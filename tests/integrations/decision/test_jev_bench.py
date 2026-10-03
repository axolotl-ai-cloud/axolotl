"""Human distributions take precedence over a single majority label."""

from pathlib import Path

import pytest

from axolotl.integrations.decision.adapters import normalize_record
from axolotl.integrations.decision.adapters.jev_bench import (
    normalize_jev_bench,
)
from axolotl.integrations.decision.template import (
    EXPANDED52,
    RESERVED151,
    RESERVED151_TOKEN_IDS,
    SPREADSHEET151,
    SchemaError,
    parse_decision_schema,
    parse_schema,
    resolve_template,
)


def record(question, label, soft=None):
    return {
        "id": "task/train/1",
        "source": "task",
        "state": '{"evidence":"text"}',
        "question": question,
        "label": label,
        "soft_label": soft,
        "meta": '{"license":"example-license"}',
    }


def test_choice_identity_order_and_description():
    result = normalize_jev_bench(
        record({"type": "choice", "criteria": {"B": "second", "A": "first"}}, "A")
    )
    assert result["labels"]["q1"] == {"kind": "hard", "gold_idx": 1}
    assert result["questions"]["q1"]["options"][0] == {
        "name": "B",
        "description": "second",
    }
    assert result["source_metadata"]["license"] == "example-license"


def test_score_uses_index_identity_not_level_description():
    result = normalize_jev_bench(
        record(
            {"type": "score", "criteria": ["low", "medium", "high"]},
            "1",
            '{"2":0.2,"0":0.3,"1":0.5}',
        )
    )
    assert result["labels"]["q1"] == {"kind": "dist", "probs": [0.3, 0.5, 0.2]}
    assert result["questions"]["q1"]["levels"] == ["low", "medium", "high"]


def test_noul_remaps_human_votes_to_yes_no():
    result = normalize_jev_bench(record({"type": "noul"}, "1", {"0": 0.4, "1": 0.6}))
    assert result["labels"]["q1"] == {"kind": "dist", "probs": [0.6, 0.4]}


def test_noul_scalar_human_yes_share_is_expanded_before_remapping():
    result = normalize_jev_bench(record({"type": "noul"}, "1", "0.3"))

    assert result["labels"]["q1"] == {"kind": "dist", "probs": [0.3, 0.7]}


def test_score_human_distribution_array_uses_published_level_order():
    result = normalize_jev_bench(
        record(
            {"type": "score", "criteria": ["low", "medium", "high"]},
            "2",
            "[0.1, 0.2, 0.7]",
        )
    )

    assert result["questions"]["q1"]["levels"] == ["low", "medium", "high"]
    assert result["labels"]["q1"] == {"kind": "dist", "probs": [0.1, 0.2, 0.7]}


def test_missing_probability_is_rejected():
    with pytest.raises(ValueError, match="each alternative"):
        normalize_jev_bench(record({"type": "noul"}, "1", {"1": 0.6}))


@pytest.mark.parametrize("soft", ["-0.1", "1.1", "NaN", "true"])
def test_noul_scalar_requires_a_finite_probability(soft):
    with pytest.raises(
        ValueError, match="finite and nonnegative|sum to one|specify each alternative"
    ):
        normalize_jev_bench(record({"type": "noul"}, "1", soft))


@pytest.mark.parametrize("soft", ["[0.4, 0.6]", "[0.2, -0.1, 0.9]"])
def test_score_array_requires_complete_valid_probability_mass(soft):
    with pytest.raises(ValueError, match="one probability|finite and nonnegative"):
        normalize_jev_bench(
            record({"type": "score", "criteria": ["low", "medium", "high"]}, "1", soft)
        )


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        {"type": "score", "criteria": ["low", "high"]},
    ],
)
def test_scalar_soft_label_is_only_valid_for_noul(question):
    with pytest.raises(ValueError, match="specify each alternative"):
        normalize_jev_bench(record(question, "0", "0.5"))


def test_expanded_codebook_accepts_a_28_way_choice_without_changing_default():
    schema = {
        "questions": [
            {
                "id": "q",
                "type": "choice",
                "options": list("ABCDEFGHIJKLMNOPQRSTUVWXYZab"),
            }
        ]
    }
    with pytest.raises(SchemaError, match="at most 26"):
        parse_schema(schema)
    parsed = parse_schema(schema, alphabet=EXPANDED52)
    assert parsed["questions"][0]["labels"][-2:] == ["a", "b"]


@pytest.mark.parametrize("cardinality", [26, 28, 52, 60, 77, 100, 151])
def test_spreadsheet151_accepts_categorical_cardinalities(cardinality):
    schema = {
        "questions": [
            {"id": "q", "type": "choice", "options": list(map(str, range(cardinality)))}
        ]
    }
    parsed = parse_schema(schema, alphabet=SPREADSHEET151, allow_multichar_labels=True)
    assert parsed["questions"][0]["labels"] == list(SPREADSHEET151[:cardinality])


def test_spreadsheet151_is_opt_in_and_preserves_targets_and_order():
    criteria = {f"class-{index}": f"description-{index}" for index in range(60)}
    raw = record(
        {"type": "choice", "criteria": criteria},
        "class-59",
        {key: 1 / 60 for key in criteria},
    )
    normalized = normalize_record(
        "jev_bench", raw, training=False, codebook="spreadsheet151"
    )
    assert normalized["questions"]["q1"]["options"][-1]["name"] == "class-59"
    assert normalized["labels"]["q1"] == {"kind": "dist", "probs": [1 / 60] * 60}
    with pytest.raises(SchemaError, match="at most 26"):
        normalize_record("jsonl", normalized, training=False)
    assert (
        normalize_record("jsonl", normalized, training=False, codebook="spreadsheet151")
        == normalized
    )


def test_spreadsheet151_rejects_152_without_changing_default_limit():
    schema = {
        "questions": [
            {"id": "q", "type": "choice", "options": list(map(str, range(152)))}
        ]
    }
    with pytest.raises(SchemaError, match="at most 151"):
        parse_schema(schema, alphabet=SPREADSHEET151, allow_multichar_labels=True)
    with pytest.raises(SchemaError, match="at most 26"):
        parse_schema(
            {
                "questions": [
                    {**schema["questions"][0], "options": list(map(str, range(27)))}
                ]
            }
        )


def test_spreadsheet151_first_26_labels_match_vendored26():
    assert SPREADSHEET151[:26] == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_reserved151_uses_existing_ids_above_the_control_boundary():
    assert RESERVED151_TOKEN_IDS == tuple(range(200, 351))
    assert len(RESERVED151) == 151
    assert 100 not in RESERVED151_TOKEN_IDS


def test_reserved151_rewrites_all_finite_answer_kinds_without_mutating_targets():
    schema = parse_decision_schema(
        {
            "questions": [
                {"id": "n", "type": "noul", "criteria": {"true": "y", "false": "n"}},
                {"id": "s", "type": "score", "levels": ["low", "mid", "high"]},
                {"id": "c", "type": "choice", "options": ["a", "b"]},
            ]
        },
        codebook="reserved151",
    )
    assert [question["labels"] for question in schema["questions"]] == [
        list(RESERVED151[:2]),
        list(RESERVED151[:3]),
        list(RESERVED151[:2]),
    ]


def test_reserved151_resolves_in_actual_answer_context_with_pinned_tokenizer():
    tokenizer_json = Path("/tmp/nemotron3b-checkpoint/tokenizer.json")
    if not tokenizer_json.is_file():
        pytest.skip("pinned 3B tokenizer unavailable")
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(str(tokenizer_json))

    class TokenizerAdapter:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return raw.encode(text).ids

    schema = parse_decision_schema(
        {
            "questions": [
                {"id": "n", "type": "noul", "criteria": {"true": "y", "false": "n"}},
                {"id": "s", "type": "score", "levels": ["low", "mid", "high"]},
                {"id": "c", "type": "choice", "options": list(map(str, range(151)))},
            ]
        },
        codebook="reserved151",
    )
    for fmt in ("lines", "indexed"):
        _, slots = resolve_template(
            TokenizerAdapter(), schema["questions"], fmt=fmt, width=128
        )
        assert slots[0]["label_ids"] == list(RESERVED151_TOKEN_IDS[:2])
        assert slots[1]["label_ids"] == list(RESERVED151_TOKEN_IDS[:3])
        assert slots[2]["label_ids"] == list(RESERVED151_TOKEN_IDS)


@pytest.mark.parametrize("label", [" A", "A ", "A\nB", "A\rB"])
def test_multichar_codebook_rejects_separator_labels(label):
    with pytest.raises(SchemaError, match="distinct nonempty labels"):
        parse_schema(
            {"questions": [{"id": "q", "type": "choice", "options": ["a", "b"]}]},
            alphabet=["A", label],
            allow_multichar_labels=True,
        )


_NEMOTRON_TOKENIZER = Path(
    "/mnt/data/hf_cache/hub/models--nvidia--Nemotron-Labs-Diffusion-8B/snapshots/"
    "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc"
)


@pytest.mark.skipif(
    not _NEMOTRON_TOKENIZER.is_dir(), reason="local Nemotron tokenizer unavailable"
)
@pytest.mark.parametrize("count", [1, 2, 10, 20])
def test_spreadsheet151_resolves_with_local_nemotron_tokenizer(count):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _NEMOTRON_TOKENIZER, local_files_only=True, trust_remote_code=True
    )
    schema = parse_schema(
        {
            "questions": [
                {
                    "id": f"q{index}",
                    "type": "choice",
                    "options": list(map(str, range(151))),
                }
                for index in range(count)
            ]
        },
        alphabet=SPREADSHEET151,
        allow_multichar_labels=True,
    )
    for fmt in ("lines", "indexed"):
        _, slots = resolve_template(tokenizer, schema["questions"], fmt=fmt, width=128)
        assert len(slots) == count
        assert all(len(set(slot["label_ids"])) == 151 for slot in slots)
