"""Nimble provenance and rule certificates never become prompt state."""

import pytest

from axolotl.integrations.decision.adapters.nimble import normalize_nimble
from axolotl.integrations.decision.template import parse_schema, system_text


def record(question, target):
    return {
        "id": "example-1",
        "source_family": "family-a",
        "family": "variant-group",
        "input": {"state": {"observed": "facts"}, "questions": {"decision": question}},
        "reference": {"target": target},
        "evidence_certificate": {"gold": "secret"},
    }


@pytest.mark.parametrize("target,index", [(True, 0), (False, 1), ("true", 0), ("0", 1)])
def test_boolean_target_order(target, index):
    result = normalize_nimble(record({"type": "noul"}, target))
    assert result["labels"]["decision"] == {"kind": "hard", "gold_idx": index}
    assert result["state"] == {"observed": "facts"}
    assert result["family"] == "family-a"
    assert "evidence_certificate" not in result


@pytest.mark.parametrize("target", [1, "high"])
def test_score_accepts_index_or_named_level(target):
    result = normalize_nimble(
        record({"type": "score", "criteria": ["low", "high"]}, target)
    )
    assert result["labels"]["decision"]["gold_idx"] == 1


def test_ambiguous_multi_question_reference_rejected():
    row = record({"type": "noul"}, True)
    row["input"]["questions"]["other"] = {"type": "noul"}
    with pytest.raises(ValueError, match="exactly one"):
        normalize_nimble(row)


def test_global_instructions_survive_into_decision_system_prompt():
    row = record({"type": "noul", "instructions": "Is access permitted?"}, True)
    row["input"]["instructions"] = "Access requires both certificates."
    normalized = normalize_nimble(row)
    schema = parse_schema(
        {
            "instructions": normalized["instructions"],
            "questions": [
                {**question, "id": key}
                for key, question in normalized["questions"].items()
            ],
        }
    )
    prompt = system_text(schema)
    assert "Access requires both certificates." in prompt
    assert "Is access permitted?" in prompt
    assert "secret" not in prompt
