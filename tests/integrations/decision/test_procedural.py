import json

import pytest

from axolotl.integrations.decision.adapters.procedural import normalize


def test_soft_bool_preserved():
    r = {
        "id": "x:1",
        "state": "s",
        "questions": json.dumps({"q": {"type": "noul", "instructions": "x"}}),
        "answers": json.dumps({"q": {"noul": 2 / 3}}),
    }
    x = normalize(r)
    assert x["labels"]["q"]["kind"] == "dist"
    assert x["labels"]["q"]["probs"] == pytest.approx([2 / 3, 1 / 3])


def test_choice_identity():
    r = {
        "id": "x:1",
        "state": "s",
        "questions": json.dumps(
            {"q": {"type": "choice", "criteria": {"b": "b", "a": "a"}}}
        ),
        "answers": json.dumps({"q": {"probabilities": {"b": 0, "a": 1}}}),
    }
    assert normalize(r)["labels"]["q"]["gold_idx"] == 1


def test_missing_noul_rejected():
    r = {
        "id": "x:1",
        "state": "s",
        "questions": '{"q":{"type":"noul"}}',
        "answers": '{"q":{}}',
    }
    with pytest.raises(ValueError, match="missing noul"):
        normalize(r)


def test_choice_description_retained():
    r = {
        "id": "x:1",
        "state": "s",
        "questions": '{"q":{"type":"choice","criteria":{"a":"alpha","b":"beta"}}}',
        "answers": '{"q":{"probabilities":{"a":1,"b":0}}}',
    }
    assert normalize(r)["questions"]["q"]["options"] == [
        {"name": "a", "description": "alpha"},
        {"name": "b", "description": "beta"},
    ]


def test_noul_criteria_retained():
    r = {
        "id": "x:1",
        "state": "s",
        "questions": '{"q":{"type":"noul","criteria":{"true":"yes desc","false":"no desc"}}}',
        "answers": '{"q":{"noul":1}}',
    }
    assert normalize(r)["questions"]["q"]["criteria"]["true"] == "yes desc"


def test_source_id_is_the_group_without_an_invented_task_family():
    result = normalize(
        {
            "id": "arithmetic:train:7",
            "state": "s",
            "questions": '{"q":{"type":"noul"}}',
            "answers": '{"q":{"noul":1}}',
        }
    )
    assert result["group"] == "arithmetic:train:7"
    assert "family" not in result
    assert result["source_metadata"] == {
        "task": "arithmetic",
        "official_split": "train",
    }


def test_score_legend_identity_overrides_dictionary_order():
    row = {
        "id": "task:train:1",
        "state": "{}",
        "questions": {"q": {"type": "score", "criteria": ["low", "high"]}},
        "answers": {
            "q": {
                "legend": {"0": "high", "1": "low"},
                "probabilities": {"0": 0.8, "1": 0.2},
            }
        },
    }
    result = normalize(row)
    assert result["labels"]["q"]["probs"] == [0.2, 0.8]
    assert result["questions"]["q"]["levels"] == ["low", "high"]
    row["answers"]["q"]["legend"]["1"] = "high"
    with pytest.raises(ValueError, match="exactly once"):
        normalize(row)
