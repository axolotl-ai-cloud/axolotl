import pytest

from axolotl.integrations.diffusion_decision.grouping import group_records


def _record(
    identifier, *, source="source", group="group", state='{"a": 1}', instructions=""
):
    return {
        "id": identifier,
        "source": source,
        "group": group,
        "state": state,
        "instructions": instructions,
        "questions": {"q": {"type": "choice", "options": ["a", "b"]}},
        "labels": {"q": {"kind": "hard", "gold_idx": 0 if identifier == "one" else 1}},
        "source_metadata": {"origin": identifier},
    }


def test_groups_identical_context_and_stably_renames_question_collisions():
    grouped = group_records([_record("one"), _record("two")], max_questions=4)

    assert len(grouped) == 1
    record = grouped[0]
    assert list(record["questions"]) == ["q", "q__2"]
    assert record["labels"]["q"] == {"kind": "hard", "gold_idx": 0}
    assert record["labels"]["q__2"] == {"kind": "hard", "gold_idx": 1}
    assert record["grouped_record_ids"] == ("one", "two")
    assert record["grouped_question_metadata"]["q__2"] == {
        "record_id": "two",
        "original_question_id": "q",
        "source_metadata": {"origin": "two"},
    }


def test_keeps_mismatched_state_source_group_or_global_instructions_separate():
    records = [
        _record("one"),
        _record("two", state='{"a":1}'),
        _record("three", source="other"),
        _record("four", group="other"),
        _record("five", instructions="global"),
        {**_record("six"), "family": "other"},
    ]

    grouped = group_records(records, max_questions=4)

    assert len(grouped) == len(records)
    assert [record["id"] for record in grouped] == [record["id"] for record in records]
    assert [record["source_metadata"] for record in grouped] == [
        record["source_metadata"] for record in records
    ]


def test_rejects_empty_question_mappings():
    record = _record("one")
    record["questions"] = {}
    record["labels"] = {}

    try:
        group_records([record], max_questions=2)
    except ValueError as error:
        assert "at least one question" in str(error)
    else:
        raise AssertionError("empty questions were accepted")


def test_splits_at_question_bound_without_truncating_targets_or_metadata():
    first = _record("one")
    first["questions"] = {f"q{index}": {"type": "noul"} for index in range(3)}
    first["labels"] = {
        f"q{index}": {"kind": "dist", "probs": [index, 3 - index]} for index in range(3)
    }
    second = _record("two")
    second["questions"] = {"x": {"type": "score"}, "y": {"type": "choice"}}
    second["labels"] = {
        "x": {"kind": "set", "allowed_set": [0, 1]},
        "y": {"kind": "hard", "gold_idx": 1},
    }

    grouped = group_records([first, second], max_questions=2)

    assert [len(record["questions"]) for record in grouped] == [2, 2, 1]
    targets = [target for record in grouped for target in record["labels"].values()]
    assert targets == [
        {"kind": "dist", "probs": [0, 3]},
        {"kind": "dist", "probs": [1, 2]},
        {"kind": "dist", "probs": [2, 1]},
        {"kind": "set", "allowed_set": [0, 1]},
        {"kind": "hard", "gold_idx": 1},
    ]


def test_rejects_duplicate_record_ids_within_a_group():
    records = [_record("one"), _record("one"), _record("two")]
    with pytest.raises(ValueError, match="duplicated: one"):
        group_records(records, max_questions=1)


def test_records_without_ids_get_positional_ids_and_keep_their_metadata():
    records = [_record("one"), _record("two")]
    for record in records:
        del record["id"]
    grouped = group_records(records, max_questions=1)

    assert [record["id"] for record in grouped] == [
        "group#record0#group0",
        "group#record0#group1",
    ]
    assert grouped[1]["grouped_record_ids"] == ("group#record1",)
    assert grouped[1]["grouped_source_metadata"] == {"group#record1": {"origin": "two"}}
