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


def _split_halves(group_suffixes):
    records = []
    for half, (suffix, count) in enumerate(zip(group_suffixes, (6, 5), strict=True)):
        record = _record(f"state/{half}", group=f"state{suffix}", state="shared")
        record["family"] = "state"
        record["questions"] = {
            f"h{half}q{index}": {"type": "noul"} for index in range(count)
        }
        record["labels"] = {
            f"h{half}q{index}": {"kind": "hard", "gold_idx": index % 2}
            for index in range(count)
        }
        records.append(record)
    return records


def test_exporter_split_halves_with_distinct_groups_stay_separate_and_unchanged():
    records = _split_halves(("/0", "/1"))

    grouped = group_records(records, max_questions=10)

    assert [record["id"] for record in grouped] == ["state/0", "state/1"]
    assert [len(record["questions"]) for record in grouped] == [6, 5]
    assert grouped == records
    assert all("grouped_record_ids" not in record for record in grouped)


def test_exporter_split_halves_sharing_a_group_are_merged_and_resplit():
    records = _split_halves(("", ""))

    grouped = group_records(records, max_questions=10)

    assert [record["id"] for record in grouped] == [
        "state/0#group0",
        "state/0#group1",
    ]
    assert [len(record["questions"]) for record in grouped] == [10, 1]
    assert [record["grouped_record_ids"] for record in grouped] == [
        ("state/0", "state/1"),
        ("state/1",),
    ]


class _CollidingState(str):
    """A state string whose hash collides with every other instance."""

    def __hash__(self):
        return 0


def test_hash_collisions_never_merge_unequal_states():
    records = [
        _record("one", state=_CollidingState("first")),
        _record("two", state=_CollidingState("second")),
        _record("three", state=_CollidingState("first")),
    ]

    grouped = group_records(records, max_questions=4)

    assert [record["id"] for record in grouped] == ["one#group0", "two"]
    assert grouped[0]["grouped_record_ids"] == ("one", "three")


def test_structured_state_groups_with_its_compact_json_text():
    records = [_record("one", state={"a": 1}), _record("two", state='{"a":1}')]

    grouped = group_records(records, max_questions=4)

    assert len(grouped) == 1
    assert grouped[0]["grouped_record_ids"] == ("one", "two")


def test_unchanged_records_are_new_top_level_dicts():
    record = _record("one")

    (grouped,) = group_records([record], max_questions=4)

    assert grouped == record
    assert grouped is not record
    grouped["id"] = "changed"
    assert record["id"] == "one"


def test_split_canvases_do_not_alias_input_targets():
    first, second = _record("one"), _record("two")

    grouped = group_records([first, second], max_questions=1)

    grouped[0]["labels"]["q"]["gold_idx"] = 9
    grouped[1]["grouped_source_metadata"]["two"]["origin"] = "changed"
    assert first["labels"]["q"]["gold_idx"] == 0
    assert second["source_metadata"] == {"origin": "two"}
