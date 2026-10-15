"""Group compatible normalized decision records without changing their labels."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any


def group_records(
    records: Sequence[Mapping[str, Any]], *, max_questions: int
) -> list[dict[str, Any]]:
    """Combine only records with identical serving context and split without loss.

    Records sharing ``(source, group, family, state, instructions, images)`` are
    merged in input order and split into canvases of at most ``max_questions``
    questions. A record that is alone in its group and already fits is returned
    as a shallow copy (nested ``questions``/``labels``/metadata are shared with
    the input, so do not mutate them in place); merged or split canvases are
    built from deep copies.
    """
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")
    # Plain dicts keep insertion order; the key holds the original state and
    # instruction strings by reference, so grouping costs no extra copy of the
    # state text and dict equality (not the hash alone) decides every merge.
    groups: dict[_GroupKey, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(_key(record), []).append(record)
    result: list[dict[str, Any]] = []
    for group in groups.values():
        result.extend(_split_group(group, max_questions))
    return result


_GroupKey = tuple[str, str, str, str, str, str]


def _key(record: Mapping[str, Any]) -> _GroupKey:
    source = record.get("source")
    group = record.get("group")
    if not isinstance(source, str) or not source:
        raise ValueError("decision record requires a nonempty source")
    if not isinstance(group, str) or not group:
        raise ValueError("decision record requires a nonempty source group")
    family = record.get("family", group)
    if not isinstance(family, str) or not family:
        raise ValueError("decision record requires a nonempty family when present")
    return (
        source,
        group,
        family,
        _text(record.get("state")),
        _text(record.get("instructions", "")),
        _text(list(record.get("images") or ())),
    )


def _text(value: Any) -> str:
    """Return the serving text of a state; equal text means equal UTF-8 bytes."""
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
        allow_nan=False,
    )


_ATOMIC = frozenset({str, int, float, bool, type(None)})


def _copy(value: Any) -> Any:
    """Deep-copy a JSON-shaped tree much faster than ``copy.deepcopy``.

    Plain dicts, lists and scalars are copied directly; anything else falls
    back to ``copy.deepcopy``. Unlike ``deepcopy`` this does not preserve
    aliasing between subtrees, which normalized JSON records do not rely on.
    """
    kind = type(value)
    if kind is dict:
        return {
            key: item if type(item) in _ATOMIC else _copy(item)
            for key, item in value.items()
        }
    if kind is list:
        return [item if type(item) in _ATOMIC else _copy(item) for item in value]
    if kind in _ATOMIC:
        return value
    return copy.deepcopy(value)


def _split_group(
    records: Sequence[Mapping[str, Any]], max_questions: int
) -> list[dict[str, Any]]:
    if len(records) == 1:
        record = records[0]
        questions = record.get("questions")
        labels = record.get("labels")
        if not isinstance(questions, Mapping) or not isinstance(labels, Mapping):
            raise ValueError("decision record requires questions and labels mappings")
        if not questions:
            raise ValueError("decision record requires at least one question")
        if set(questions) != set(labels):
            raise ValueError("decision record question and label ids differ")
        if len(questions) <= max_questions:
            # Unchanged records are returned as a new top-level dict that shares
            # its nested values with the input; deep-copying every record
            # dominated preprocessing time and doubled label memory, and the
            # canvas pipeline deep-copies again in ``permute_record``.
            return [dict(record)]
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    used: set[str] = set()
    seen_ids: set[str] = set()
    for position, record in enumerate(records):
        questions = record.get("questions")
        labels = record.get("labels")
        record_id = _record_id(record, position)
        if not isinstance(questions, Mapping) or not isinstance(labels, Mapping):
            raise ValueError("decision record requires questions and labels mappings")
        if not questions:
            raise ValueError("decision record requires at least one question")
        if record_id in seen_ids:
            raise ValueError(f"decision record id is duplicated: {record_id}")
        seen_ids.add(record_id)
        if set(questions) != set(labels):
            raise ValueError("decision record question and label ids differ")
        for question_id, question in questions.items():
            if not isinstance(question_id, str) or not question_id:
                raise ValueError("decision question ids must be nonempty strings")
            if current is None or len(current["questions"]) == max_questions:
                current = _new_chunk(records[0], len(chunks))
                chunks.append(current)
                used = set()
            output_id = _unique_id(question_id, used)
            used.add(output_id)
            current["questions"][output_id] = _copy(question)
            current["labels"][output_id] = _copy(labels[question_id])
            current["grouped_record_ids"].append(record_id)
            if record_id not in current["grouped_source_metadata"]:
                current["grouped_source_metadata"][record_id] = _copy(
                    record.get("source_metadata", {})
                )
            current["grouped_question_metadata"][output_id] = {
                "record_id": record_id,
                "original_question_id": question_id,
                "source_metadata": _copy(record.get("source_metadata", {})),
            }
    for chunk in chunks:
        chunk["grouped_record_ids"] = tuple(dict.fromkeys(chunk["grouped_record_ids"]))
    return chunks


def _record_id(record: Mapping[str, Any], position: int) -> str:
    record_id = record.get("id")
    if isinstance(record_id, str) and record_id:
        return record_id
    if record_id is not None:
        raise ValueError("decision record id must be a nonempty string")
    return f"{record.get('group')}#record{position}"


def _new_chunk(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = {
        key: _copy(value)
        for key, value in record.items()
        if key not in {"questions", "labels", "id", "source_metadata"}
    }
    result["id"] = f"{_record_id(record, 0)}#group{index}"
    result["source_metadata"] = _copy(record.get("source_metadata", {}))
    result["questions"] = OrderedDict()
    result["labels"] = OrderedDict()
    result["grouped_record_ids"] = []
    result["grouped_question_metadata"] = {}
    result["grouped_source_metadata"] = {}
    return result


def _unique_id(question_id: str, used: set[str]) -> str:
    if question_id not in used:
        return question_id
    candidate = f"{question_id}__2"
    suffix = 2
    while candidate in used:
        suffix += 1
        candidate = f"{question_id}__{suffix}"
    return candidate
