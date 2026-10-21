"""Group compatible normalized decision records without changing their labels."""

from __future__ import annotations

import copy
import gc
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

Fits = Callable[[Mapping[str, Mapping[str, Any]]], bool]
# Optional ``fits.many``: ``fits`` of each mapping in a batch, None where it raises.
FitsMany = Callable[[Sequence[Mapping[str, Mapping[str, Any]]]], Sequence[bool | None]]


def group_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_questions: int,
    fits: Fits | None = None,
) -> list[dict[str, Any]]:
    """Combine only records with identical serving context and split without loss.

    Records sharing ``(source, group, family, state, instructions, images)`` are
    merged in input order and split into canvases of at most ``max_questions``
    questions. A record that is alone in its group and already fits is returned
    as a shallow copy (nested ``questions``/``labels``/metadata are shared with
    the input, so do not mutate them in place); merged or split canvases are
    built from deep copies.

    ``fits`` (optional) takes a canvas's ``{question id: question}`` mapping and
    says whether its answer template fits one canvas. With it, a canvas closes
    when it reaches ``max_questions`` or when the next question would not fit,
    so groups that would overflow are split instead; a single question that does
    not fit alone still gets its own canvas (and is dropped at canvas build).
    Without it, grouping counts questions only (the historical behaviour).
    When ``fits`` has a ``many`` method (a batched ``fits`` returning None where
    ``fits`` would raise), the canvases ``fits`` would be asked about are measured
    in batches up front: every lone record in one call, then the greedy packing
    of all other groups in rounds (one call per round, each round the next
    canvas of every group). The output is the same as calling ``fits`` on each.
    The cyclic garbage collector is paused while grouping runs.
    """
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")
    # Grouping allocates millions of small acyclic containers next to every
    # loaded record; the cyclic collector's full passes over that heap cost more
    # than the grouping itself, and it has nothing to collect here.
    collecting = gc.isenabled()
    gc.disable()
    try:
        return _group_records(records, max_questions, fits)
    finally:
        if collecting:
            gc.enable()


def _group_records(
    records: Sequence[Mapping[str, Any]],
    max_questions: int,
    fits: Fits | None,
) -> list[dict[str, Any]]:
    # Plain dicts keep insertion order; the key holds the original state and
    # instruction strings by reference, so grouping costs no extra copy of the
    # state text and dict equality (not the hash alone) decides every merge.
    groups: dict[_GroupKey, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(_key(record), []).append(record)
    group_list = list(groups.values())
    many: FitsMany | None = getattr(fits, "many", None) if fits is not None else None
    alone = _alone_fits(group_list, max_questions, many) if many else {}
    merges = _merge_fits(group_list, max_questions, many, alone) if many else {}
    result: list[dict[str, Any]] = []
    for index, group in enumerate(group_list):
        result.extend(
            _split_group(
                group, max_questions, fits, alone.get(index), merges.get(index)
            )
        )
    return result


def _alone_fits(
    groups: Iterable[Sequence[Mapping[str, Any]]],
    max_questions: int,
    many: FitsMany,
) -> dict[int, bool]:
    """``fits`` of every record alone in its group that ``_split_group`` measures whole.

    Keyed by group position. Covers exactly the records that pass the checks
    ``_split_group`` makes before its whole-record ``fits`` call; a record whose
    batched result is None is left out, so ``_split_group`` calls ``fits`` on it
    (and raises there, in input order).
    """
    positions: list[int] = []
    batch: list[Mapping[str, Mapping[str, Any]]] = []
    for position, group in enumerate(groups):
        if len(group) != 1:
            continue
        questions = group[0].get("questions")
        labels = group[0].get("labels")
        if (
            isinstance(questions, Mapping)
            and isinstance(labels, Mapping)
            and questions
            and set(questions) == set(labels)
            and len(questions) <= max_questions
        ):
            positions.append(position)
            batch.append(questions)
    if not batch:
        return {}
    results = many(batch)
    if len(results) != len(batch):
        raise ValueError("fits.many must return one result per canvas")
    return {
        position: fit
        for position, fit in zip(positions, results, strict=True)
        if fit is not None
    }


# A planned packing decision: (stream index of the canvas's first question,
# stream index of the question to add) -> (fits result, output id it was added as).
MergePlan = dict[tuple[int, int], tuple[bool, str]]


def _merge_fits(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    max_questions: int,
    many: FitsMany,
    alone: Mapping[int, bool],
) -> dict[int, MergePlan]:
    """The ``fits`` answers ``_split_group``'s packing loop asks for, batched.

    Replays the loop's greedy packing over each group's question stream (records
    in order, questions in order): a canvas starting at stream index ``s`` asks
    ``fits`` about its questions ``s..k`` for each next question ``k`` while it
    holds fewer than ``max_questions``, with the output ids the loop assigns.
    All groups' prefixes of their current canvas go into one ``many`` call per
    round; a canvas ends at the first prefix that does not fit (the next canvas
    starts there) or at ``max_questions``. Prefixes past the end of a canvas are
    measured but never used. A group stops being planned at a None result, and
    groups with a record the loop would reject are not planned, so the loop
    calls ``fits`` itself wherever the plan has no answer.
    """
    streams: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
    for position, group in enumerate(groups):
        if len(group) == 1:
            questions = group[0].get("questions")
            whole = alone.get(position)
            if not isinstance(questions, Mapping) or not (
                whole is False or (whole is None and len(questions) > max_questions)
            ):
                continue
        stream = _question_stream(group)
        if stream is not None and len(stream) > 1:
            streams[position] = stream
    plans: dict[int, MergePlan] = {position: {} for position in streams}
    starts = dict.fromkeys(streams, 0)
    while starts:
        batch: list[dict[str, Mapping[str, Any]]] = []
        asked: list[tuple[int, int, int, str]] = []
        for position, start in starts.items():
            stream = streams[position]
            used: set[str] = set()
            canvas: dict[str, Mapping[str, Any]] = {}
            for index in range(start, min(start + max_questions, len(stream))):
                question_id, question = stream[index]
                output_id = _unique_id(question_id, used)
                used.add(output_id)
                canvas[output_id] = question
                if index > start:
                    batch.append(dict(canvas))
                    asked.append((position, start, index, output_id))
        results = many(batch) if batch else []
        if len(results) != len(batch):
            raise ValueError("fits.many must return one result per canvas")
        next_starts: dict[int, int] = {}
        stopped: set[int] = set()
        for (position, start, index, output_id), fit in zip(
            asked, results, strict=True
        ):
            if position in stopped:
                continue
            if fit is None:
                stopped.add(position)
                continue
            plans[position][(start, index)] = (fit, output_id)
            if not fit:
                next_starts[position] = index
                stopped.add(position)
        for position, start in starts.items():
            full = start + max_questions
            if position not in stopped and full < len(streams[position]):
                next_starts[position] = full
        starts = next_starts
    return {position: plan for position, plan in plans.items() if plan}


def _question_stream(
    group: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]] | None:
    """(question id, question) in packing order, or None if the loop would raise."""
    stream: list[tuple[str, Mapping[str, Any]]] = []
    seen_ids: set[str] = set()
    for position, record in enumerate(group):
        questions = record.get("questions")
        labels = record.get("labels")
        record_id = record.get("id")
        if record_id is None:
            record_id = f"{record.get('group')}#record{position}"
        elif not isinstance(record_id, str) or not record_id:
            return None
        if (
            not isinstance(questions, Mapping)
            or not isinstance(labels, Mapping)
            or not questions
            or record_id in seen_ids
            or set(questions) != set(labels)
        ):
            return None
        seen_ids.add(record_id)
        for question_id, question in questions.items():
            if not isinstance(question_id, str) or not question_id:
                return None
            stream.append((question_id, question))
    return stream


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
    records: Sequence[Mapping[str, Any]],
    max_questions: int,
    fits: Fits | None = None,
    alone_fits: bool | None = None,
    plan: MergePlan | None = None,
) -> list[dict[str, Any]]:
    """Canvases of one group.

    ``alone_fits``: a lone record's precomputed ``fits``; ``plan``: precomputed
    packing answers from ``_merge_fits``, used only where the output id agrees.
    """
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
        if len(questions) <= max_questions and (
            fits is None or (fits(questions) if alone_fits is None else alone_fits)
        ):
            # Unchanged records are returned as a new top-level dict that shares
            # its nested values with the input; deep-copying every record
            # dominated preprocessing time and doubled label memory, and the
            # canvas pipeline deep-copies again in ``permute_record``.
            return [dict(record)]
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    used: set[str] = set()
    seen_ids: set[str] = set()
    stream_index = -1  # position of the question in the group's packing order
    canvas_start = 0  # stream index of the current canvas's first question
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
            stream_index += 1
            output_id = None
            if current is not None and fits is not None:
                if len(current["questions"]) < max_questions:
                    output_id = _unique_id(question_id, used)
                    planned = (
                        None if plan is None else plan.get((canvas_start, stream_index))
                    )
                    if planned is not None and planned[1] == output_id:
                        fit = planned[0]
                    else:
                        fit = fits({**current["questions"], output_id: question})
                    if not fit:
                        output_id = None
                        current = None
            if current is None or len(current["questions"]) == max_questions:
                current = _new_chunk(records[0], len(chunks))
                chunks.append(current)
                used = set()
                output_id = None
                canvas_start = stream_index
            if output_id is None:
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
