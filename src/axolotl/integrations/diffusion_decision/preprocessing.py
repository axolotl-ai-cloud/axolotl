"""Build model-tokenized decision canvases from validated source records."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from .adapters.jsonl import normalize_jsonl
from .prompts import decision_prompt_ids
from .readers.base import validate_canvas
from .records import DecisionCanvas, OrdinalMetadata
from .slots import SlotPlan
from .template import (
    SchemaError,
    parse_decision_schema,
    resolve_template,
    system_text,
)


def build_decision_canvas(
    tokenizer,
    record: Mapping[str, Any],
    prompt_ids: Sequence[int] | None = None,
    *,
    scaffold_ids: Sequence[int],
    turn_close_id: int,
    pad_id: int,
    vocab_size: int,
    width: int = 128,
    seed: int = 0,
    steps: int = 1,
    noise_kind: str = "uniform",
    mask_token_id: int | None = None,
    slot_plan: SlotPlan | None = None,
    thought_open_ids: Sequence[int] = (),
    thought_close_ids: Sequence[int] = (),
    include_ordinal_metadata: bool = False,
    codebook: str = "vendored26",
    prevalidated_record: bool = False,
) -> DecisionCanvas:
    if steps < 1 or width < 2 or vocab_size < 2:
        raise ValueError("steps, canvas width, and vocabulary must be positive")
    if noise_kind not in {"uniform", "absorbing"}:
        raise ValueError("unknown noise kind")
    if noise_kind == "absorbing" and (
        mask_token_id is None or not 0 <= mask_token_id < vocab_size
    ):
        raise ValueError("absorbing noise requires a valid mask_token_id")
    normalized = (
        dict(record)
        if prevalidated_record
        else normalize_jsonl(record, codebook=codebook)
    )
    schema = parse_decision_schema(
        {
            "instructions": normalized.get("instructions"),
            "questions": [
                {**question, "id": key}
                for key, question in normalized["questions"].items()
            ],
        },
        codebook=codebook,
    )
    if prompt_ids is None:
        prompt_ids = decision_prompt_ids(
            tokenizer, system_text(schema), normalized["state"]
        )
    prompt_ids = tuple(prompt_ids)
    plan = _slot_plan(slot_plan)
    prompt_slot_mask: tuple[bool, ...] = ()
    if plan is not None and plan.placement == "prompt":
        prompt_ids = (*plan.ids, *prompt_ids)
        prompt_slot_mask = (True,) * len(plan.ids) + (False,) * (
            len(prompt_ids) - len(plan.ids)
        )
    head = tuple(scaffold_ids)
    if plan is not None and plan.placement == "thought":
        head = (*head, *thought_open_ids, *plan.ids, *thought_close_ids)
    pad_slots = () if plan is None or plan.placement != "after_turn" else plan.ids
    base, slots = resolve_template(
        tokenizer,
        schema["questions"],
        head=head,
        fmt=schema["format"],
        width=width - len(pad_slots),
    )
    ids = list(base) + [turn_close_id]
    slot_positions: tuple[int, ...] = ()
    if plan is not None:
        if plan.placement == "thought":
            start = len(scaffold_ids) + len(thought_open_ids)
            slot_positions = tuple(range(start, start + len(plan.ids)))
        elif plan.placement == "after_turn":
            start = len(ids)
            slot_positions = tuple(range(start, start + len(plan.ids)))
            ids.extend(plan.ids)
    if len(ids) > width:
        raise SchemaError("decision canvas exceeds width")
    ids.extend([pad_id] * (width - len(ids)))
    positions = tuple(slot["pos"] for slot in slots)
    generator = random.Random(seed)  # nosec B311 - Reproduce pinned serving noise.
    for position in positions:
        if noise_kind == "uniform":
            ids[position] = generator.randrange(vocab_size)
        else:
            assert mask_token_id is not None
            ids[position] = mask_token_id
    result = DecisionCanvas(
        prompt_ids=tuple(prompt_ids),
        canvas_ids=tuple(ids),
        label_positions=positions,
        allowed_ids=tuple(tuple(slot["label_ids"]) for slot in slots),
        question_ids=tuple(question["id"] for question in schema["questions"]),
        targets=tuple(
            normalized["labels"][question["id"]] for question in schema["questions"]
        ),
        pinned_mask=tuple(
            (
                steps > 1
                and index not in positions
                and not (
                    plan is not None
                    and index in slot_positions
                    and plan.update_mask[slot_positions.index(index)]
                )
            )
            or (
                plan is not None
                and index in slot_positions
                and plan.pinned_mask[slot_positions.index(index)]
            )
            for index in range(width)
        ),
        semantic_mask=(True,) * width,
        slot_mask=tuple(index in slot_positions for index in range(width)),
        template_length=len(base),
        prompt_slot_mask=prompt_slot_mask,
        ordinal_metadata=(
            _ordinal_metadata(
                normalized, tuple(tuple(slot["label_ids"]) for slot in slots)
            )
            if include_ordinal_metadata
            else ()
        ),
    )
    validate_canvas(result)
    if any(
        token >= vocab_size
        for tokens in (result.prompt_ids, result.canvas_ids, *result.allowed_ids)
        for token in tokens
    ):
        raise ValueError("decision token ID exceeds the model vocabulary")
    return result


def _slot_plan(slot_plan: SlotPlan | None) -> SlotPlan | None:
    if slot_plan is None:
        return None
    widths = {
        len(slot_plan.ids),
        len(slot_plan.pinned_mask),
        len(slot_plan.update_mask),
        len(slot_plan.loss_mask),
    }
    if len(widths) != 1:
        raise ValueError("slot plan masks must match the number of slot IDs")
    if any(slot_plan.loss_mask):
        raise ValueError("decision slots cannot carry label loss")
    if any(
        update and pinned
        for update, pinned in zip(
            slot_plan.update_mask, slot_plan.pinned_mask, strict=True
        )
    ):
        raise ValueError("pinned decision slots cannot be updateable")
    if slot_plan.placement == "none":
        if slot_plan.ids:
            raise ValueError("a none slot plan cannot contain slots")
        return None
    if slot_plan.placement not in {"thought", "after_turn", "prompt"}:
        raise ValueError(f"unknown slot placement: {slot_plan.placement!r}")
    return slot_plan


def _ordinal_metadata(
    normalized: Mapping[str, Any], allowed_ids: tuple[tuple[int, ...], ...]
) -> tuple[OrdinalMetadata | None, ...]:
    questions = normalized["questions"]
    result: list[OrdinalMetadata | None] = []
    for (question_id, question), candidates in zip(
        questions.items(), allowed_ids, strict=True
    ):
        del question_id
        if question.get("type") != "score":
            result.append(None)
            continue
        levels = question.get("levels")
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            raise ValueError("score ordinal metadata requires source-ordered levels")
        if len(levels) != len(candidates):
            raise ValueError("score levels must align with template candidates")
        rendered_levels = tuple(str(level) for level in levels)
        if any(not level for level in rendered_levels):
            raise ValueError("score levels must be nonempty")
        result.append(
            OrdinalMetadata(
                levels=rendered_levels,
                source_ids=tuple(str(index) for index in range(len(candidates))),
                candidate_ranks=tuple(range(len(candidates))),
            )
        )
    return tuple(result)


def ordinal_metadata_for_canvas(
    record: Mapping[str, Any], canvas: DecisionCanvas, *, codebook: str = "vendored26"
) -> tuple[OrdinalMetadata | None, ...]:
    """Derive score ranks from the normalized source criteria, never token IDs."""
    normalized = normalize_jsonl(record, codebook=codebook)
    result = _ordinal_metadata(
        normalized, tuple(tuple(ids) for ids in canvas.allowed_ids)
    )
    if tuple(normalized["questions"]) != tuple(canvas.question_ids):
        raise ValueError(
            "normalized score questions must align with the prepared canvas"
        )
    return result
