"""Build model-tokenized decision canvases from validated source records."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from ._image_tokens import (
    NEMOTRON_VLM_IMAGE_START_ID,
    NEMOTRON_VLM_IMAGE_START_MARKER,
)
from .adapters.jsonl import normalize_jsonl
from .prompts import decision_prompt
from .readers.base import validate_canvas
from .records import DecisionCanvas, OrdinalMetadata
from .slots import SlotPlan
from .template import (
    SchemaError,
    answer_text,
    parse_decision_schema,
    resolve_template,
    system_text,
)

ANSWER_FORMATS = ("auto", "lines", "indexed")
_DEPENDENCY_KEYS = frozenset({"depends_on", "ask_if"})


def canvas_head(
    scaffold_ids: Sequence[int],
    plan: SlotPlan | None,
    thought_open_ids: Sequence[int] = (),
    thought_close_ids: Sequence[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The token run a canvas starts with and the slot ids appended after the turn close."""
    head = tuple(scaffold_ids)
    if plan is not None and plan.placement == "thought":
        head = (*head, *thought_open_ids, *plan.ids, *thought_close_ids)
    pad_slots = () if plan is None or plan.placement != "after_turn" else plan.ids
    return head, tuple(pad_slots)


def template_fits(
    tokenizer,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    head: Sequence[int],
    pad_slots: int,
    width: int,
    answer_format: str,
    codebook: str = "vendored26",
) -> bool:
    """Whether these normalized questions' answer template fits one canvas.

    Measures the same base template ``build_decision_canvas`` resolves (every
    question at its first label, plus the turn close and any after-turn slots).
    Codebook labels are single tokens, so neither the question permutation nor
    label shuffling changes the length.
    """
    if answer_format not in ("lines", "indexed"):
        raise ValueError("template_fits needs an explicit lines or indexed format")
    text = _base_answer_text(questions, answer_format, codebook)
    length = len(head) + len(tokenizer.encode(text, add_special_tokens=False))
    return length + 1 + pad_slots <= width


def template_fits_many(
    tokenizer,
    batch: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    head: Sequence[int],
    pad_slots: int,
    width: int,
    answer_format: str,
    codebook: str = "vendored26",
    chunk_size: int = 4096,
) -> list[bool | None]:
    """``template_fits`` of each question mapping, tokenizing in batches.

    Each result equals ``template_fits`` of that mapping: the text is built by
    the same function, equal texts are encoded once, and a fast tokenizer's
    batch call runs the same encode pipeline per text as ``encode`` (which
    batches a single text). An entry is None where building or encoding its
    text raised; ``template_fits`` of that mapping raises the same error, so
    callers measure it with ``template_fits`` at the point the error belongs.
    """
    results: list[bool | None] = [None] * len(batch)
    if answer_format not in ("lines", "indexed"):
        return results  # template_fits raises the format error
    for start in range(0, len(batch), chunk_size):
        # Many records share one answer template; each distinct text is
        # encoded once (equal strings encode to equal ids).
        text_of: dict[int, str] = {}
        unique: dict[str, int] = {}
        for index in range(start, min(start + chunk_size, len(batch))):
            text = _base_answer_text_or_none(batch[index], answer_format, codebook)
            if text is not None:
                text_of[index] = text
                unique.setdefault(text, len(unique))
        if not unique:
            continue
        encoded = _encode_texts_or_none(tokenizer, list(unique))
        if encoded is None or len(encoded) != len(unique):
            continue
        fit = [len(head) + len(ids) + 1 + pad_slots <= width for ids in encoded]
        for index, text in text_of.items():
            results[index] = fit[unique[text]]
    return results


def _base_answer_text(
    questions: Mapping[str, Mapping[str, Any]], answer_format: str, codebook: str
) -> str:
    # Dependencies do not change the template; dropping them lets a partial
    # group be measured before the questions it depends on join it.
    schema = parse_decision_schema(
        {
            "questions": [
                {
                    **{k: v for k, v in question.items() if k not in _DEPENDENCY_KEYS},
                    "id": key,
                }
                for key, question in questions.items()
            ]
        },
        codebook=codebook,
    )
    qs = schema["questions"]
    return answer_text(qs, [0] * len(qs), answer_format)


def _base_answer_text_or_none(
    questions: Mapping[str, Mapping[str, Any]], answer_format: str, codebook: str
) -> str | None:
    try:
        return _base_answer_text(questions, answer_format, codebook)
    except Exception:  # noqa: BLE001  (template_fits re-raises it)
        return None


def _encode_texts_or_none(
    tokenizer, texts: Sequence[str]
) -> list[Sequence[int]] | None:
    try:
        return _encode_texts(tokenizer, texts)
    except Exception:  # noqa: BLE001  (template_fits re-raises it)
        return None


def _encode_texts(tokenizer, texts: Sequence[str]) -> list[Sequence[int]]:
    """``tokenizer.encode(text, add_special_tokens=False)`` of each text.

    A fast (Rust) tokenizer encodes the list in one call: transformers routes
    both ``encode(text)`` and ``tokenizer(texts)`` through ``_encode_plus`` and
    the backend's ``encode_batch``, which encodes each text independently
    (without padding or truncation here). Other tokenizers encode one by one.
    """
    if getattr(tokenizer, "is_fast", False) is True:
        return tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [tokenizer.encode(text, add_special_tokens=False) for text in texts]


def _reserves_image_markers(tokenizer) -> bool:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    return (
        callable(convert)
        and convert(NEMOTRON_VLM_IMAGE_START_MARKER) == NEMOTRON_VLM_IMAGE_START_ID
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
    max_image_size: int = 1400,
    answer_format: str = "auto",
) -> DecisionCanvas:
    check_canvas_arguments(
        answer_format=answer_format,
        steps=steps,
        width=width,
        vocab_size=vocab_size,
        noise_kind=noise_kind,
        mask_token_id=mask_token_id,
    )
    normalized = (
        dict(record)
        if prevalidated_record
        else normalize_jsonl(record, codebook=codebook)
    )
    schema = decision_schema(normalized, codebook=codebook, answer_format=answer_format)
    image_refs = tuple(normalized.get("images") or ())
    image_sizes: tuple[tuple[int, int], ...] = ()
    if prompt_ids is None:
        prompt_ids, image_sizes = decision_prompt(
            tokenizer,
            system_text(schema),
            normalized["state"],
            images=image_refs,
            max_image_size=max_image_size,
        )
    elif image_refs:
        raise ValueError("image records cannot use precomputed prompt_ids")
    plan = _slot_plan(slot_plan)
    head, pad_slots = canvas_head(
        scaffold_ids, plan, thought_open_ids, thought_close_ids
    )
    base, slots = resolve_template(
        tokenizer,
        schema["questions"],
        head=head,
        fmt=schema["format"],
        width=width - len(pad_slots),
    )
    return finish_decision_canvas(
        normalized,
        schema,
        prompt_ids,
        base,
        slots,
        plan=plan,
        scaffold_ids=scaffold_ids,
        turn_close_id=turn_close_id,
        pad_id=pad_id,
        vocab_size=vocab_size,
        width=width,
        seed=seed,
        steps=steps,
        noise_kind=noise_kind,
        mask_token_id=mask_token_id,
        thought_open_ids=thought_open_ids,
        include_ordinal_metadata=include_ordinal_metadata,
        image_refs=image_refs,
        image_sizes=image_sizes,
        image_markers_reserved=_reserves_image_markers(tokenizer),
    )


def check_canvas_arguments(
    *,
    answer_format: str,
    steps: int,
    width: int,
    vocab_size: int,
    noise_kind: str,
    mask_token_id: int | None,
) -> None:
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"unknown decision answer format {answer_format!r}")
    if steps < 1 or width < 2 or vocab_size < 2:
        raise ValueError("steps, canvas width, and vocabulary must be positive")
    if noise_kind not in {"uniform", "absorbing"}:
        raise ValueError("unknown noise kind")
    if noise_kind == "absorbing" and (
        mask_token_id is None or not 0 <= mask_token_id < vocab_size
    ):
        raise ValueError("absorbing noise requires a valid mask_token_id")


def decision_schema(
    normalized: Mapping[str, Any], *, codebook: str, answer_format: str
) -> dict[str, Any]:
    """The parsed schema of a normalized record, in its question order."""
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
    if answer_format != "auto":
        schema["format"] = answer_format
    return schema


def finish_decision_canvas(
    normalized: Mapping[str, Any],
    schema: Mapping[str, Any],
    prompt_ids: Sequence[int],
    base: Sequence[int],
    slots: Sequence[Mapping[str, Any]],
    *,
    plan: SlotPlan | None,
    scaffold_ids: Sequence[int],
    turn_close_id: int,
    pad_id: int,
    vocab_size: int,
    width: int,
    seed: int,
    steps: int,
    noise_kind: str,
    mask_token_id: int | None,
    thought_open_ids: Sequence[int] = (),
    include_ordinal_metadata: bool = False,
    trusted_prompt: bool = False,
    image_refs: Sequence[str] = (),
    image_sizes: Sequence[tuple[int, int]] = (),
    image_markers_reserved: bool = False,
) -> DecisionCanvas:
    """The canvas from a resolved prompt and answer template (``resolve_template``'s
    ``base`` and ``slots``); ``plan`` is already validated by ``_slot_plan``.

    ``trusted_prompt``: ``prompt_ids`` are known nonnegative ints (pre-tokenized
    spans), so the per-token type check is skipped and the vocabulary bound is
    checked with ``max``; the result and any error are the same.

    ``image_refs``/``image_sizes`` are the prompt's images (``decision_prompt``);
    ``image_markers_reserved`` is ``validate_canvas``'s flag for the tokenizer.
    """
    prompt_ids = tuple(prompt_ids)
    prompt_slot_mask: tuple[bool, ...] = ()
    if plan is not None and plan.placement == "prompt":
        prompt_ids = (*plan.ids, *prompt_ids)
        prompt_slot_mask = (True,) * len(plan.ids) + (False,) * (
            len(prompt_ids) - len(plan.ids)
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
        image_refs=tuple(image_refs),
        image_sizes=tuple(image_sizes),
    )
    validate_canvas(
        result,
        check_prompt_tokens=not trusted_prompt,
        image_markers_reserved=image_markers_reserved,
    )
    if (
        trusted_prompt and result.prompt_ids and max(result.prompt_ids) >= vocab_size
    ) or any(
        token >= vocab_size
        for tokens in (
            *(() if trusted_prompt else (result.prompt_ids,)),
            result.canvas_ids,
            *result.allowed_ids,
        )
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
