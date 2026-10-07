"""Pure count sampling and maximum-canvas slot materialization helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Final

from .records import DecisionCanvas
from .slots import SlotPlan

UNIFORM_INCLUSIVE_V1: Final = "uniform_inclusive_v1"


@dataclass(frozen=True)
class DecisionDraw:
    """One pre-sharding logical decision draw with a stable sampling identity."""

    index: int
    epoch: int
    global_draw_ordinal: int

    def __post_init__(self) -> None:
        _validate_nonnegative_int(self.index, "index")
        _validate_nonnegative_int(self.epoch, "epoch")
        _validate_nonnegative_int(self.global_draw_ordinal, "global_draw_ordinal")


def sample_slot_count(
    *,
    seed: int,
    epoch: int,
    global_draw_ordinal: int,
    max_slots: int,
    min_slots: int = 0,
    policy: str = UNIFORM_INCLUSIVE_V1,
) -> int:
    """Return a stateless count for one logical training draw."""
    _validate_nonnegative_int(seed, "seed")
    _validate_nonnegative_int(epoch, "epoch")
    _validate_nonnegative_int(global_draw_ordinal, "global_draw_ordinal")
    _validate_nonnegative_int(min_slots, "min_slots")
    _validate_nonnegative_int(max_slots, "max_slots")
    if min_slots > max_slots:
        raise ValueError("min_slots must not exceed max_slots")
    if policy != UNIFORM_INCLUSIVE_V1:
        raise ValueError(f"unsupported slot-count policy: {policy!r}")

    population = max_slots - min_slots + 1
    if population > 1 << 256:
        raise ValueError("slot-count population exceeds the sampler domain")
    limit = (1 << 256) - ((1 << 256) % population)
    counter = 0
    while True:
        payload = f"{policy}:{seed}:{epoch}:{global_draw_ordinal}:{counter}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < limit:
            return min_slots + (value % population)
        counter += 1


def project_slot_plan(maximum: SlotPlan, count: int) -> SlotPlan:
    """Keep the stable prefix of a validated maximum-count plan."""
    _validate_nonnegative_int(count, "count")
    maximum_count = _validate_slot_plan(maximum)
    if count > maximum_count:
        raise ValueError("slot count exceeds the maximum slot plan")
    if count == maximum_count:
        return maximum
    return SlotPlan(
        ids=maximum.ids[:count],
        placement=maximum.placement,
        pinned_mask=maximum.pinned_mask[:count],
        update_mask=maximum.update_mask[:count],
        loss_mask=maximum.loss_mask[:count],
        trainable_token_ids=maximum.trainable_token_ids[:count],
    )


def project_max_canvas(
    maximum_canvas: DecisionCanvas,
    maximum_plan: SlotPlan,
    count: int,
    *,
    pad_token_id: int,
    padding_pinned: bool,
) -> DecisionCanvas:
    """Materialize a smaller concrete slot count from a maximum-count canvas."""
    _validate_nonnegative_int(count, "count")
    _validate_nonnegative_int(pad_token_id, "pad_token_id")
    if not isinstance(padding_pinned, bool):
        raise ValueError("padding_pinned must be a bool")
    maximum_count = _validate_slot_plan(maximum_plan)
    if count > maximum_count:
        raise ValueError("slot count exceeds the maximum slot plan")
    if count == maximum_count:
        return maximum_canvas

    if maximum_plan.placement == "none":
        raise ValueError("a none slot plan has no count to project")
    if maximum_plan.placement == "prompt":
        return _project_prompt(maximum_canvas, maximum_plan, count)

    positions = _canvas_slot_positions(maximum_canvas, maximum_plan)
    remove = positions[count:]
    if maximum_plan.placement == "thought":
        if any(
            position <= positions[-1] for position in maximum_canvas.label_positions
        ):
            raise ValueError("thought slots must precede every label position")
        label_positions = tuple(
            position - len(remove) for position in maximum_canvas.label_positions
        )
        template_length = maximum_canvas.template_length - len(remove)
    elif maximum_plan.placement == "after_turn":
        label_positions = tuple(maximum_canvas.label_positions)
        template_length = maximum_canvas.template_length
    else:
        raise ValueError(f"unsupported slot placement: {maximum_plan.placement!r}")

    return replace(
        maximum_canvas,
        canvas_ids=_delete_and_extend(maximum_canvas.canvas_ids, remove, pad_token_id),
        label_positions=label_positions,
        pinned_mask=_delete_and_extend(
            maximum_canvas.pinned_mask, remove, padding_pinned
        ),
        semantic_mask=_delete_and_extend(maximum_canvas.semantic_mask, remove, True),
        slot_mask=_delete_and_extend(maximum_canvas.slot_mask, remove, False),
        template_length=template_length,
    )


def _project_prompt(
    canvas: DecisionCanvas, maximum: SlotPlan, count: int
) -> DecisionCanvas:
    maximum_count = len(maximum.ids)
    if len(canvas.prompt_slot_mask) != len(canvas.prompt_ids):
        raise ValueError("prompt slots require a complete prompt_slot_mask")
    if tuple(canvas.prompt_ids[:maximum_count]) != maximum.ids:
        raise ValueError("prompt slot IDs must be the leading prompt IDs")
    if any(not value for value in canvas.prompt_slot_mask[:maximum_count]) or any(
        canvas.prompt_slot_mask[maximum_count:]
    ):
        raise ValueError("prompt slot mask must mark only the leading slot IDs")
    if any(canvas.slot_mask):
        raise ValueError("prompt slots must not occupy canvas positions")
    return replace(
        canvas,
        prompt_ids=(
            *maximum.ids[:count],
            *canvas.prompt_ids[maximum_count:],
        ),
        prompt_slot_mask=(True,) * count
        + (False,) * (len(canvas.prompt_ids) - maximum_count),
    )


def _canvas_slot_positions(
    canvas: DecisionCanvas, maximum: SlotPlan
) -> tuple[int, ...]:
    positions = tuple(index for index, value in enumerate(canvas.slot_mask) if value)
    if len(positions) != len(maximum.ids):
        raise ValueError("canvas slot mask does not match the maximum slot plan")
    if positions != tuple(range(positions[0], positions[0] + len(positions))):
        raise ValueError("canvas slot positions must be contiguous")
    if tuple(canvas.canvas_ids[index] for index in positions) != maximum.ids:
        raise ValueError("canvas slot IDs do not match the maximum slot plan")
    return positions


def _delete_and_extend(values, remove: tuple[int, ...], padding_value):
    removed = set(remove)
    kept = tuple(value for index, value in enumerate(values) if index not in removed)
    return (*kept, *(padding_value for _ in remove))


def _validate_slot_plan(plan: SlotPlan) -> int:
    count = len(plan.ids)
    if plan.placement not in {"thought", "after_turn", "prompt"}:
        raise ValueError("slot sampling requires a materialized slot placement")
    masks = (plan.pinned_mask, plan.update_mask, plan.loss_mask)
    if any(len(mask) != count for mask in masks):
        raise ValueError("slot plan masks must match slot IDs")
    if any(plan.loss_mask):
        raise ValueError("sampled decision slots cannot carry loss")
    if any(
        pinned and update
        for pinned, update in zip(plan.pinned_mask, plan.update_mask, strict=True)
    ):
        raise ValueError("pinned sampled slots cannot be updateable")
    if plan.trainable_token_ids != plan.ids[: len(plan.trainable_token_ids)]:
        raise ValueError("trainable slot IDs must be a stable slot prefix")
    return count


def _validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
