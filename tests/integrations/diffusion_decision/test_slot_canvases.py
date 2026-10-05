"""Slot plans affect only declared decision-canvas regions."""

import pytest

from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.slots import SlotInit
from axolotl.integrations.diffusion_decision.template import SchemaError
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


def _spec(noise: DiffusionNoise) -> DiffusionSpec:
    return DiffusionSpec(
        noise=noise,
        layout=DiffusionLayout.FULL_SEQUENCE,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=False,
        max_canvas=128,
        max_context=1024,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=(
            MaskTokenPolicy.NONE
            if noise is DiffusionNoise.UNIFORM
            else MaskTokenPolicy.MODEL
        ),
        default_time_weighting=TimeWeighting.NONE,
        objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
        generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        reduction_scope=ReductionScope.MICROBATCH,
    )


def _record():
    return {
        "id": "record",
        "source": "test",
        "group": "test",
        "state": "state",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick.",
                "options": ["one", "two"],
            }
        },
        "labels": {"q": {"kind": "hard", "gold_idx": 0}},
    }


def _plan(mode, *, count=2, ids=(), noise=DiffusionNoise.UNIFORM):
    return SlotInit(
        mode,
        token_ids=ids,
        num_slots=count,
        vocab_size=256,
        pad_id=0,
        spec=_spec(noise),
        mask_token_id=9,
    ).build(seed=17 if mode == "free" and noise is DiffusionNoise.UNIFORM else None)


def _canvas(plan=None, *, steps=1, noise_kind="uniform", **kwargs):
    arguments = {
        "width": 128,
        "seed": 23,
        "steps": steps,
        "noise_kind": noise_kind,
        "mask_token_id": 9 if noise_kind == "absorbing" else None,
        "slot_plan": plan,
        "thought_open_ids": (70, 71),
        "thought_close_ids": (72,),
    }
    arguments.update(kwargs)
    return build_decision_canvas(
        CharacterTokenizer(),
        _record(),
        prompt_ids=(99,),
        scaffold_ids=(),
        turn_close_id=106,
        pad_id=0,
        vocab_size=256,
        **arguments,
    )


def test_none_slot_plan_keeps_existing_canvas_bytes_even_with_explicit_tags():
    baseline = _canvas()
    none = _canvas(_plan("none", count=0))

    assert none == baseline


@pytest.mark.parametrize(
    ("mode", "ids", "placement", "expected_slots"),
    [
        ("pad", (), "after_turn", (0, 0)),
        ("pinned", (7,), "thought", (7, 7)),
        ("pinned", (7, 8), "thought", (7, 8)),
        ("learned", (7, 8), "thought", (7, 8)),
        ("prompt", (7, 8), "prompt", (7, 8)),
    ],
)
def test_slot_modes_preserve_labels_and_mark_only_canvas_slots(
    mode, ids, placement, expected_slots
):
    plan = _plan(mode, ids=ids)
    canvas = _canvas(plan)
    baseline = _canvas()

    assert (
        canvas.canvas_ids[canvas.label_positions[0]]
        == baseline.canvas_ids[baseline.label_positions[0]]
    )
    assert canvas.allowed_ids == baseline.allowed_ids
    assert canvas.targets == baseline.targets
    assert len(canvas.canvas_ids) == len(baseline.canvas_ids) == 128
    if placement == "thought":
        assert canvas.canvas_ids[:5] == (70, 71, *expected_slots, 72)
        assert [index for index, value in enumerate(canvas.slot_mask) if value] == [
            2,
            3,
        ]
        assert canvas.pinned_mask[2:4] == (True, True)
    elif placement == "after_turn":
        assert canvas.canvas_ids[canvas.template_length] == 106
        assert (
            canvas.canvas_ids[canvas.template_length + 1 : canvas.template_length + 3]
            == expected_slots
        )
        assert [index for index, value in enumerate(canvas.slot_mask) if value] == [
            canvas.template_length + 1,
            canvas.template_length + 2,
        ]
        assert all(
            canvas.pinned_mask[index]
            for index, value in enumerate(canvas.slot_mask)
            if value
        )
    else:
        assert canvas.prompt_ids == (*expected_slots, 99)
        assert canvas.prompt_slot_mask == (True, True, False)
        assert not any(canvas.slot_mask)
        assert canvas.canvas_ids == baseline.canvas_ids
    if placement != "prompt":
        assert canvas.prompt_slot_mask == ()
    assert all(canvas.semantic_mask)
    assert all(not canvas.slot_mask[position] for position in canvas.label_positions)


def test_mask_and_free_slots_follow_noise_specific_initialization_and_pin_policy():
    masked = _canvas(
        _plan("mask", noise=DiffusionNoise.ABSORBING), noise_kind="absorbing"
    )
    free_plan = _plan("free")
    free = _canvas(free_plan)

    assert masked.canvas_ids[:5] == (70, 71, 9, 9, 72)
    assert masked.pinned_mask[2:4] == (True, True)
    assert masked.canvas_ids[masked.label_positions[0]] == 9
    assert free.slot_mask[2:4] == (True, True)
    assert free.pinned_mask[2:4] == (False, False)
    assert free.canvas_ids[2:4] == free_plan.ids


def test_fixed_slots_stay_pinned_at_one_read_and_non_slot_behavior_stays_source_faithful():
    pinned = _canvas(_plan("pinned", ids=(7,)), steps=1)
    repeated = _canvas(_plan("pinned", ids=(7,)), steps=2)

    assert pinned.pinned_mask[2:4] == (True, True)
    assert pinned.pinned_mask[pinned.label_positions[0]] is False
    assert repeated.pinned_mask[pinned.label_positions[0]] is False
    assert all(
        repeated.pinned_mask[index] == (index not in repeated.label_positions)
        for index in range(128)
    )


def test_distinct_pinned_slots_match_learned_canvas_coordinates_without_trainable_rows():
    pinned_plan = _plan("pinned", ids=(7, 8))
    learned_plan = _plan("learned", ids=(7, 8))
    pinned = _canvas(pinned_plan, steps=2)
    learned = _canvas(learned_plan, steps=2)

    assert pinned.canvas_ids == learned.canvas_ids
    assert pinned.label_positions == learned.label_positions
    assert pinned.slot_mask == learned.slot_mask
    assert pinned.pinned_mask == learned.pinned_mask
    assert pinned_plan.trainable_token_ids == ()
    assert learned_plan.trainable_token_ids == (7, 8)


def test_slot_budget_overflow_raises_schema_error_without_truncation():
    with pytest.raises(SchemaError, match="canvas"):
        _canvas(_plan("learned", ids=(7, 8)), width=7)
