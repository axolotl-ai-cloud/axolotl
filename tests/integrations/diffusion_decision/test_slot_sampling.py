"""Pure maximum-canvas projection tests for sampled decision slot counts."""

from __future__ import annotations

import random
from typing import cast

import pytest

from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.slot_sampling import (
    UNIFORM_INCLUSIVE_V1,
    DecisionDraw,
    project_max_canvas,
    project_slot_plan,
    sample_slot_count,
)
from axolotl.integrations.diffusion_decision.slots import SlotInit, SlotMode, SlotPlan
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


def _maximum_plan(mode: str, noise: DiffusionNoise) -> SlotPlan:
    ids = {
        "pinned": (7,),
        "learned": (7, 8, 10),
        "prompt": (7, 8, 10),
    }.get(mode, ())
    return SlotInit(
        cast(SlotMode, mode),
        token_ids=ids,
        num_slots=3,
        vocab_size=256,
        pad_id=0,
        spec=_spec(noise),
        mask_token_id=9,
    ).build(seed=17 if mode == "free" and noise is DiffusionNoise.UNIFORM else None)


def _canvas(plan: SlotPlan, *, noise: DiffusionNoise, width: int = 128):
    return build_decision_canvas(
        CharacterTokenizer(),
        _record(),
        prompt_ids=(99,),
        scaffold_ids=(),
        turn_close_id=106,
        pad_id=0,
        vocab_size=256,
        width=width,
        seed=23,
        steps=2,
        noise_kind="absorbing" if noise is DiffusionNoise.ABSORBING else "uniform",
        mask_token_id=9 if noise is DiffusionNoise.ABSORBING else None,
        slot_plan=plan,
        thought_open_ids=(70, 71),
        thought_close_ids=(72,),
    )


@pytest.mark.parametrize(
    ("mode", "noise"),
    [
        ("pad", DiffusionNoise.UNIFORM),
        ("pinned", DiffusionNoise.UNIFORM),
        ("learned", DiffusionNoise.UNIFORM),
        ("prompt", DiffusionNoise.UNIFORM),
        ("mask", DiffusionNoise.ABSORBING),
        ("free", DiffusionNoise.UNIFORM),
    ],
)
@pytest.mark.parametrize("count", [0, 1, 3])
def test_projection_matches_direct_builder_for_all_supported_placements(
    mode: str, noise: DiffusionNoise, count: int
):
    maximum_plan = _maximum_plan(mode, noise)
    maximum_canvas = _canvas(maximum_plan, noise=noise)

    projected_plan = project_slot_plan(maximum_plan, count)
    projected_canvas = project_max_canvas(
        maximum_canvas,
        maximum_plan,
        count,
        pad_token_id=0,
        padding_pinned=True,
    )
    direct = _canvas(projected_plan, noise=noise)

    assert projected_canvas == direct
    assert projected_canvas.targets == maximum_canvas.targets
    assert projected_canvas.allowed_ids == maximum_canvas.allowed_ids
    assert projected_canvas.question_ids == maximum_canvas.question_ids
    assert all(
        not projected_canvas.slot_mask[position]
        for position in projected_canvas.label_positions
    )
    assert all(not value for value in projected_plan.loss_mask)
    if count == len(maximum_plan.ids):
        assert projected_plan is maximum_plan
        assert projected_canvas is maximum_canvas


def test_projection_preserves_free_slot_update_mask_and_typed_metadata():
    maximum_plan = _maximum_plan("free", DiffusionNoise.UNIFORM)
    maximum_canvas = _canvas(maximum_plan, noise=DiffusionNoise.UNIFORM)

    plan = project_slot_plan(maximum_plan, 1)
    canvas = project_max_canvas(
        maximum_canvas,
        maximum_plan,
        1,
        pad_token_id=0,
        padding_pinned=True,
    )

    assert plan.update_mask == (True,)
    assert plan.pinned_mask == (False,)
    assert canvas.ordinal_metadata == maximum_canvas.ordinal_metadata
    slot_position = canvas.slot_mask.index(True)
    assert canvas.pinned_mask[slot_position] is False
    assert canvas.canvas_ids[slot_position] == maximum_plan.ids[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True, "epoch": 0, "global_draw_ordinal": 0, "max_slots": 3},
        {"seed": 1, "epoch": -1, "global_draw_ordinal": 0, "max_slots": 3},
        {
            "seed": 1,
            "epoch": 0,
            "global_draw_ordinal": 0,
            "max_slots": 3,
            "min_slots": 4,
        },
        {
            "seed": 1,
            "epoch": 0,
            "global_draw_ordinal": 0,
            "max_slots": 3,
            "policy": "other",
        },
        {"seed": 1, "epoch": 0, "global_draw_ordinal": 0, "max_slots": 1 << 256},
    ],
)
def test_slot_count_sampler_rejects_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        sample_slot_count(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index": -1, "epoch": 0, "global_draw_ordinal": 0},
        {"index": 0, "epoch": True, "global_draw_ordinal": 0},
        {"index": 0, "epoch": 0, "global_draw_ordinal": -1},
    ],
)
def test_decision_draw_rejects_invalid_stable_identity(kwargs):
    with pytest.raises(ValueError):
        DecisionDraw(**kwargs)


def test_slot_count_sampler_is_stateless_replayable_and_uniform(monkeypatch):
    def forbidden_random(*_args, **_kwargs):
        raise AssertionError("global RNG")

    monkeypatch.setattr(random, "Random", forbidden_random)

    first = [
        sample_slot_count(seed=41, epoch=2, global_draw_ordinal=index, max_slots=4)
        for index in range(10_000)
    ]
    second = [
        sample_slot_count(seed=41, epoch=2, global_draw_ordinal=index, max_slots=4)
        for index in range(10_000)
    ]
    changed_epoch = sample_slot_count(
        seed=41, epoch=3, global_draw_ordinal=0, max_slots=4
    )
    counts = [first.count(value) for value in range(5)]

    assert first == second
    assert 0 <= changed_epoch <= 4
    assert all(abs(value - 2_000) < 180 for value in counts)
    assert UNIFORM_INCLUSIVE_V1 == "uniform_inclusive_v1"


def test_projection_rejects_invalid_count_and_malformed_maximum_canvas():
    plan = _maximum_plan("learned", DiffusionNoise.UNIFORM)
    canvas = _canvas(plan, noise=DiffusionNoise.UNIFORM)

    with pytest.raises(ValueError, match="exceeds"):
        project_slot_plan(plan, 4)
    with pytest.raises(ValueError, match="exceeds"):
        project_max_canvas(canvas, plan, 4, pad_token_id=0, padding_pinned=True)
    malformed = SlotPlan(
        ids=plan.ids,
        placement=plan.placement,
        pinned_mask=plan.pinned_mask,
        update_mask=plan.update_mask,
        loss_mask=(True, False, False),
        trainable_token_ids=plan.trainable_token_ids,
    )
    with pytest.raises(ValueError, match="cannot carry loss"):
        project_max_canvas(canvas, malformed, 1, pad_token_id=0, padding_pinned=True)


@pytest.mark.parametrize("mode", ["learned", "pad"])
def test_projection_handles_exact_fit_maximum_canvases(mode: str):
    noise = DiffusionNoise.UNIFORM
    plan = _maximum_plan(mode, noise)
    padded = _canvas(plan, noise=noise)
    width = padded.template_length + 1 + (len(plan.ids) if mode == "pad" else 0)
    maximum = _canvas(plan, noise=noise, width=width)
    assert maximum.slot_mask[-1] or maximum.canvas_ids[-1] != 0

    projected = project_max_canvas(
        maximum, plan, 1, pad_token_id=0, padding_pinned=True
    )
    direct = _canvas(project_slot_plan(plan, 1), noise=noise, width=width)

    assert projected == direct
    assert projected.canvas_ids[-2:] == (0, 0)
    assert projected.slot_mask[-2:] == (False, False)
    assert projected.pinned_mask[-2:] == (True, True)
