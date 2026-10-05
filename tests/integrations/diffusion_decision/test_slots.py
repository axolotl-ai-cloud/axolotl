"""Slot initialization contracts stay model-agnostic and pure."""

import random

import pytest

from axolotl.integrations.diffusion_decision.slots import SlotInit
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


def _spec(noise: DiffusionNoise) -> DiffusionSpec:
    return DiffusionSpec(
        noise=noise,
        layout=DiffusionLayout.FULL_SEQUENCE,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=noise is DiffusionNoise.UNIFORM,
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


def _init(mode, *, count=3, ids=(), noise=DiffusionNoise.UNIFORM, **kwargs):
    return SlotInit(
        mode,
        token_ids=ids,
        num_slots=count,
        vocab_size=32,
        pad_id=0,
        spec=_spec(noise),
        mask_token_id=9,
        **kwargs,
    )


def test_none_has_no_ids_and_no_slot_masks():
    slot = SlotInit()

    assert slot.ids == ()
    assert slot.build().ids == ()
    assert slot.build().placement == "none"


@pytest.mark.parametrize(
    ("mode", "ids", "placement", "expected", "trainable"),
    [
        ("pad", (), "after_turn", (0, 0, 0), ()),
        ("pinned", (7,), "thought", (7, 7, 7), ()),
        ("pinned", (7, 8, 10), "thought", (7, 8, 10), ()),
        ("learned", (7, 8, 9), "thought", (7, 8, 9), (7, 8, 9)),
        ("prompt", (7, 8, 9), "prompt", (7, 8, 9), (7, 8, 9)),
    ],
)
def test_fixed_slot_modes_expose_placement_pinning_and_no_loss(
    mode, ids, placement, expected, trainable
):
    plan = _init(mode, ids=ids).build()

    assert plan.ids == expected
    assert plan.placement == placement
    assert plan.pinned_mask == (True, True, True)
    assert plan.update_mask == (False, False, False)
    assert plan.loss_mask == (False, False, False)
    assert plan.trainable_token_ids == trainable


def test_mask_is_absorbing_only_and_never_updates_or_contributes_loss():
    plan = _init("mask", noise=DiffusionNoise.ABSORBING).build()

    assert plan.ids == (9, 9, 9)
    assert plan.placement == "thought"
    assert plan.pinned_mask == (True, True, True)
    assert plan.update_mask == (False, False, False)
    assert plan.loss_mask == (False, False, False)
    with pytest.raises(ValueError, match="absorbing"):
        _init("mask").build()


def test_free_uniform_is_seeded_and_free_absorbing_starts_masked():
    uniform = _init("free").build(seed=17)
    generator = random.Random(17)

    assert uniform.ids == tuple(generator.randrange(32) for _ in range(3))
    assert uniform.placement == "thought"
    assert uniform.pinned_mask == (False, False, False)
    assert uniform.update_mask == (True, True, True)
    assert uniform.loss_mask == (False, False, False)
    assert _init("free").build(seed=17) == uniform
    with pytest.raises(ValueError, match="explicit integer seed"):
        _init("free").build()
    with pytest.raises(ValueError, match="either seed or generator"):
        _init("free").build(seed=17, generator=random.Random(17))

    absorbing = _init("free", noise=DiffusionNoise.ABSORBING).build()
    assert absorbing.ids == (9, 9, 9)
    assert absorbing.pinned_mask == (False, False, False)
    assert absorbing.update_mask == (True, True, True)


@pytest.mark.parametrize(
    ("slot", "message"),
    [
        (SlotInit("none", token_ids=(1,)), "mode=none"),
        (_init("pad", ids=(1,)), "does not accept"),
        (_init("pinned", ids=(1, 2)), "exactly 1 token_id or exactly num_slots"),
        (_init("pinned", ids=(1, 1, 2)), "distinct token_ids"),
        (_init("learned", ids=(1, 1, 2)), "distinct"),
        (_init("prompt", ids=(1, 2)), "exactly 3"),
        (_init("free", ids=(1,)), "does not accept"),
        (_init("mask", ids=(1,), noise=DiffusionNoise.ABSORBING), "does not accept"),
        (_init("pinned", ids=(32,)), "within the model vocabulary"),
        (_init("pinned", count=0, ids=(1,)), "positive"),
    ],
)
def test_slot_modes_reject_invalid_counts_ids_and_vocab(slot, message):
    with pytest.raises(ValueError, match=message):
        slot.build()
