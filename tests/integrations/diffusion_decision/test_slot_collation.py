"""Fixed decision slots retain their canvas and collation invariants."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from axolotl.integrations.diffusion_decision.loss import decision_example_from_canvas
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.slots import SlotInit, SlotMode
from axolotl.integrations.diffusion_decision.training_collator import (
    DecisionTrainingCollator,
)
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


def _spec(layout: DiffusionLayout, noise: DiffusionNoise) -> DiffusionSpec:
    return DiffusionSpec(
        noise=noise,
        layout=layout,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=layout is DiffusionLayout.ENCODER_CANVAS,
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


def _record(identifier: str):
    return {
        "id": identifier,
        "source": "test",
        "group": "test",
        "state": f"state-{identifier}",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick.",
                "options": ["one", "two"],
            }
        },
        "labels": {"q": {"kind": "hard", "gold_idx": 0}},
    }


def _plan(mode: SlotMode, noise: DiffusionNoise):
    return SlotInit(
        mode,
        token_ids={"pinned": (7,), "learned": (7, 8), "prompt": (7, 8)}.get(mode, ()),
        num_slots=2,
        vocab_size=256,
        pad_id=0,
        spec=_spec(DiffusionLayout.FULL_SEQUENCE, noise),
        mask_token_id=9,
    ).build()


def _canvas(identifier: str, mode: str, noise: DiffusionNoise):
    return build_decision_canvas(
        CharacterTokenizer(),
        _record(identifier),
        prompt_ids=(90 + int(identifier),),
        scaffold_ids=(),
        turn_close_id=106,
        pad_id=0,
        vocab_size=256,
        width=128,
        seed=23,
        steps=1,
        noise_kind=noise.value,
        mask_token_id=9 if noise is DiffusionNoise.ABSORBING else None,
        slot_plan=_plan(cast(SlotMode, mode), noise),
        thought_open_ids=(70, 71),
        thought_close_ids=(72,),
    )


def _rows(mode: str, noise: DiffusionNoise):
    canvases = tuple(_canvas(str(index), mode, noise) for index in range(2))
    return canvases, [
        {
            "canvas": canvas,
            "source": f"source-{index}",
            "decision_example": decision_example_from_canvas(canvas),
        }
        for index, canvas in enumerate(canvases)
    ]


@pytest.mark.parametrize(
    ("mode", "noise", "expected_ids"),
    [
        ("pinned", DiffusionNoise.UNIFORM, (7, 7)),
        ("learned", DiffusionNoise.UNIFORM, (7, 8)),
        ("mask", DiffusionNoise.ABSORBING, (9, 9)),
        ("pad", DiffusionNoise.UNIFORM, (0, 0)),
        ("prompt", DiffusionNoise.UNIFORM, (7, 8)),
    ],
)
@pytest.mark.parametrize(
    "layout", [DiffusionLayout.FULL_SEQUENCE, DiffusionLayout.ENCODER_CANVAS]
)
def test_fixed_slot_collation_keeps_slots_pinned_and_out_of_canvas_loss(
    mode: str,
    noise: DiffusionNoise,
    expected_ids: tuple[int, ...],
    layout: DiffusionLayout,
):
    canvases, rows = _rows(mode, noise)
    batch = DecisionTrainingCollator(_spec(layout, noise))(rows)

    for canvas in canvases:
        slot_positions = tuple(
            index for index, value in enumerate(canvas.slot_mask) if value
        )
        assert set(slot_positions).isdisjoint(canvas.label_positions)
        if mode == "prompt":
            assert canvas.prompt_ids[: len(expected_ids)] == expected_ids
            assert canvas.prompt_slot_mask == (True, True, False)
            assert not any(canvas.slot_mask)
        else:
            assert (
                tuple(canvas.canvas_ids[index] for index in slot_positions)
                == expected_ids
            )
            assert all(canvas.pinned_mask[index] for index in slot_positions)

    if layout is DiffusionLayout.FULL_SEQUENCE:
        _assert_full_sequence_contract(batch, canvases, mode, expected_ids)
    else:
        _assert_encoder_canvas_contract(
            batch["diffusion_batch"], canvases, mode, expected_ids
        )


def _assert_full_sequence_contract(batch, canvases, mode, expected_ids):
    input_ids = batch["input_ids"][0]
    pinned = batch["canvas_input_pinned_mask"][0]
    loss = batch["canvas_loss_mask"][0]
    corruptible = batch["canvas_corruptible_mask"][0]
    update = batch["canvas_update_mask"][0]
    offset = 0
    for document, canvas in enumerate(canvases):
        prompt_length = len(canvas.prompt_ids)
        canvas_start = offset + prompt_length
        slot_positions = tuple(
            index for index, value in enumerate(canvas.slot_mask) if value
        )
        fixed_positions = (
            tuple(range(offset, offset + len(expected_ids)))
            if mode == "prompt"
            else tuple(canvas_start + index for index in slot_positions)
        )
        assert (
            tuple(input_ids[position].item() for position in fixed_positions)
            == expected_ids
        )
        assert torch.all(pinned[list(fixed_positions)])
        assert not torch.any(loss[list(fixed_positions)])
        assert not torch.any(corruptible[list(fixed_positions)])
        assert not torch.any(update[list(fixed_positions)])
        assert batch["document_ids"][
            0, offset : offset + prompt_length + len(canvas.canvas_ids)
        ].tolist() == [document] * (prompt_length + len(canvas.canvas_ids))
        assert batch["position_ids"][
            0, offset : offset + prompt_length + len(canvas.canvas_ids)
        ].tolist() == list(range(prompt_length + len(canvas.canvas_ids)))
        if mode == "prompt":
            assert max(fixed_positions) < canvas_start
            assert canvas.prompt_slot_mask == (True, True, False)
        offset += prompt_length + len(canvas.canvas_ids)


def _assert_encoder_canvas_contract(diffusion_batch, canvases, mode, expected_ids):
    for row, canvas in enumerate(canvases):
        slot_positions = tuple(
            index for index, value in enumerate(canvas.slot_mask) if value
        )
        if mode == "prompt":
            fixed_positions = tuple(range(len(expected_ids)))
            assert canvas.prompt_slot_mask == (True, True, False)
            assert (
                tuple(
                    diffusion_batch.encoder_input_ids[
                        row, list(fixed_positions)
                    ].tolist()
                )
                == expected_ids
            )
            assert torch.all(
                diffusion_batch.encoder_validity[row, list(fixed_positions)]
            )
            assert not torch.any(
                diffusion_batch.encoder_ar_valid_mask[row, list(fixed_positions)]
            )
            assert diffusion_batch.encoder_ar_valid_mask[row, len(expected_ids)]
        else:
            fixed_positions = slot_positions
            assert (
                tuple(
                    diffusion_batch.canvas_clean_ids[
                        row, list(fixed_positions)
                    ].tolist()
                )
                == expected_ids
            )
        assert torch.all(
            diffusion_batch.canvas_input_pinned_mask[row, list(slot_positions)]
        )
        assert not torch.any(
            diffusion_batch.canvas_loss_mask[row, list(slot_positions)]
        )
        assert not torch.any(
            diffusion_batch.canvas_corruptible_mask[row, list(slot_positions)]
        )
        assert not torch.any(
            diffusion_batch.canvas_update_mask[row, list(slot_positions)]
        )
        assert not torch.any(
            diffusion_batch.canvas_sc_eligible_mask[row, list(slot_positions)]
        )
        assert torch.all(
            diffusion_batch.canvas_read_only_mask[row, list(slot_positions)]
        )
