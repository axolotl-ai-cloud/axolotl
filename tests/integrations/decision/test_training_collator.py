from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from axolotl.integrations.decision.datasets import DecisionDataset
from axolotl.integrations.decision.loss import decision_example_from_canvas
from axolotl.integrations.decision.records import DecisionCanvas
from axolotl.integrations.decision.training_collator import (
    DecisionTrainingCollator,
)
from axolotl.model_support.diffusion import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    TimeWeighting,
)


def _spec(layout):
    return DiffusionSpec(
        noise=DiffusionNoise.UNIFORM,
        layout=layout,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.REQUIRES_PREDECESSOR,
        self_conditioning=layout is DiffusionLayout.ENCODER_CANVAS,
        max_canvas=8 if layout is DiffusionLayout.ENCODER_CANVAS else None,
        max_context=None,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=MaskTokenPolicy.NONE,
        default_time_weighting=TimeWeighting.NONE,
        objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
        generation_adapter=GenerationAdapter.FULL_SEQUENCE,
    )


def _canvas(prompt, canvas, positions, name):
    return DecisionCanvas(
        prompt_ids=prompt,
        canvas_ids=canvas,
        label_positions=positions,
        allowed_ids=tuple((1, 2) for _ in positions),
        question_ids=tuple(f"{name}{i}" for i in range(len(positions))),
        targets=tuple({"kind": "hard", "gold_idx": 0} for _ in positions),
        pinned_mask=(False,) * len(canvas),
        semantic_mask=(True,) * len(canvas),
        slot_mask=(False,) * len(canvas),
        template_length=len(canvas),
    )


def _rows():
    canvases = (
        _canvas((4, 5), (6, 7, 8, 9), (1, 3), "a"),
        _canvas((10, 11, 12), (13, 14, 15, 16), (2,), "b"),
    )
    return [
        {
            "canvas": canvas,
            "source": f"source-{index}",
            "decision_example": decision_example_from_canvas(
                canvas, source_weight=index + 1
            ),
        }
        for index, canvas in enumerate(canvases)
    ]


def test_full_sequence_concat_preserves_document_and_label_coordinates():
    batch = DecisionTrainingCollator(_spec(DiffusionLayout.FULL_SEQUENCE))(_rows())
    assert batch["input_ids"].tolist() == [
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ]
    assert batch["document_ids"].tolist() == [[0] * 6 + [1] * 7]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6]]
    assert batch["decision_label_rows"].tolist() == [[0, 0], [0, -1]]
    assert batch["decision_label_positions"].tolist() == [[3, 5], [11, -1]]
    assert batch["decision_question_mask"].tolist() == [[True, True], [True, False]]
    assert torch.equal(
        batch["canvas_update_mask"],
        batch["canvas_corruptible_mask"] & ~batch["canvas_input_pinned_mask"],
    )
    assert [q.position for e in batch["decision_examples"] for q in e.questions] == [
        0,
        1,
        0,
    ]
    assert batch["decision_sources"] == ("source-0", "source-1")
    assert "decision_slot_counts" not in batch
    assert "decision_draws" not in batch


def test_full_sequence_nested_packing_preserves_documents_labels_and_attention():
    rows = _rows()
    batch = DecisionTrainingCollator(_spec(DiffusionLayout.FULL_SEQUENCE))(
        [[rows[0]], [rows[1]]]
    )

    assert batch["document_ids"].tolist() == [[0] * 6 + [1] * 7]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6]]
    assert batch["decision_label_rows"].tolist() == [[0, 0], [0, -1]]
    assert batch["decision_label_positions"].tolist() == [[3, 5], [11, -1]]

    from axolotl.integrations.diffusion.lm.backends.full_sequence import (
        FullSequenceBackend,
    )

    packed = FullSequenceBackend(mask_token_id=99).pack(
        batch["input_ids"],
        batch["document_ids"],
        batch["semantic_validity"],
        batch["position_ids"],
    )
    attention = packed["attention_mask"]
    assert attention[0, 0, 0, 5]
    assert attention[0, 0, 6, 12]
    assert not attention[0, 0, 0, 6]
    assert not attention[0, 0, 6, 5]


def test_multipack_dataloader_covers_each_decision_once():
    from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

    rows = [
        {**row, "length": len(row["canvas"].prompt_ids) + len(row["canvas"].canvas_ids)}
        for row in _rows()
    ]
    dataset = DecisionDataset(rows, {"per_batch_stratified": False})
    sampler = MultipackBatchSampler(
        SequentialSampler(dataset),
        lengths=get_dataset_lengths(dataset),
        batch_max_len=20,
        batch_size=1,
        bin_size=20,
        group_size=2,
        num_processes=1,
        sequential=True,
        drop_last=True,
    )
    batches = list(
        DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=DecisionTrainingCollator(_spec(DiffusionLayout.FULL_SEQUENCE)),
        )
    )

    assert len(batches) == 1
    assert batches[0]["decision_sources"] == ("source-0", "source-1")
    assert batches[0]["document_ids"].tolist() == [[0] * 6 + [1] * 7]


def test_builder_compatible_constructor_accepts_standard_collator_options():
    collator = DecisionTrainingCollator(
        SimpleNamespace(pad_token_id=7),
        spec=_spec(DiffusionLayout.FULL_SEQUENCE),
        padding=True,
        max_length=None,
        pad_to_multiple_of=64,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    assert collator.pad_token_id == 7
    assert collator.pad_to_multiple_of == 64


def test_full_sequence_rejects_microbatch_above_payload_capacity():
    collator = DecisionTrainingCollator(
        _spec(DiffusionLayout.FULL_SEQUENCE), physical_payload_capacity=12
    )
    with pytest.raises(ValueError, match="physical payload capacity"):
        collator(_rows())
