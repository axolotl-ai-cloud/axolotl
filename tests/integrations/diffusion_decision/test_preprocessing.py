"""Preprocessing uses independent upstream fixtures with actual tokenizer IDs."""

import random

import pytest

from axolotl.integrations.diffusion_decision.adapters.jsonl import normalize_jsonl
from axolotl.integrations.diffusion_decision.preprocessing import build_decision_canvas
from axolotl.integrations.diffusion_decision.slots import SlotInit

from tests.integrations.diffusion_decision.helpers import (
    DJEV_FIXTURE,
    RecordedTokenizer,
    make_spec,
)


def inputs(family, index):
    reference = DJEV_FIXTURE["tokenizers"][family]
    case = reference["cases"][index]
    labels = {}
    for i, q in enumerate(case["parsed"]["questions"]):
        count = len(q["labels"])
        labels[q["id"]] = (
            {"kind": "hard", "gold_idx": 0}
            if i % 3 == 0
            else {"kind": "dist", "probs": [1 / count] * count}
            if i % 3 == 1
            else {"kind": "set", "allowed_set": [0, count - 1]}
        )
    record = {
        "source": "fixture",
        "group": "synthetic",
        "state": "observed",
        "questions": {
            q["id"]: {k: v for k, v in q.items() if k != "id"}
            for q in case["schema"]["questions"]
        },
        "labels": labels,
    }
    return reference, case, record


@pytest.mark.parametrize("family", ["gemma", "dream"])
@pytest.mark.parametrize("index", [0, 9, 10, 19])
def test_exact_template_positions_noise_and_target_preservation(family, index):
    reference, case, record = inputs(family, index)
    kwargs = dict(
        scaffold_ids=case["head"],
        turn_close_id=106 if family == "gemma" else 151643,
        pad_id=0 if family == "gemma" else 151643,
        vocab_size=262144 if family == "gemma" else 152064,
        seed=23,
    )
    first = build_decision_canvas(
        RecordedTokenizer(reference["encodings"]), record, [1, 2], **kwargs
    )
    second = build_decision_canvas(
        RecordedTokenizer(reference["encodings"]), record, [1, 2], steps=2, **kwargs
    )
    base, slots = case["template"]
    expected = (
        base + [kwargs["turn_close_id"]] + [kwargs["pad_id"]] * (128 - len(base) - 1)
    )
    rng = random.Random(23)
    for slot in slots:
        expected[slot["pos"]] = rng.randrange(kwargs["vocab_size"])
    assert list(first.canvas_ids) == expected
    assert first.canvas_ids == second.canvas_ids
    assert first.template_length == len(base)
    assert first.targets == tuple(record["labels"].values())
    assert not any(first.pinned_mask)
    assert all(
        second.pinned_mask[i] == (i not in first.label_positions) for i in range(128)
    )
    assert all(first.semantic_mask) and not any(first.slot_mask)


def test_prevalidated_record_matches_validated_canvas():
    reference, case, record = inputs("gemma", 0)
    kwargs = dict(
        scaffold_ids=case["head"],
        turn_close_id=106,
        pad_id=0,
        vocab_size=262144,
        seed=23,
    )
    tokenizer = RecordedTokenizer(reference["encodings"])
    expected = build_decision_canvas(tokenizer, record, [1, 2], **kwargs)
    actual = build_decision_canvas(
        tokenizer,
        normalize_jsonl(record),
        [1, 2],
        prevalidated_record=True,
        **kwargs,
    )
    assert actual == expected


def test_absorbing_noise_and_invalid_soft_target():
    reference, case, record = inputs("dream", 2)
    kwargs = dict(
        scaffold_ids=case["head"],
        turn_close_id=151643,
        pad_id=151643,
        vocab_size=152064,
        noise_kind="absorbing",
        mask_token_id=151666,
    )
    result = build_decision_canvas(
        RecordedTokenizer(reference["encodings"]), record, [1], **kwargs
    )
    assert all(result.canvas_ids[pos] == 151666 for pos in result.label_positions)
    record["labels"]["1"]["probs"] = [0.1]
    with pytest.raises(ValueError, match="one probability"):
        build_decision_canvas(
            RecordedTokenizer(reference["encodings"]), record, [1], **kwargs
        )


@pytest.mark.parametrize("steps", [2, 3])
def test_generated_free_slots_remain_updateable_while_kstep_scaffold_is_pinned(steps):
    reference, case, record = inputs("gemma", 0)
    plan = SlotInit(
        "free",
        num_slots=2,
        vocab_size=262144,
        spec=make_spec(max_canvas=None, max_context=None),
    ).build(seed=7)
    canvas = build_decision_canvas(
        RecordedTokenizer(reference["encodings"]),
        record,
        [1, 2],
        scaffold_ids=case["head"],
        turn_close_id=106,
        pad_id=0,
        vocab_size=262144,
        steps=steps,
        slot_plan=plan,
    )
    slots = [index for index, value in enumerate(canvas.slot_mask) if value]
    labels = set(canvas.label_positions)
    assert len(slots) == 2
    assert all(not canvas.pinned_mask[index] for index in slots)
    assert all(not canvas.pinned_mask[index] for index in labels)
    assert all(
        canvas.pinned_mask[index]
        for index in range(len(canvas.canvas_ids))
        if index not in labels and index not in slots
    )


@pytest.mark.parametrize("family", ["gemma", "dream"])
def test_canvas_builds_source_prompt_from_validated_record(family):
    reference, case, record = inputs(family, 0)
    record["state"] = {"observed": "20°"}
    record["instructions"] = case["schema"]["instructions"]

    class ChatTokenizer(RecordedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "system", "content": case["system_text"]},
                {"role": "user", "content": '{"observed": "20°"}'},
            ]
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return {"input_ids": [1, 2, 3]}

    result = build_decision_canvas(
        ChatTokenizer(reference["encodings"]),
        record,
        scaffold_ids=case["head"],
        turn_close_id=106 if family == "gemma" else 151643,
        pad_id=0 if family == "gemma" else 151643,
        vocab_size=262144 if family == "gemma" else 152064,
    )
    assert result.prompt_ids == (1, 2, 3)
    assert result.targets == tuple(record["labels"].values())
