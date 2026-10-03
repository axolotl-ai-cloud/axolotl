"""Decision canvas visibility and atomic budget contracts."""

import pytest

from axolotl.integrations.decision.collator import DecisionCanvasCollator
from axolotl.integrations.decision.records import DecisionCanvas
from axolotl.integrations.decision.template import resolve_template


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def test_template_indexed_and_overflow():
    questions = [{"id": str(index), "labels": ["a", "b"]} for index in range(64)]
    _, slots = resolve_template(
        CharacterTokenizer(), questions, fmt="indexed", width=256
    )
    assert len(slots) == 64
    assert all(len(slot["label_ids"]) == 2 for slot in slots)
    with pytest.raises(ValueError):
        resolve_template(CharacterTokenizer(), questions, fmt="indexed", width=8)


def test_collator_preserves_prompt_and_canvas_metadata():
    canvas = DecisionCanvas(
        prompt_ids=[1, 2],
        canvas_ids=[3, 4, 0, 0],
        label_positions=[1],
        allowed_ids=[[4, 5]],
        question_ids=["q"],
        targets=[{"kind": "hard", "gold_idx": 0}],
        pinned_mask=[True, False, True, True],
        semantic_mask=[True, True, True, True],
        slot_mask=[False, False, False, False],
        template_length=2,
    )
    batch = DecisionCanvasCollator()([canvas])
    assert batch.canvas_loss_mask.tolist() == [[False, True, False, False]]
    assert batch.canvas_input_pinned_mask.tolist() == [[True, False, True, True]]
    assert batch.decoder_prefix_lengths.tolist() == [2]
    assert batch.encoder_lengths.tolist() == [2]
    assert batch.canvas_semantic_validity.tolist() == [[True] * 4]
