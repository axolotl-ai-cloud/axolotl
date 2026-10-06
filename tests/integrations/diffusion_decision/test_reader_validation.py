"""Reject malformed canvas metadata before invoking a model or serving API."""

from dataclasses import replace

import pytest
import torch

from axolotl.integrations.diffusion_decision.readers.base import (
    restricted_probabilities,
    validate_canvas,
)

from tests.integrations.diffusion_decision.helpers import (
    make_canvas,
)


def canvas():
    return make_canvas(
        (1, 2),
        (3, 4, 5, 0),
        (1,),
        allowed=(6, 7),
        question_ids=("answer",),
        template_length=2,
    )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"prompt_ids": ()}, "requires a prompt"),
        ({"canvas_ids": (3, -1, 5, 0)}, "nonnegative"),
        ({"allowed_ids": ((True, 7),)}, "nonnegative"),
        ({"template_length": 4}, "template_length"),
        ({"label_positions": (2,)}, "outside"),
        ({"slot_mask": (False, True, False, False)}, "latent slots"),
        ({"question_ids": ("",)}, "distinct nonempty"),
    ],
)
def test_invalid_canvas_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_canvas(replace(canvas(), **changes))


def test_duplicate_question_ids_rejected():
    with pytest.raises(ValueError, match="distinct nonempty"):
        validate_canvas(replace(canvas(), question_ids=("answer", "answer")))


def test_full_width_pad_tokens_remain_semantically_visible():
    assert validate_canvas(canvas()) == 4


def test_candidate_mask_does_not_extract_tensor_scalars(monkeypatch):
    logits = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), -1)

    def fail(*_args, **_kwargs):
        raise AssertionError("candidate mask extracted a tensor scalar")

    with monkeypatch.context() as patch:
        for name in ("item", "__bool__", "__int__"):
            patch.setattr(torch.Tensor, name, fail)
        _, mask, probabilities = restricted_probabilities(logits, ((0, 2), (1,)))
    assert mask.tolist() == [[True, True], [True, False]]
    torch.testing.assert_close(
        probabilities[0], torch.softmax(torch.tensor([1.0, 3.0]), 0)
    )
    torch.testing.assert_close(probabilities[1], torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize("token", [-1, 3])
def test_candidate_token_outside_vocabulary_rejected(token):
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        restricted_probabilities(torch.zeros(1, 3), ((token,),))


def test_validate_canvas_rejects_pinned_label_positions():
    pinned = list(canvas().pinned_mask)
    pinned[1] = True
    with pytest.raises(ValueError, match="pinned tokens cannot be label positions"):
        validate_canvas(replace(canvas(), pinned_mask=tuple(pinned)))
