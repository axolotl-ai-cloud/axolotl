"""Tests for the shared seq_idx injection into a model forward."""

import torch

from axolotl.monkeypatch.models.mamba_utils import patch_model_forward_seq_idx


class _Model:
    def __init__(self):
        self.calls = []

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        self.calls.append(kwargs)
        return input_ids


class _Cache:
    def __init__(self, has_previous_state):
        self.has_previous_state = has_previous_state


def _fresh_model_cls():
    return type("Model", (_Model,), {})


def test_seq_idx_derived_from_position_ids_kwarg():
    cls = _fresh_model_cls()
    patch_model_forward_seq_idx(cls)
    model = cls()

    model.forward(torch.zeros(1, 7), position_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2]]))

    assert model.calls[0]["seq_idx"].tolist() == [[0, 0, 0, 0, 1, 1, 1]]
    assert model.calls[0]["seq_idx"].dtype == torch.int32


def test_seq_idx_derived_from_positional_position_ids():
    cls = _fresh_model_cls()
    patch_model_forward_seq_idx(cls)
    model = cls()

    model.forward(torch.zeros(1, 4), None, torch.tensor([[0, 1, 0, 1]]))

    assert model.calls[0]["seq_idx"].tolist() == [[0, 0, 1, 1]]


def test_caller_provided_seq_idx_wins():
    cls = _fresh_model_cls()
    patch_model_forward_seq_idx(cls)
    model = cls()
    mine = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)

    model.forward(
        torch.zeros(1, 4), position_ids=torch.tensor([[0, 1, 0, 1]]), seq_idx=mine
    )

    assert model.calls[0]["seq_idx"] is mine


def test_no_seq_idx_while_decoding_or_without_position_ids():
    cls = _fresh_model_cls()
    patch_model_forward_seq_idx(cls)
    model = cls()

    model.forward(torch.zeros(1, 4))
    model.forward(
        torch.zeros(1, 1),
        position_ids=torch.tensor([[5]]),
        past_key_values=_Cache(has_previous_state=True),
    )

    assert all("seq_idx" not in call for call in model.calls)


def test_patch_is_idempotent():
    cls = _fresh_model_cls()
    patch_model_forward_seq_idx(cls)
    once = cls.forward
    patch_model_forward_seq_idx(cls)

    assert cls.forward is once
