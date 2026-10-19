"""Tests for AxolotlKDTrainer.compute_loss."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Skip the entire module if the KD trainer can't be imported (env not set up).
axolotl_kd_trainer = pytest.importorskip("axolotl.integrations.kd.trainer")
AxolotlKDTrainer = axolotl_kd_trainer.AxolotlKDTrainer
_resolve_lm_head = axolotl_kd_trainer._resolve_lm_head


HIDDEN = 8
VOCAB = 16
SEQ = 4
BSZ = 2
TOP_K = 3


def test_resolve_lm_head_standard():
    model = SimpleNamespace(lm_head=nn.Linear(HIDDEN, VOCAB, bias=False))
    assert _resolve_lm_head(model) is model.lm_head


def test_resolve_lm_head_multimodal():
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    language_model = SimpleNamespace(lm_head=lm_head)
    model = SimpleNamespace(language_model=language_model)
    assert _resolve_lm_head(model) is lm_head


def test_resolve_lm_head_peft_wrapped():
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    inner = SimpleNamespace(lm_head=lm_head)
    model = SimpleNamespace(get_base_model=lambda: inner)
    assert _resolve_lm_head(model) is lm_head


def test_resolve_lm_head_peft_wrapped_multimodal():
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    language_model = SimpleNamespace(lm_head=lm_head)
    inner = SimpleNamespace(language_model=language_model)
    model = SimpleNamespace(get_base_model=lambda: inner)
    assert _resolve_lm_head(model) is lm_head


def test_resolve_lm_head_missing_raises():
    model = SimpleNamespace()
    with pytest.raises(AttributeError, match="could not find lm_head"):
        _resolve_lm_head(model)


def _build_fake_model():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

        def forward(self, input_ids=None, output_hidden_states=False, **kw):
            assert output_hidden_states is True
            assert "labels" not in kw
            assert "target_token_ids" not in kw
            assert "target_logprobs" not in kw
            assert "target_mask" not in kw
            assert "num_items_in_batch" not in kw
            hidden = torch.randn(BSZ, SEQ, HIDDEN, requires_grad=True)
            return SimpleNamespace(
                loss=None,
                logits=None,
                hidden_states=(hidden,),
                past_key_values=None,
                attentions=None,
            )

    return TinyModel()


def _build_inputs(labels=None):
    if labels is None:
        labels = torch.randint(0, VOCAB, (BSZ, SEQ))
    return {
        "input_ids": torch.randint(0, VOCAB, (BSZ, SEQ)),
        "labels": labels,
        "target_token_ids": torch.randint(0, VOCAB, (BSZ, SEQ, TOP_K)),
        "target_logprobs": torch.log_softmax(torch.randn(BSZ, SEQ, TOP_K), dim=-1),
        "target_mask": torch.ones(BSZ, SEQ, TOP_K, dtype=torch.bool),
    }


def test_compute_loss_calls_kd_loss_with_correct_shapes():
    kd_loss_fn = MagicMock(return_value=torch.tensor(2.0, requires_grad=True))
    fake_self = SimpleNamespace(
        args=SimpleNamespace(sample_packing=False),
        model_accepts_loss_kwargs=True,
        _kd_loss_fn=kd_loss_fn,
    )
    model = _build_fake_model()
    inputs = _build_inputs()
    expected_labels = inputs["labels"].clone()
    expected_target_ids = inputs["target_token_ids"].clone()

    loss = AxolotlKDTrainer.compute_loss(fake_self, model, inputs)

    kd_loss_fn.assert_called_once()
    args, kwargs = kd_loss_fn.call_args
    assert args[0].shape == (VOCAB, HIDDEN)
    assert args[1].shape == (BSZ, SEQ, HIDDEN)
    assert args[2].shape == expected_target_ids.shape
    assert "true_labels" in kwargs
    assert kwargs["true_labels"].shape == expected_labels.shape
    assert torch.isfinite(loss).all()


def test_compute_loss_divides_by_num_items_in_batch_from_labels():
    kd_loss_fn = MagicMock(return_value=torch.tensor(8.0))
    fake_self = SimpleNamespace(
        args=SimpleNamespace(sample_packing=False),
        model_accepts_loss_kwargs=True,
        _kd_loss_fn=kd_loss_fn,
    )
    model = _build_fake_model()
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, -100]])
    inputs = _build_inputs(labels=labels)

    loss = AxolotlKDTrainer.compute_loss(fake_self, model, inputs)
    assert torch.isclose(loss, torch.tensor(2.0))


def test_compute_loss_uses_explicit_num_items_in_batch():
    kd_loss_fn = MagicMock(return_value=torch.tensor(8.0))
    fake_self = SimpleNamespace(
        args=SimpleNamespace(sample_packing=False),
        model_accepts_loss_kwargs=True,
        _kd_loss_fn=kd_loss_fn,
    )
    model = _build_fake_model()
    inputs = _build_inputs()

    loss = AxolotlKDTrainer.compute_loss(fake_self, model, inputs, num_items_in_batch=2)
    assert torch.isclose(loss, torch.tensor(4.0))


def test_compute_loss_does_not_divide_when_zero_items():
    kd_loss_fn = MagicMock(return_value=torch.tensor(8.0))
    fake_self = SimpleNamespace(
        args=SimpleNamespace(sample_packing=False),
        model_accepts_loss_kwargs=True,
        _kd_loss_fn=kd_loss_fn,
    )
    model = _build_fake_model()
    labels = torch.full((BSZ, SEQ), -100)
    inputs = _build_inputs(labels=labels)

    loss = AxolotlKDTrainer.compute_loss(fake_self, model, inputs)
    assert torch.isclose(loss, torch.tensor(8.0))


def test_compute_loss_raises_when_kd_keys_missing():
    kd_loss_fn = MagicMock(return_value=torch.tensor(1.0))
    fake_self = SimpleNamespace(
        args=SimpleNamespace(sample_packing=False),
        model_accepts_loss_kwargs=True,
        _kd_loss_fn=kd_loss_fn,
    )
    model = _build_fake_model()
    inputs = {
        "input_ids": torch.randint(0, VOCAB, (BSZ, SEQ)),
        "labels": torch.randint(0, VOCAB, (BSZ, SEQ)),
    }
    with pytest.raises(KeyError, match="KD batch missing required keys"):
        AxolotlKDTrainer.compute_loss(fake_self, model, inputs)


def test_compute_loss_raises_when_hidden_states_missing():
    kd_loss_fn = MagicMock(return_value=torch.tensor(1.0))
    fake_self = SimpleNamespace(
        args=SimpleNamespace(sample_packing=False),
        model_accepts_loss_kwargs=True,
        _kd_loss_fn=kd_loss_fn,
    )

    class NoHiddenStatesModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

        def forward(self, **kw):
            return SimpleNamespace(
                loss=None,
                logits=None,
                hidden_states=None,
                past_key_values=None,
                attentions=None,
            )

    inputs = _build_inputs()
    with pytest.raises(RuntimeError, match="did not return hidden_states"):
        AxolotlKDTrainer.compute_loss(fake_self, NoHiddenStatesModel(), inputs)


def test_compute_loss_does_not_mutate_caller_inputs():
    kd_loss_fn = MagicMock(return_value=torch.tensor(1.0))
    fake_self = SimpleNamespace(
        args=SimpleNamespace(sample_packing=False),
        model_accepts_loss_kwargs=True,
        _kd_loss_fn=kd_loss_fn,
    )
    model = _build_fake_model()
    inputs = _build_inputs()
    original_keys = set(inputs.keys())

    AxolotlKDTrainer.compute_loss(fake_self, model, inputs)
    assert set(inputs.keys()) == original_keys
