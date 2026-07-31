"""Teacher-forward prefetch: identity guard, FIFO slot, loader wrapper, gating."""

from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.distill import (
    TernaryDistillTrainer,
    _forward_inputs,
    _last_hidden,
    _TeacherPrefetch,
    _TeacherPrefetchLoader,
)


def _tiny_teacher() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=48,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()


def _batch() -> dict:
    return {
        "input_ids": torch.randint(0, 48, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
        "labels": torch.randint(0, 48, (2, 6)),
    }


def test_take_returns_the_inline_teacher_hidden():
    teacher = _tiny_teacher()
    inputs = _batch()
    prefetch = _TeacherPrefetch()
    prefetch.submit(teacher, inputs)

    hidden = prefetch.take(inputs)
    assert hidden is not None
    with torch.no_grad():
        keys = {k: inputs[k] for k in ("input_ids", "attention_mask")}
        inline = _last_hidden(teacher(**_forward_inputs(keys, teacher)), teacher)
    assert torch.equal(hidden, inline)
    assert prefetch.take(inputs) is None  # slot consumed


def test_replaced_tensor_falls_back():
    teacher = _tiny_teacher()
    inputs = _batch()
    prefetch = _TeacherPrefetch()
    prefetch.submit(teacher, inputs)
    inputs["attention_mask"] = inputs["attention_mask"].clone()
    assert prefetch.take(inputs) is None


def test_dropped_or_added_key_falls_back():
    teacher = _tiny_teacher()
    prefetch = _TeacherPrefetch()

    inputs = _batch()
    prefetch.submit(teacher, inputs)
    del inputs["attention_mask"]
    assert prefetch.take(inputs) is None

    inputs = _batch()
    prefetch.submit(teacher, inputs)
    inputs["position_ids"] = torch.arange(6).expand(2, 6)
    assert prefetch.take(inputs) is None


def test_loader_wrapper_submits_and_delegates():
    calls = []
    trainer = SimpleNamespace(_submit_teacher_prefetch=calls.append)
    base = [_batch(), _batch()]
    wrapped = _TeacherPrefetchLoader(base, trainer)

    assert list(wrapped) == base
    assert len(calls) == 2
    assert len(wrapped) == 2
    assert wrapped.index == base.index  # attribute delegation


def test_submit_gating_skips_before_teacher_ready():
    prefetch = _TeacherPrefetch()
    fake = SimpleNamespace(
        distill_multiplier=1.0,
        attn_relation_layer=None,
        _teacher_ready=False,
        _teacher=_tiny_teacher(),
        _teacher_prefetch=prefetch,
        prefetch_teacher=True,
    )
    TernaryDistillTrainer._submit_teacher_prefetch(fake, _batch())
    assert prefetch._slot is None

    fake._teacher_ready = True
    TernaryDistillTrainer._submit_teacher_prefetch(fake, _batch())
    assert prefetch._slot is not None
