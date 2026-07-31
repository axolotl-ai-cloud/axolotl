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


def test_loader_wrapper_submits_windows_and_delegates():
    calls = []
    trainer = SimpleNamespace(_submit_teacher_prefetch=calls.append)
    base = [_batch(), _batch(), _batch()]

    assert list(_TeacherPrefetchLoader(base, trainer)) == base
    assert [len(w) for w in calls] == [1, 1, 1]

    calls.clear()
    wrapped = _TeacherPrefetchLoader(base, trainer, depth=2)
    assert list(wrapped) == base
    assert [len(w) for w in calls] == [2, 1]
    assert len(wrapped) == 3
    assert wrapped.index == base.index  # attribute delegation


def test_fused_submit_matches_per_batch_hidden():
    teacher = _tiny_teacher()
    a, b = _batch(), _batch()

    fused = _TeacherPrefetch()
    fused.submit_many(teacher, [a, b])
    single = _TeacherPrefetch()
    single.submit(teacher, a)
    single.submit(teacher, b)

    for inputs in (a, b):
        got, want = fused.take(inputs), single.take(inputs)
        assert got is not None and want is not None
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_out_of_order_take_clears_the_queue():
    teacher = _tiny_teacher()
    a, b = _batch(), _batch()
    prefetch = _TeacherPrefetch()
    prefetch.submit_many(teacher, [a, b])

    assert prefetch.take(b) is None  # order broke
    assert prefetch.take(a) is None  # and the queue was dropped with it


def test_packed_rows_fuse_along_the_sequence():
    # values under stock attention differ from per-batch (no packing-aware
    # segmentation here); this pins the mechanics: slot count, order, shapes
    teacher = _tiny_teacher()
    batches = []
    for width in (8, 8, 12):
        batches.append(
            {
                "input_ids": torch.randint(0, 48, (1, width)),
                "attention_mask": torch.ones(1, width, dtype=torch.long),
                "position_ids": torch.arange(width).reshape(1, width),
            }
        )
    prefetch = _TeacherPrefetch()
    prefetch.submit_many(teacher, batches)
    assert len(prefetch._slots) == 3
    for batch in batches:
        hidden = prefetch.take(batch)
        assert hidden is not None
        assert hidden.shape[:2] == (1, batch["input_ids"].shape[1])


def test_ragged_window_falls_back_to_per_batch():
    teacher = _tiny_teacher()
    a = _batch()
    b = {
        "input_ids": torch.randint(0, 48, (2, 9)),
        "attention_mask": torch.ones(2, 9, dtype=torch.long),
    }
    prefetch = _TeacherPrefetch()
    prefetch.submit_many(teacher, [a, b])
    assert prefetch.take(a) is not None
    assert prefetch.take(b) is not None


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
    TernaryDistillTrainer._submit_teacher_prefetch(fake, [_batch()])
    assert not prefetch._slots

    fake._teacher_ready = True
    TernaryDistillTrainer._submit_teacher_prefetch(fake, [_batch()])
    assert prefetch._slots
