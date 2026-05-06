"""Tests for the ProTrain calibration profiler's batch_factory.

Covers:

* Task-type detection across the four supported heads (causal LM,
  sequence classification, token classification, encoder-decoder)
  using HuggingFace tiny configs.
* Per-task batch shapes and dtypes.
* End-to-end forward + backward on a non-causal-LM head — the
  acceptance test that proves the profiler can build a valid batch
  for sequence classification without falling back to causal-LM
  shapes.
* Causal-LM regression — the legacy ``_dummy_batch`` shape
  (``input_ids`` + ``labels``, no ``attention_mask``) is preserved
  bit-for-bit so cached profiler traces from prior runs remain valid.

All tests are CPU-only and use HF configs to construct tiny randomly-
initialised models — no network calls, no GPU needed, fast lane.
"""

from __future__ import annotations

import torch

from axolotl.integrations.protrain.profiler.batch_factory import (
    KNOWN_TASKS,
    TASK_CAUSAL_LM,
    TASK_SEQ2SEQ_LM,
    TASK_SEQ_CLASSIFICATION,
    TASK_TOKEN_CLASSIFICATION,
    build_batch,
    detect_task_type,
    get_factory,
    register_factory,
    reset_factories,
)

# ---- detection ----------------------------------------------------------


def _make_seqcls_model(num_labels: int = 3):
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(cfg)


def _make_tokcls_model(num_labels: int = 4):
    from transformers import BertConfig, BertForTokenClassification

    cfg = BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=num_labels,
    )
    return BertForTokenClassification(cfg)


def _make_seq2seq_model():
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(
        vocab_size=64,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=8,
        decoder_start_token_id=0,
        pad_token_id=0,
    )
    return T5ForConditionalGeneration(cfg)


def _make_causal_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
    )
    return GPT2LMHeadModel(cfg)


def test_detect_task_type_causal_lm():
    """GPT-2 (``LMHeadModel``-suffixed) is detected as causal LM."""
    model = _make_causal_model()
    assert detect_task_type(model) == TASK_CAUSAL_LM


def test_detect_task_type_sequence_classification():
    model = _make_seqcls_model()
    assert detect_task_type(model) == TASK_SEQ_CLASSIFICATION


def test_detect_task_type_token_classification():
    model = _make_tokcls_model()
    assert detect_task_type(model) == TASK_TOKEN_CLASSIFICATION


def test_detect_task_type_encoder_decoder():
    model = _make_seq2seq_model()
    assert detect_task_type(model) == TASK_SEQ2SEQ_LM


def test_detect_task_type_via_architectures_attribute():
    """When ``config.architectures`` is populated, it wins over module class.

    Simulates a model loaded from a saved checkpoint where HF stamps
    the concrete class name into ``config.architectures``.
    """

    class _Cfg:
        architectures = ["LlamaForSequenceClassification"]
        is_encoder_decoder = False

    class _Model:
        config = _Cfg()

    assert detect_task_type(_Model()) == TASK_SEQ_CLASSIFICATION


def test_detect_task_type_via_is_encoder_decoder_flag():
    """Falls back to ``config.is_encoder_decoder`` when architectures is empty."""

    class _Cfg:
        architectures = None
        is_encoder_decoder = True

    class _Model:
        config = _Cfg()

    assert detect_task_type(_Model()) == TASK_SEQ2SEQ_LM


def test_detect_task_type_unknown_defaults_to_causal_lm():
    """Unknown configs degrade to causal LM (preserves legacy behaviour)."""

    class _Cfg:
        architectures = None
        is_encoder_decoder = False

    class _Model:
        config = _Cfg()

    assert detect_task_type(_Model()) == TASK_CAUSAL_LM


# ---- batch shape contracts ----------------------------------------------


def test_causal_lm_batch_shape_preserves_legacy_keys():
    """Causal-LM batches MUST have exactly ``{input_ids, labels}`` to
    keep cached profiler traces from prior runs valid (the cache key is
    keyed on op_order, which depends on the kwargs passed to the
    forward — adding/removing keys changes the trace)."""
    model = _make_causal_model()
    batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
    assert set(batch.keys()) == {"input_ids", "labels"}
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_seq_classification_batch_shape():
    model = _make_seqcls_model(num_labels=3)
    batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
    # Per-sequence labels — shape (B,), not (B, S).
    assert batch["labels"].shape == (2,)
    assert batch["labels"].dtype == torch.long
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    # Labels must respect num_labels.
    assert int(batch["labels"].max()) < 3


def test_token_classification_batch_shape():
    model = _make_tokcls_model(num_labels=4)
    batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
    # Per-token labels — shape (B, S).
    assert batch["labels"].shape == (2, 8)
    assert batch["labels"].dtype == torch.long
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert int(batch["labels"].max()) < 4


def test_seq2seq_lm_batch_shape():
    model = _make_seq2seq_model()
    batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
    # Encoder-decoder: labels are decoder targets (B, S).
    assert batch["labels"].shape == (2, 8)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)


# ---- end-to-end forward + backward on a non-causal-LM head --------------


def test_seq_classification_batch_drives_forward_and_backward_cpu():
    """ACCEPTANCE: the profiler can build a valid batch for a non-causal-LM
    head and drive ``model(**batch)`` + ``loss.backward()`` end-to-end on
    CPU.

    This exercises the path that the calibration profiler takes when the
    cache misses — without the batch_factory fix, the wrapper would
    construct an ``input_ids`` + ``labels`` pair shaped for causal LM,
    which Bert's sequence-classification head reads as per-sequence
    labels of the wrong shape and either crashes or computes a nonsense
    loss against ``num_labels`` classes.
    """
    model = _make_seqcls_model(num_labels=3)
    batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
    out = model(**batch)
    # Loss must be a finite scalar tensor.
    assert out.loss is not None
    assert out.loss.dim() == 0
    assert torch.isfinite(out.loss).item()
    # Logits shape must match (B, num_labels) — proves the head saw
    # per-sequence labels rather than per-token (which would give
    # (B, S, num_labels)).
    assert out.logits.shape == (2, 3)
    # Backward must succeed — proves labels are dtype-compatible with
    # the head's CrossEntropyLoss.
    out.loss.backward()
    # At least one parameter received a non-zero gradient.
    grad_seen = any(
        (p.grad is not None and p.grad.abs().sum() > 0) for p in model.parameters()
    )
    assert grad_seen, "no parameter received a gradient on the seq-cls head"


def test_token_classification_batch_drives_forward_and_backward_cpu():
    """Token-classification head accepts per-token labels of shape (B, S)."""
    model = _make_tokcls_model(num_labels=4)
    batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
    out = model(**batch)
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    assert out.logits.shape == (2, 8, 4)
    out.loss.backward()


def test_seq2seq_lm_batch_drives_forward_and_backward_cpu():
    """T5 conditional-generation accepts ``labels`` and shifts internally."""
    model = _make_seq2seq_model()
    batch = build_batch(model, batch_size=2, seq_len=8, device="cpu")
    out = model(**batch)
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    out.loss.backward()


# ---- model_wrapper._dummy_batch delegates to the factory ----------------


def test_dummy_batch_delegates_to_factory_for_seq_classification():
    """``model_wrapper._dummy_batch`` MUST reach the new factory dispatch.

    Regression guard: if a future refactor inlines causal-LM logic back
    into ``_dummy_batch``, this test catches it.
    """
    from axolotl.integrations.protrain.api.model_wrapper import _dummy_batch

    model = _make_seqcls_model(num_labels=5)
    batch = _dummy_batch(model, 2, 8, "cpu")
    # Per-sequence labels prove the dispatch — the legacy code-path
    # would have produced (B, S) labels.
    assert batch["labels"].shape == (2,)
    assert int(batch["labels"].max()) < 5


def test_dummy_batch_preserves_causal_lm_shape():
    """Causal-LM regression guard: ``{input_ids, labels}`` exactly."""
    from axolotl.integrations.protrain.api.model_wrapper import _dummy_batch

    model = _make_causal_model()
    batch = _dummy_batch(model, 2, 8, "cpu")
    assert set(batch.keys()) == {"input_ids", "labels"}
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)


# ---- registry plumbing --------------------------------------------------


def test_register_custom_factory_overrides_default():
    """Users (or another integration) can register a custom factory."""
    # ``build_batch`` validates the registered factory's output (must be a
    # dict containing 'labels') so the profiler can synthesize backward;
    # the sentinel includes ``labels`` so the override-takes-precedence
    # check stays focused on registry plumbing rather than the validator.
    sentinel = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "labels": torch.zeros(1, 1, dtype=torch.long),
    }

    def _custom(model, bs, sl, dev):
        return sentinel

    try:
        register_factory(TASK_CAUSAL_LM, _custom)
        model = _make_causal_model()
        batch = build_batch(model, 2, 8, "cpu")
        assert batch is sentinel
    finally:
        reset_factories()


def test_get_factory_unknown_falls_back_to_causal_lm():
    """Unknown task-type strings fall back rather than raising.

    Defensive: the profiler should never crash because of an unknown
    task taxonomy entry — degrading to causal LM is preferable.
    """
    from axolotl.integrations.protrain.profiler.batch_factory import (
        causal_lm_batch_factory,
    )

    factory = get_factory("totally-not-a-real-task")
    assert factory is causal_lm_batch_factory


def test_known_tasks_covers_all_acceptance_criteria_heads():
    """The acceptance criteria list 4 head types — they must all be in
    the public taxonomy."""
    expected = {
        TASK_CAUSAL_LM,
        TASK_SEQ_CLASSIFICATION,
        TASK_TOKEN_CLASSIFICATION,
        TASK_SEQ2SEQ_LM,
    }
    assert expected.issubset(set(KNOWN_TASKS))


# ---- explicit task_type override ----------------------------------------


def test_build_batch_explicit_task_type_override():
    """Caller can force a task type, bypassing detection."""
    # GPT-2 model but force seq-classification batch shape.
    model = _make_causal_model()
    batch = build_batch(model, 2, 8, "cpu", task_type=TASK_SEQ_CLASSIFICATION)
    # Per-sequence labels — shape (B,) — matches forced override, not
    # GPT-2's natural causal-LM shape.
    assert batch["labels"].shape == (2,)
