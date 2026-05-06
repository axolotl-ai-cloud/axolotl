"""Task-type-aware sample batch construction for the calibration profiler.

The profiler needs to drive a single forward (and optionally backward)
pass on the user's model so it can record per-op memory deltas, op
order, and steady-state timings. Until now the wrapper hard-coded a
``{"input_ids": ..., "labels": ...}`` batch which is correct for
HuggingFace causal LMs but wrong for other heads — a sequence
classifier wants integer ``labels`` of shape ``(batch_size,)``, a token
classifier wants per-token labels of shape ``(batch_size, seq_len)``,
and an encoder-decoder model needs a ``decoder_input_ids`` (and
``labels`` shaped to the decoder, not the encoder).

This module introduces a small registry of *batch factories* keyed by
the HuggingFace auto-class taxonomy that axolotl already uses
elsewhere (``AutoModelForCausalLM`` /
``AutoModelForSequenceClassification`` /
``AutoModelForTokenClassification`` /
``AutoModelForSeq2SeqLM``) so the profiler can ask the model for an
appropriate batch instead of hard-coding causal-LM shapes.

Detection priority — see :func:`detect_task_type`:

1. ``model.config.architectures`` — HF stamps the concrete class name
   here (``BertForSequenceClassification``, ``T5ForConditionalGeneration``,
   ...). We string-match suffixes against the taxonomy.
2. ``model.config.is_encoder_decoder`` — covers seq2seq models whose
   architectures attribute is missing or generic.
3. Fall back to causal-LM, which preserves the prior wrapper behaviour.

The taxonomy is intentionally aligned with axolotl's existing
``type_of_model`` / ``model_type`` strings (see
``utils/schemas/validation.py::set_reward_model_defaults``) so the same
set of strings flows from the user-facing schema through the loader to
the profiler without a translation layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

LOG = get_logger(__name__)


# ---- task-type taxonomy --------------------------------------------------
# Strings rather than an Enum so callers (the plugin, future factories
# registered from a different package) can pass the HF auto-class name
# directly without an extra import.

TASK_CAUSAL_LM = "causal_lm"
TASK_SEQ_CLASSIFICATION = "seq_classification"
TASK_TOKEN_CLASSIFICATION = "token_classification"  # nosec B105  # noqa: S105 - task type label, not a password
TASK_SEQ2SEQ_LM = "seq2seq_lm"

KNOWN_TASKS: tuple[str, ...] = (
    TASK_CAUSAL_LM,
    TASK_SEQ_CLASSIFICATION,
    TASK_TOKEN_CLASSIFICATION,
    TASK_SEQ2SEQ_LM,
)

# Mapping from a class-name SUFFIX to the canonical task string. The
# match is suffix-based because HF spells the class names as
# ``<ModelName>ForCausalLM`` etc. — both the auto-class
# (``AutoModelForCausalLM``) and the concrete class (``LlamaForCausalLM``)
# end in the same suffix. Keep the longest suffixes first so a
# ``ForConditionalGeneration`` match beats a generic ``ForGeneration``.
_ARCHITECTURE_SUFFIX_TASKS: tuple[tuple[str, str], ...] = (
    ("ForConditionalGeneration", TASK_SEQ2SEQ_LM),
    ("ForSeq2SeqLM", TASK_SEQ2SEQ_LM),
    ("ForSequenceClassification", TASK_SEQ_CLASSIFICATION),
    ("ForTokenClassification", TASK_TOKEN_CLASSIFICATION),
    ("ForCausalLM", TASK_CAUSAL_LM),
    ("LMHeadModel", TASK_CAUSAL_LM),  # GPT-2 historic naming
)


def detect_task_type(model: "nn.Module") -> str:
    """Return the canonical task-type string for ``model``.

    Inspection order matches the module docstring. Always returns one of
    the ``TASK_*`` constants; defaults to :data:`TASK_CAUSAL_LM` so the
    profiler keeps its prior behaviour when detection cannot conclude.
    """
    cfg = getattr(model, "config", None)

    # 1. config.architectures — most authoritative; HF stamps the
    #    concrete class name(s) here.
    archs = getattr(cfg, "architectures", None) if cfg is not None else None
    if archs:
        for arch in archs:
            for suffix, task in _ARCHITECTURE_SUFFIX_TASKS:
                if isinstance(arch, str) and arch.endswith(suffix):
                    return task

    # 2. is_encoder_decoder — covers T5/BART/etc. whose architectures
    #    attribute might be missing in trimmed configs.
    if cfg is not None and getattr(cfg, "is_encoder_decoder", False):
        return TASK_SEQ2SEQ_LM

    # 3. Module-class fallback for models constructed without
    #    config.architectures populated (common in tests and tiny
    #    randomly-initialised models).
    cls_name = type(model).__name__
    for suffix, task in _ARCHITECTURE_SUFFIX_TASKS:
        if cls_name.endswith(suffix):
            return task

    # 4. Default — preserve the legacy causal-LM behaviour.
    return TASK_CAUSAL_LM


# ---- batch factories ----------------------------------------------------

BatchFactory = Callable[["nn.Module", int, int, "torch.device | str"], dict]


def _infer_vocab_size(model: "nn.Module") -> int:
    """Best-effort vocab size from common HF config shapes."""
    from torch import nn as _nn

    cfg = getattr(model, "config", None)
    for attr in ("vocab_size", "n_vocab", "vocabulary_size"):
        if cfg is not None and hasattr(cfg, attr):
            val = getattr(cfg, attr)
            if isinstance(val, int) and val > 0:
                return val
    # Fallback: peek at the first Embedding layer.
    for m in model.modules():
        if isinstance(m, _nn.Embedding):
            return int(m.num_embeddings)
    return 1024


def _infer_num_labels(model: "nn.Module", default: int = 2) -> int:
    """Best-effort label count for classification heads.

    Reads ``config.num_labels`` first (HF's canonical attribute). Falls
    back to inspecting the head's ``out_features`` and finally to
    ``default`` (binary classification).
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        n = getattr(cfg, "num_labels", None)
        if isinstance(n, int) and n > 0:
            return n
    # Walk the model for the last Linear; HF classifiers typically end in
    # ``classifier`` (Bert) or ``score`` (Llama-for-classification).
    last_linear_out: int | None = None
    from torch import nn as _nn

    for m in model.modules():
        if isinstance(m, _nn.Linear):
            last_linear_out = int(m.out_features)
    if last_linear_out is not None and last_linear_out > 0:
        return last_linear_out
    return default


def causal_lm_batch_factory(
    model: "nn.Module",
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
) -> dict:
    """Causal-LM batch: ``input_ids`` + ``labels`` of identical shape.

    Preserves the exact behaviour of the legacy ``_dummy_batch`` so
    existing causal-LM calibration paths see no change. Note that
    ``attention_mask`` is intentionally OMITTED — the cached profiler
    fingerprint is keyed off the *batch keys*, and adding a mask would
    invalidate every cached trace from prior runs without any
    corresponding accuracy gain (HF causal LMs synthesize a default
    mask when none is supplied).
    """
    import torch

    vocab_size = _infer_vocab_size(model)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def seq_classification_batch_factory(
    model: "nn.Module",
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
) -> dict:
    """Sequence-classification batch: ``input_ids`` + per-sequence labels.

    Includes ``attention_mask`` because BERT-style encoders compute the
    pooled representation as a masked mean over the sequence dimension
    and HF errors out without one on some checkpoints.

    Label shape/dtype follows ``model.config.problem_type`` so we exercise
    the same loss path the real training run would:

    * ``"regression"`` — float tensor of shape ``(batch_size,)`` for
      single-target regression or ``(batch_size, num_labels)`` for
      multi-target regression (HF uses MSE; integer labels would either
      crash or silently cast).
    * ``"multi_label_classification"`` — float tensor of shape
      ``(batch_size, num_labels)`` with 0/1 entries (HF uses BCE-with-logits).
    * Anything else (single-label / unset) — long tensor of shape
      ``(batch_size,)`` drawn uniformly over ``[0, num_labels)``.
    """
    import torch

    vocab_size = _infer_vocab_size(model)
    num_labels = _infer_num_labels(model)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    cfg = getattr(model, "config", None)
    problem_type = getattr(cfg, "problem_type", None) if cfg is not None else None
    inferred_regression = problem_type == "regression" or (
        problem_type is None and num_labels == 1
    )
    if inferred_regression:
        # Multi-target regression uses (batch_size, num_labels); single-target
        # uses (batch_size,). HF's MSELoss path squeezes/handles both, but the
        # shapes must match num_labels to avoid broadcasting bugs / crashes.
        regression_shape = (batch_size, num_labels) if num_labels > 1 else (batch_size,)
        labels = torch.randn(
            regression_shape,
            device=device,
            dtype=torch.float,
        )
    elif problem_type == "multi_label_classification":
        labels = torch.randint(
            low=0,
            high=2,
            size=(batch_size, max(num_labels, 1)),
            device=device,
            dtype=torch.long,
        ).to(dtype=torch.float)
    else:
        labels = torch.randint(
            low=0,
            high=max(num_labels, 1),
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def token_classification_batch_factory(
    model: "nn.Module",
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
) -> dict:
    """Token-classification batch: per-token integer labels.

    Labels are shape ``(batch_size, seq_len)``. We deliberately do NOT
    set any positions to ``-100`` (HF's "ignore" index) — every token
    contributes to the loss so the gradient graph the profiler walks
    has the same fan-out as a real training batch.
    """
    import torch

    vocab_size = _infer_vocab_size(model)
    num_labels = _infer_num_labels(model)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
    labels = torch.randint(
        low=0,
        high=max(num_labels, 1),
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def seq2seq_lm_batch_factory(
    model: "nn.Module",
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
) -> dict:
    """Encoder-decoder batch: encoder ``input_ids`` + decoder ``labels``.

    HF seq2seq models accept ``labels`` directly and internally derive
    ``decoder_input_ids`` by right-shifting them with the model's
    ``decoder_start_token_id``. We keep encoder and decoder lengths
    equal because the profiler's cache key only carries a single
    ``seq_len``; a future extension can split this if needed.

    We also synthesize ``decoder_input_ids`` explicitly here. Models
    whose config has ``decoder_start_token_id is None`` (a small but
    real subset — some custom checkpoints, encoder-only-style heads
    pretending to be seq2seq) raise ``ValueError`` inside the model's
    own ``shift_tokens_right`` helper, breaking the profile loop. We
    prefer the model's canonical
    ``prepare_decoder_input_ids_from_labels`` helper when present
    (BART, T5, EncoderDecoderModel, ...) so we benefit from any
    model-specific shift logic; otherwise we right-shift ``labels``
    ourselves with a best-effort start-token id.
    """
    import torch

    vocab_size = _infer_vocab_size(model)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
    labels = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    # Prefer the model's canonical helper, which encodes any
    # checkpoint-specific quirks (e.g. T5's pad-token handling). Fall
    # back to a manual right-shift with a best-effort start-token id
    # for models that do not expose the helper.
    prepare = getattr(model, "prepare_decoder_input_ids_from_labels", None)
    if callable(prepare):
        decoder_input_ids = prepare(labels)
    else:
        cfg = getattr(model, "config", None)
        start_id = getattr(cfg, "decoder_start_token_id", None)
        if start_id is None:
            start_id = getattr(cfg, "bos_token_id", None)
        if start_id is None:
            start_id = getattr(cfg, "eos_token_id", None)
        if start_id is None:
            start_id = getattr(cfg, "pad_token_id", None)
        if start_id is None:
            start_id = 0
        decoder_input_ids = torch.empty_like(labels)
        decoder_input_ids[:, 0] = int(start_id)
        if seq_len > 1:
            decoder_input_ids[:, 1:] = labels[:, :-1]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


# ---- public registry ----------------------------------------------------

_DEFAULT_FACTORIES: dict[str, BatchFactory] = {
    TASK_CAUSAL_LM: causal_lm_batch_factory,
    TASK_SEQ_CLASSIFICATION: seq_classification_batch_factory,
    TASK_TOKEN_CLASSIFICATION: token_classification_batch_factory,
    TASK_SEQ2SEQ_LM: seq2seq_lm_batch_factory,
}

# Module-level dict so users (or another integration) can register a
# custom factory. The default mapping is restored by
# :func:`reset_factories` (test-only convenience).
_FACTORIES: dict[str, BatchFactory] = dict(_DEFAULT_FACTORIES)


def register_factory(task_type: str, factory: BatchFactory) -> None:
    """Register (or override) the batch factory for ``task_type``."""
    _FACTORIES[task_type] = factory


def reset_factories() -> None:
    """Restore the default factory registry. Test-only convenience."""
    _FACTORIES.clear()
    _FACTORIES.update(_DEFAULT_FACTORIES)


def get_factory(task_type: str) -> BatchFactory:
    """Return the registered factory for ``task_type``.

    Falls back to the causal-LM factory for unknown task types so the
    profiler degrades gracefully instead of raising.
    """
    factory = _FACTORIES.get(task_type)
    if factory is None:
        LOG.debug(
            "ProTrain batch_factory: no factory registered for task_type=%r; "
            "falling back to causal-LM",
            task_type,
        )
        factory = _FACTORIES[TASK_CAUSAL_LM]
    return factory


def build_batch(
    model: "nn.Module",
    batch_size: int,
    seq_len: int,
    device: "torch.device | str",
    *,
    task_type: str | None = None,
) -> dict:
    """Build a sample batch appropriate for ``model``'s task type.

    Parameters
    ----------
    model:
        The model that will receive the batch via ``model(**batch)``.
    batch_size, seq_len:
        Batch shape — passed through to the per-task factory.
    device:
        Target device for all tensors in the batch.
    task_type:
        Optional override. When ``None`` (default) the task type is
        detected via :func:`detect_task_type`.

    Returns
    -------
    dict
        Keyword-argument batch suitable for ``model(**batch)``. The
        returned dict always contains a ``labels`` entry so the profiler
        can synthesize a backward pass without further inspection.
    """
    if task_type is None:
        task_type = detect_task_type(model)
    factory = get_factory(task_type)
    batch = factory(model, batch_size, seq_len, device)
    factory_id = getattr(factory, "__qualname__", None) or repr(factory)
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"batch_factory for task_type={task_type!r} ({factory_id}) "
            f"must return a mapping, got {type(batch).__name__}"
        )
    # Normalize ``Mapping`` subclasses (e.g. ``BatchEncoding``) into a
    # plain ``dict`` so the rest of the function (and downstream
    # consumers expecting ``dict`` semantics) keep working unchanged.
    batch = dict(batch)
    if "labels" not in batch:
        raise ValueError(
            f"batch_factory for task_type={task_type!r} ({factory_id}) "
            f"returned a mapping without a 'labels' key; the profiler "
            f"requires 'labels' to synthesize a backward pass "
            f"(got keys: {sorted(batch.keys())!r})"
        )
    return batch


def factories_view() -> Mapping[str, BatchFactory]:
    """Return a read-only view of the current factory registry.

    Exposed for tests / introspection. Mutating the returned mapping is
    a no-op on the registry.
    """
    return dict(_FACTORIES)


__all__ = [
    "BatchFactory",
    "KNOWN_TASKS",
    "TASK_CAUSAL_LM",
    "TASK_SEQ2SEQ_LM",
    "TASK_SEQ_CLASSIFICATION",
    "TASK_TOKEN_CLASSIFICATION",
    "build_batch",
    "causal_lm_batch_factory",
    "detect_task_type",
    "factories_view",
    "get_factory",
    "register_factory",
    "reset_factories",
    "seq2seq_lm_batch_factory",
    "seq_classification_batch_factory",
    "token_classification_batch_factory",
]
