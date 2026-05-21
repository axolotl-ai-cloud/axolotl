"""Task-type-aware sample batch construction for the calibration profiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

LOG = get_logger(__name__)


# String constants so callers pass HF auto-class names without enum imports.
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

# Suffix-based HF class-name match; longest suffixes first.
_ARCHITECTURE_SUFFIX_TASKS: tuple[tuple[str, str], ...] = (
    ("ForConditionalGeneration", TASK_SEQ2SEQ_LM),
    ("ForSeq2SeqLM", TASK_SEQ2SEQ_LM),
    ("ForSequenceClassification", TASK_SEQ_CLASSIFICATION),
    ("ForTokenClassification", TASK_TOKEN_CLASSIFICATION),
    ("ForCausalLM", TASK_CAUSAL_LM),
    ("LMHeadModel", TASK_CAUSAL_LM),  # GPT-2 historic naming
)


def detect_task_type(model: "nn.Module") -> str:
    """Return canonical task-type for model (config.architectures → is_encoder_decoder → causal_lm)."""
    cfg = getattr(model, "config", None)

    # config.architectures is authoritative when present.
    archs = getattr(cfg, "architectures", None) if cfg is not None else None
    if archs:
        for arch in archs:
            for suffix, task in _ARCHITECTURE_SUFFIX_TASKS:
                if isinstance(arch, str) and arch.endswith(suffix):
                    return task

    # Module-class check before is_encoder_decoder so T5ForSequenceClassification etc. don't get misrouted to seq2seq_lm.
    cls_name = type(model).__name__
    for suffix, task in _ARCHITECTURE_SUFFIX_TASKS:
        if cls_name.endswith(suffix):
            return task

    if cfg is not None and getattr(cfg, "is_encoder_decoder", False):
        return TASK_SEQ2SEQ_LM

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
    """Best-effort label count: config.num_labels → last Linear out_features → default."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        n = getattr(cfg, "num_labels", None)
        if isinstance(n, int) and n > 0:
            return n
    # Last Linear out_features matches HF classifier head (Bert: classifier, Llama: score).
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
    """Causal-LM batch: input_ids + labels (no attention_mask to keep cache fingerprint stable)."""
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
    """Seq-classification batch; label shape/dtype follows config.problem_type."""
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
        # Match num_labels to avoid broadcasting bugs in HF's MSELoss path.
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
    """Token-classification batch with per-token labels; no -100 ignore positions."""
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
    """Seq2seq batch with explicit decoder_input_ids for configs lacking decoder_start_token_id."""
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
    # Prefer model's canonical helper; fall back to manual right-shift on TypeError/ValueError.
    prepare = getattr(model, "prepare_decoder_input_ids_from_labels", None)
    decoder_input_ids = None
    if callable(prepare):
        try:
            decoder_input_ids = prepare(labels)
        except (TypeError, ValueError):
            decoder_input_ids = None

    if decoder_input_ids is None:
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

# Module-level dict; reset_factories() restores defaults.
_FACTORIES: dict[str, BatchFactory] = dict(_DEFAULT_FACTORIES)


def register_factory(task_type: str, factory: BatchFactory) -> None:
    """Register (or override) the batch factory for ``task_type``."""
    _FACTORIES[task_type] = factory


def reset_factories() -> None:
    """Restore the default factory registry. Test-only convenience."""
    _FACTORIES.clear()
    _FACTORIES.update(_DEFAULT_FACTORIES)


def get_factory(task_type: str) -> BatchFactory:
    """Return the registered factory; falls back to causal-LM for unknown tasks."""
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
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
        raise ValueError(f"seq_len must be a positive int, got {seq_len!r}")
    factory: BatchFactory
    if task_type is None:
        # Auto-detect path: ``get_factory`` may fall back to the
        # causal-LM factory when ``detect_task_type`` returns a string
        # that wasn't registered (defensive — for forward-compat with
        # new task heads). The fallback is only applied here, on the
        # auto-detect path; an explicitly-passed ``task_type`` below
        # gets a hard error so caller typos / stale overrides surface
        # immediately rather than silently profiling the wrong graph.
        task_type = detect_task_type(model)
        factory = get_factory(task_type)
    else:
        # Bind to a separately-typed Optional first so mypy can narrow
        # the assignment to ``factory`` after the ``is None`` raise —
        # otherwise the ``factory: BatchFactory`` annotation above
        # collides with ``_FACTORIES.get`` returning ``BatchFactory | None``.
        factory_or_none = _FACTORIES.get(task_type)
        if factory_or_none is None:
            raise ValueError(
                f"build_batch: unknown task_type={task_type!r}. Registered "
                f"types: {sorted(_FACTORIES)}. Pass task_type=None to "
                "auto-detect from the model, or register the factory via "
                "register_factory(task_type, fn) before calling."
            )
        factory = factory_or_none
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
