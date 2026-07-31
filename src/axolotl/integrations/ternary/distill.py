"""In-process distillation: a frozen FP teacher beside the ternary student."""

from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from typing_extensions import override

from axolotl.core.trainers import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .args import DistillSchedule, HiddenLoss
from .callbacks import DistillAnchorCallback

LOG = get_logger(__name__)

IGNORE_INDEX: int = -100

# one logits chunk is capped at this many elements, so a 256k vocab still fits
CHUNK_ELEMENT_BUDGET: int = 1 << 23
MIN_CHUNK_TOKENS: int = 32
MAX_CHUNK_TOKENS: int = 1024

TEACHER_INPUT_KEYS: tuple[str, ...] = ("input_ids", "attention_mask", "position_ids")
ATTN_PROJECTIONS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")

_FORWARD_PARAMETERS: dict[type, frozenset[str]] = {}


@dataclass
class TernaryDistillTrainingArgsMixin:
    """Distillation knobs carried onto `TrainingArguments` by the plugin."""

    ternary_distill_teacher_model: str | None = None
    ternary_distill_logits_weight: float | None = None
    ternary_distill_logits_temperature: float | None = None
    ternary_distill_hidden_weight: float | None = None
    ternary_distill_hidden_loss: HiddenLoss | None = None
    ternary_distill_hidden_huber_delta: float | None = None
    ternary_distill_attn_relation_layer: int | None = None
    ternary_distill_teacher_device_map: str | None = None
    ternary_distill_prefetch_teacher: bool = True
    ternary_distill_teacher_prefetch_depth: int | None = None
    ternary_distill_schedule: DistillSchedule | None = None
    ternary_distill_anchor_start: float | None = None


def _teacher_keys(inputs: dict) -> tuple[str, ...]:
    return tuple(
        sorted(
            k for k in inputs if k in TEACHER_INPUT_KEYS and torch.is_tensor(inputs[k])
        )
    )


def _ident(inputs: dict, keys: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    return tuple((k, id(inputs[k])) for k in keys)


class _TeacherPrefetch:
    """FIFO of teacher final hidden states, enqueued at batch-fetch time.

    The teacher is frozen, so its forwards can run on their own device while the
    student computes, and any number of batches may be in flight (no staleness).
    `submit_many` fuses same-shape batches into one forward for throughput; a slot
    is only honored when the loss receives the exact tensors it was computed from
    (object identity per teacher input key), so any input fixup between fetch and
    loss silently falls back to the inline path.
    """

    def __init__(self) -> None:
        self._slots: deque[tuple[tuple[tuple[str, int], ...], torch.Tensor]] = deque()

    def submit_many(self, teacher: PreTrainedModel, batches: list[dict]) -> None:
        batches = [b for b in batches if "input_ids" in _teacher_keys(b)]
        if not batches:
            return
        keys = _teacher_keys(batches[0])
        same_keys = all(_teacher_keys(b) == keys for b in batches)
        if len(batches) > 1 and same_keys:
            if "position_ids" in keys and all(
                b["input_ids"].shape[0] == 1 for b in batches
            ):
                # packed rows: sequence resets mark doc boundaries, so K rows
                # concatenated along the sequence are one bigger packed row in
                # the varlen kernels' native flat batch-1 form
                self._submit_fused(teacher, batches, keys, dim=1)
                return
            if "position_ids" not in keys and all(
                all(b[k].shape[1:] == batches[0][k].shape[1:] for k in keys)
                for b in batches
            ):
                self._submit_fused(teacher, batches, keys, dim=0)
                return
        for batch in batches:
            self.submit(teacher, batch)

    def _submit_fused(
        self,
        teacher: PreTrainedModel,
        batches: list[dict],
        keys: tuple[str, ...],
        dim: int,
    ) -> None:
        device = next(teacher.parameters()).device
        fused = {
            k: torch.cat([b[k] for b in batches], dim=dim).to(device, non_blocking=True)
            for k in keys
        }
        with torch.no_grad():
            hidden = _last_hidden(teacher(**_forward_inputs(fused, teacher)), teacher)
        offset = 0
        for batch in batches:
            span = batch["input_ids"].shape[dim]
            piece = (
                hidden[:, offset : offset + span]
                if dim == 1
                else hidden[offset : offset + span]
            )
            self._slots.append((_ident(batch, keys), piece))
            offset += span

    def submit(self, teacher: PreTrainedModel, inputs: dict) -> None:
        keys = _teacher_keys(inputs)
        if "input_ids" not in keys:
            return
        device = next(teacher.parameters()).device
        moved = {k: inputs[k].to(device, non_blocking=True) for k in keys}
        with torch.no_grad():
            hidden = _last_hidden(teacher(**_forward_inputs(moved, teacher)), teacher)
        self._slots.append((_ident(inputs, keys), hidden))

    def take(self, inputs: dict) -> torch.Tensor | None:
        if not self._slots:
            return None
        ident, hidden = self._slots.popleft()
        keys = tuple(k for k, _ in ident)
        if _teacher_keys(inputs) != keys or _ident(inputs, keys) != ident:
            # order broke; anything still queued is for batches we cannot match
            self._slots.clear()
            return None
        return hidden


class _TeacherPrefetchLoader:
    """Wraps the train dataloader to enqueue teacher forwards at fetch time.

    With `depth > 1`, up to `depth` upcoming batches are drained from the source
    and handed to the trainer as one fused teacher submission before being
    yielded onward in order.
    """

    def __init__(
        self, loader, trainer: "TernaryDistillTrainer", depth: int = 1
    ) -> None:
        self.loader = loader
        self.trainer = trainer
        self.depth = max(1, depth)

    def __iter__(self):
        source = iter(self.loader)
        while True:
            window = list(islice(source, self.depth))
            if not window:
                return
            self.trainer._submit_teacher_prefetch(window)
            yield from window

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str):
        return getattr(self.loader, name)


class TernaryDistillTrainer(AxolotlTrainer):
    """Trains the ternary student against a frozen in-process copy of the FP base.

    Loss is `(1 - α)·CE + α·τ²·KL(teacher/τ ‖ student/τ)` computed in token chunks so
    large vocabularies never materialize a full `(T, V)` logits pair, plus optional
    cosine hidden-state KD and single-layer attention-relation KD. The teacher runs
    in eval mode under `no_grad`.

    Every KD term is scaled by a schedule multiplier `DistillAnchorCallback` pushes in
    at each step. Under `schedule: anchored` it is 0 for the CE-dominant bulk of the
    run, which makes those steps pure-CE *and* teacher-free: nothing loads the teacher
    until the anchor ramp starts.
    """

    def __init__(self, *args, teacher_model: PreTrainedModel | None = None, **kwargs):
        """Args: teacher_model: preloaded frozen teacher; loaded from the training args when `None`."""
        super().__init__(*args, **kwargs)
        # the distillation loss is a plain token mean, so let the Trainer normalize it
        self.model_accepts_loss_kwargs = False
        self._init_distill_state(teacher_model)
        if self.distill_schedule == "anchored":
            self.add_callback(
                DistillAnchorCallback(trainer=self, anchor_start=self.anchor_start)
            )

    def _init_distill_state(self, teacher_model: PreTrainedModel | None = None) -> None:
        """Read the distillation knobs off `self.args` and reset the teacher state."""
        args = self.args
        weight = _arg(args, "ternary_distill_logits_weight", 1.0)
        if not 0.0 <= weight <= 1.0:
            LOG.warning(
                f"ternary: distill.logits_weight {weight} is outside [0, 1]; clamping "
                "(it is the KD/CE mixing coefficient, not a free multiplier)"
            )
            weight = min(max(weight, 0.0), 1.0)
        self.logits_weight = weight
        self.logits_temperature = _arg(args, "ternary_distill_logits_temperature", 2.0)
        self.hidden_weight = _arg(args, "ternary_distill_hidden_weight", 0.0)
        self.hidden_loss = (
            getattr(args, "ternary_distill_hidden_loss", None) or "cosine"
        )
        self.hidden_huber_delta = _arg(args, "ternary_distill_hidden_huber_delta", 1.0)
        self.attn_relation_layer = getattr(
            args, "ternary_distill_attn_relation_layer", None
        )
        self.teacher_model_name = getattr(args, "ternary_distill_teacher_model", None)
        self.teacher_device_map = getattr(
            args, "ternary_distill_teacher_device_map", None
        )
        # a teacher pinned off the student's device claims that GPU for itself;
        # DataParallel would replicate the student onto it and OOM. The base
        # Trainer already cached a batch size scaled by n_gpu, so refresh it too.
        if self.teacher_device_map and getattr(args, "_n_gpu", 0) > 1:
            args._n_gpu = 1
            self._train_batch_size = args.train_batch_size
        self.prefetch_teacher = bool(
            getattr(args, "ternary_distill_prefetch_teacher", True)
        )
        self.teacher_prefetch_depth = int(
            getattr(args, "ternary_distill_teacher_prefetch_depth", None) or 1
        )
        self._teacher_prefetch = _TeacherPrefetch()
        schedule = getattr(args, "ternary_distill_schedule", None) or "constant"
        self.distill_schedule = schedule
        self.anchor_start = _arg(args, "ternary_distill_anchor_start", 0.9)
        # anchored runs start teacher-free, so a step taken before the callback has
        # pushed a multiplier cannot pull the teacher into memory
        self.distill_multiplier = 0.0 if schedule == "anchored" else 1.0

        self._teacher: PreTrainedModel | None = teacher_model
        self._teacher_ready = False
        self._hook_handles: list[Any] = []
        self._captured: dict[str, torch.Tensor] = {}
        self._head_dim: int | None = None

    @override
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Return the combined CE + distillation loss (and outputs when requested).

        The student's own `(T, V)` logits are never computed: the forward keeps a
        single position (`logits_to_keep=1`) and every loss term is derived from the
        final hidden states.
        """
        inputs = dict(inputs)
        inputs.pop("num_items_in_batch", None)
        # the axolotl-specific input fixups the replaced `compute_loss` would skip
        self.prepare_loss_inputs(model, inputs)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise KeyError("ternary in-process distillation requires batch `labels`")

        student = self._unwrap(model)
        multiplier = self.distill_multiplier
        if multiplier <= 0.0:
            return self._ce_loss(model, student, inputs, labels, return_outputs)

        teacher = self._ensure_teacher(student)
        self._captured.clear()
        if self.attn_relation_layer is not None:
            self._register_relation_hooks(student, teacher)
        try:
            return self._distill_loss(
                model, student, teacher, inputs, labels, return_outputs, multiplier
            )
        finally:
            # the captures pin a (batch, seq, heads*dim) tensor per projection
            self._captured.clear()
            self._release_hooks()

    def _ce_loss(
        self,
        model: PreTrainedModel,
        student: nn.Module,
        inputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        return_outputs: bool,
    ):
        """Chunked CE alone: the schedule holds every KD term at 0, so no teacher runs."""
        outputs = model(**_forward_inputs(inputs, student))
        student_hidden = _last_hidden(outputs, model)
        student_head = _resolve_lm_head(student)

        shifted = labels[..., 1:]
        supervised = shifted != IGNORE_INDEX
        ce, _ = _chunked_ce_kd(
            student_hidden[..., :-1, :][supervised],
            None,
            student_head.weight,
            getattr(student_head, "bias", None),
            None,
            None,
            shifted[supervised],
            self.logits_temperature,
            _chunk_size(student_head.weight.shape[0]),
        )

        self._store_metrics({"ternary/ce": ce.detach().item()}, 0.0, model.training)
        return (ce, outputs) if return_outputs else ce

    def _distill_loss(
        self,
        model: PreTrainedModel,
        student: nn.Module,
        teacher: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        return_outputs: bool,
        multiplier: float = 1.0,
    ):
        outputs = model(**_forward_inputs(inputs, student))
        student_hidden = _last_hidden(outputs, model)

        prefetched = (
            self._teacher_prefetch.take(inputs) if self.prefetch_teacher else None
        )
        if prefetched is not None:
            teacher_hidden = prefetched
        else:
            teacher_device = next(teacher.parameters()).device
            teacher_inputs = {
                key: value.to(teacher_device)
                for key, value in inputs.items()
                if key in TEACHER_INPUT_KEYS
            }
            with torch.no_grad():
                teacher_outputs = teacher(**_forward_inputs(teacher_inputs, teacher))
            # left on the teacher's device: the chunked KD runs the teacher head
            # there and ships only per-chunk log-probs back, never a
            # (vocab, hidden) head copy
            teacher_hidden = _last_hidden(teacher_outputs, teacher)

        student_head = _resolve_lm_head(student)
        teacher_head = _resolve_lm_head(teacher)

        shifted = labels[..., 1:]
        supervised = shifted != IGNORE_INDEX
        teacher_supervised = supervised.to(teacher_hidden.device)
        chunk_size = _chunk_size(student_head.weight.shape[0])
        ce, kd = _chunked_ce_kd(
            student_hidden[..., :-1, :][supervised],
            teacher_hidden[..., :-1, :][teacher_supervised],
            student_head.weight,
            getattr(student_head, "bias", None),
            teacher_head.weight,
            getattr(teacher_head, "bias", None),
            shifted[supervised],
            self.logits_temperature,
            chunk_size,
        )

        alpha = multiplier * self.logits_weight
        loss = (1.0 - alpha) * ce + alpha * self.logits_temperature**2 * kd
        metrics = {
            "ternary/ce": ce.detach().item(),
            "ternary/kd_logits": kd.detach().item(),
            "ternary/teacher_prefetch_hit": 1.0 if prefetched is not None else 0.0,
        }

        mask = inputs.get("attention_mask")
        if mask is not None and mask.dim() != 2:
            mask = None
        if self.hidden_weight:
            hidden = self.hidden_feature_kd(
                student_hidden,
                teacher_hidden.to(
                    device=student_hidden.device, dtype=student_hidden.dtype
                ),
                mask=mask,
            )
            loss = loss + multiplier * self.hidden_weight * hidden
            metrics["ternary/kd_hidden"] = hidden.detach().item()
        if self.attn_relation_layer is not None:
            relation = self._attn_relation_term(mask)
            loss = loss + multiplier * relation
            metrics["ternary/kd_attn"] = relation.detach().item()

        self._store_metrics(metrics, multiplier, model.training)
        return (loss, outputs) if return_outputs else loss

    def _store_metrics(
        self, metrics: dict[str, float], multiplier: float, training: bool
    ) -> None:
        if self.distill_schedule != "constant":
            metrics["ternary/distill_weight"] = multiplier
        self.store_metrics(metrics, train_eval="train" if training else "eval")

    def set_distill_multiplier(self, value: float) -> None:
        """Scale every KD term by `value` from this step on; 0 makes the step pure CE."""
        self.distill_multiplier = float(value)

    def chunked_logits_kd(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        lm_head: torch.nn.Module,
        temperature: float,
        chunk_size: int = 1024,
        teacher_lm_head: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """KL between teacher and student logits, accumulated over token chunks.

        Args:
            student_hidden: Hidden states feeding the student head, `(..., d)`.
            teacher_hidden: The teacher's hidden states, same shape.
            lm_head: The student's output head.
            temperature: Softmax temperature applied to both models' logits.
            chunk_size: Tokens per chunk; the peak logits buffer is `chunk_size x V`.
            teacher_lm_head: The teacher's (frozen) head; `lm_head` is used for both
                when omitted.

        Returns:
            Mean KL per token.
        """
        head = teacher_lm_head if teacher_lm_head is not None else lm_head
        _, kd = _chunked_ce_kd(
            student_hidden.reshape(-1, student_hidden.shape[-1]),
            teacher_hidden.reshape(-1, teacher_hidden.shape[-1]),
            lm_head.weight,
            getattr(lm_head, "bias", None),
            head.weight,
            getattr(head, "bias", None),
            None,
            temperature,
            chunk_size,
        )
        return kd

    def hidden_feature_kd(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Feature-KD between the two residual streams under `hidden_loss`.

        Every variant reduces the feature axis to one number per position and hands
        that to the same masked mean, so which tokens count never depends on the
        variant. The raw-residual variants are computed in fp32 on the *unnormalized*
        hidden states: their magnitudes are the signal `mse`/`huber` exist to see.

        Args:
            student_hidden: Student hidden states `(batch, seq, d)`.
            teacher_hidden: Teacher hidden states, same shape.
            mask: Attention mask; padded positions are excluded when given.
        """
        if self.hidden_loss == "cosine":
            return self.hidden_cosine_kd(student_hidden, teacher_hidden, mask=mask)
        student = student_hidden.float()
        teacher = teacher_hidden.float()
        if self.hidden_loss == "mse":
            per_feature = F.mse_loss(student, teacher, reduction="none")
        elif self.hidden_loss == "huber":
            per_feature = F.huber_loss(
                student, teacher, reduction="none", delta=self.hidden_huber_delta
            )
        else:
            raise ValueError(
                f"unknown ternary.distill.hidden_loss: {self.hidden_loss!r}"
            )
        return _masked_mean(per_feature.mean(dim=-1), mask)

    def hidden_cosine_kd(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean `1 - cos(student, teacher)` over the final hidden states.

        Args:
            student_hidden: Student hidden states `(batch, seq, d)`.
            teacher_hidden: Teacher hidden states, same shape.
            mask: Attention mask; padded positions are excluded when given.
        """
        distance = 1.0 - F.cosine_similarity(
            student_hidden.float(), teacher_hidden.float(), dim=-1
        )
        return _masked_mean(distance, mask)

    def attn_relation_kd(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MiniLM-style relation KD between one student/teacher attention projection.

        Experimental: relation KD on a single late layer is sensitive to the layer
        choice and can destabilize a run that is otherwise healing.

        Args:
            student_attn: Student projection output split per head,
                `(batch, heads, seq, head_dim)`.
            teacher_attn: The teacher's, same shape.
            mask: Attention mask; padded rows and columns are excluded when given.

        Returns:
            Mean KL between the teacher's and the student's scaled self-relation
            distributions.
        """
        batch, heads, seq, head_dim = student_attn.shape
        keep = None if mask is None else mask.to(torch.bool)[:, None, None, :]
        rows = batch * seq if keep is None else int(keep.sum())
        if not rows:
            return student_attn.sum() * 0.0

        recompute = torch.is_grad_enabled() and student_attn.requires_grad
        chunk = _head_chunk(seq)
        total: torch.Tensor | None = None
        for start in range(0, heads, chunk):
            arguments = (
                student_attn[:, start : start + chunk],
                teacher_attn[:, start : start + chunk],
                keep,
                head_dim**-0.5,
            )
            if recompute:
                value = checkpoint(_relation_kl_sum, *arguments, use_reentrant=False)
            else:
                value = _relation_kl_sum(*arguments)
            total = value if total is None else total + value
        return total / (heads * rows)  # type: ignore[operator]

    def _unwrap(self, model: nn.Module) -> nn.Module:
        accelerator = getattr(self, "accelerator", None)
        return model if accelerator is None else accelerator.unwrap_model(model)

    @override
    def train(self, *args, **kwargs):
        """Load the teacher before the training loop opens any ZeRO-3 init scope."""
        if not self._defers_teacher():
            self._ensure_teacher(self._unwrap(self.model))
        try:
            return super().train(*args, **kwargs)
        finally:
            self._release_hooks()

    def _defers_teacher(self) -> bool:
        """Whether the anchored schedule may hold the teacher back until the KD tail."""
        if self.distill_schedule != "anchored" or self._teacher is not None:
            return False
        if getattr(self, "is_deepspeed_enabled", False):
            # a mid-run from_pretrained is partitioned by the live ZeRO-3 config
            LOG.warning(
                "ternary: distill.schedule: anchored keeps the teacher loaded for the "
                "whole run under DeepSpeed; the CE-phase memory win needs FSDP or DDP"
            )
            return False
        LOG.info(
            "ternary: distill.schedule: anchored defers the teacher load to the KD tail "
            f"(anchor_start {self.anchor_start})"
        )
        return True

    def get_train_dataloader(self):
        loader = super().get_train_dataloader()
        if not self.prefetch_teacher:
            return loader
        return _TeacherPrefetchLoader(loader, self, depth=self.teacher_prefetch_depth)

    def _submit_teacher_prefetch(self, batches: list[dict]) -> None:
        if (
            not self.prefetch_teacher
            or self.distill_multiplier <= 0.0
            or self.attn_relation_layer is not None
            or not self._teacher_ready
            or self._teacher is None
        ):
            return
        try:
            self._teacher_prefetch.submit_many(self._teacher, batches)
        except Exception:  # noqa: BLE001 - an optimization, never fatal
            LOG.warning(
                "ternary: teacher prefetch failed; continuing inline", exc_info=True
            )
            self.prefetch_teacher = False

    def _ensure_teacher(self, student: PreTrainedModel) -> PreTrainedModel:
        """Load the frozen teacher, once."""
        if self._teacher_ready:
            return self._teacher  # type: ignore[return-value]
        teacher = self._teacher
        if teacher is None:
            teacher = self._load_teacher(student)
        teacher.eval()
        teacher.requires_grad_(False)
        self._teacher = teacher
        self._teacher_ready = True
        if self.attn_relation_layer is not None:
            self._head_dim = _head_dim(student)
        return teacher

    def _release_hooks(self) -> None:
        """Drop the relation-KD forward hooks so they cannot outlive the step."""
        while self._hook_handles:
            self._hook_handles.pop().remove()

    def _load_teacher(self, student: PreTrainedModel) -> PreTrainedModel:
        if not self.teacher_model_name:
            raise ValueError(
                "ternary.distill.mode: inprocess needs a teacher; set "
                "ternary.distill.teacher_model or base_model"
            )
        from transformers import AutoModelForCausalLM

        param = next(student.parameters())
        kwargs: dict[str, Any] = {"dtype": param.dtype}
        attn_implementation = getattr(
            getattr(student, "config", None), "_attn_implementation", None
        )
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        cfg = getattr(self, "axolotl_cfg", None)
        if cfg is not None and getattr(cfg, "trust_remote_code", False):
            kwargs["trust_remote_code"] = True
        if self.teacher_device_map:
            kwargs["device_map"] = self.teacher_device_map

        LOG.info(
            f"ternary: loading frozen in-process teacher {self.teacher_model_name} "
            f"({param.dtype}, device_map={self.teacher_device_map or 'student device'})"
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_name, **kwargs
        )
        if not self.teacher_device_map:
            teacher = teacher.to(param.device)
        return teacher

    def _register_relation_hooks(
        self, student: PreTrainedModel, teacher: PreTrainedModel
    ) -> None:
        for tag, model in (("student", student), ("teacher", teacher)):
            attention = _attention_module(model, self.attn_relation_layer)
            for projection in ATTN_PROJECTIONS:
                module = getattr(attention, projection, None)
                if module is None:
                    raise ValueError(
                        f"ternary.distill.attn_relation_layer needs separate "
                        f"{'/'.join(ATTN_PROJECTIONS)} modules; layer "
                        f"{self.attn_relation_layer} of the {tag} has none"
                    )
                self._hook_handles.append(
                    module.register_forward_hook(self._capture(f"{tag}.{projection}"))
                )

    def _capture(self, key: str):
        def hook(_module, _args, output):
            self._captured[key] = output

        return hook

    def _attn_relation_term(self, mask: torch.Tensor | None) -> torch.Tensor:
        total: torch.Tensor | None = None
        for projection in ATTN_PROJECTIONS:
            student = self._captured.get(f"student.{projection}")
            teacher = self._captured.get(f"teacher.{projection}")
            if student is None or teacher is None:
                raise RuntimeError(
                    f"ternary: attention-relation KD captured no {projection} output; "
                    "the configured layer did not run"
                )
            student = _split_heads(student, self._head_dim)
            teacher = _split_heads(teacher.to(student.device), self._head_dim)
            value = self.attn_relation_kd(student, teacher.to(student.dtype), mask=mask)
            total = value if total is None else total + value
        return total / len(ATTN_PROJECTIONS)  # type: ignore[operator]


def _arg(args: Any, name: str, default: float) -> float:
    value = getattr(args, name, None)
    return default if value is None else float(value)


def _forward_accepts(model: nn.Module, name: str) -> bool:
    model_cls = type(model)
    parameters = _FORWARD_PARAMETERS.get(model_cls)
    if parameters is None:
        try:
            parameters = frozenset(inspect.signature(model_cls.forward).parameters)
        except (TypeError, ValueError):
            parameters = frozenset()
        _FORWARD_PARAMETERS[model_cls] = parameters
    return name in parameters


def _forward_inputs(inputs: dict[str, torch.Tensor], model: nn.Module) -> dict:
    """Return `inputs` plus the flags that expose hidden states without full logits."""
    forward = dict(inputs)
    forward["output_hidden_states"] = True
    forward["return_dict"] = True
    # no loss path decodes, and a separately loaded teacher keeps its config
    # default of use_cache=True — without this it allocates a cache per forward
    if _forward_accepts(model, "use_cache"):
        forward["use_cache"] = False
    if _forward_accepts(model, "logits_to_keep"):
        forward["logits_to_keep"] = 1
    return forward


def _last_hidden(outputs: Any, model: nn.Module) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError(
            f"{type(model).__name__}.forward returned no hidden_states; in-process "
            "distillation needs them to compute the chunked losses"
        )
    return hidden_states[-1]


def _resolve_lm_head(model: nn.Module) -> nn.Linear:
    """Return the output head of a causal LM, looking through common wrappers."""
    base = model
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    if hasattr(base, "language_model") and hasattr(base.language_model, "lm_head"):
        return base.language_model.lm_head
    if hasattr(base, "lm_head"):
        return base.lm_head
    raise AttributeError(f"could not find lm_head on {type(model).__name__}")


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder layer list of a causal LM."""
    base = model
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    for path in ("model.layers", "model.model.layers", "language_model.layers"):
        current: Any = base
        for attribute in path.split("."):
            current = getattr(current, attribute, None)
            if current is None:
                break
        if isinstance(current, nn.ModuleList):
            return current
    raise AttributeError(f"could not find decoder layers on {type(model).__name__}")


def _attention_module(model: nn.Module, layer_index: int | None) -> nn.Module:
    layers = _decoder_layers(model)
    if layer_index is None or not -len(layers) <= layer_index < len(layers):
        raise ValueError(
            f"ternary.distill.attn_relation_layer {layer_index} is out of range for "
            f"{len(layers)} layers"
        )
    layer = layers[layer_index]
    return getattr(layer, "self_attn", layer)


def _head_dim(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    dim = getattr(config, "head_dim", None)
    if dim:
        return int(dim)
    hidden = getattr(config, "hidden_size", None)
    heads = getattr(config, "num_attention_heads", None)
    if not hidden or not heads:
        raise ValueError(
            "ternary: attention-relation KD needs head_dim (or hidden_size and "
            f"num_attention_heads) on the {type(model).__name__} config"
        )
    return int(hidden) // int(heads)


def _split_heads(tensor: torch.Tensor, head_dim: int | None) -> torch.Tensor:
    """Reshape a `(batch, seq, heads * head_dim)` projection output to per-head."""
    if not head_dim or tensor.shape[-1] % head_dim:
        raise ValueError(
            f"projection width {tensor.shape[-1]} is not a multiple of head_dim {head_dim}"
        )
    batch, seq, width = tensor.shape
    return tensor.view(batch, seq, width // head_dim, head_dim).transpose(1, 2)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of `values` over the positions `mask` keeps."""
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype).expand_as(values)
    total = weights.sum()
    if not total:
        return values.sum() * 0.0
    return (values * weights).sum() / total


def _chunk_size(vocab_size: int) -> int:
    """Tokens per logits chunk for `vocab_size`, bounded by the element budget."""
    tokens = CHUNK_ELEMENT_BUDGET // max(vocab_size, 1)
    return max(MIN_CHUNK_TOKENS, min(MAX_CHUNK_TOKENS, tokens))


def _head_chunk(seq: int) -> int:
    """Heads per relation chunk; each holds a `(chunk, seq, seq)` relation matrix."""
    return max(1, CHUNK_ELEMENT_BUDGET // max(seq * seq, 1))


def _relation_kl_sum(
    student_attn: torch.Tensor,
    teacher_attn: torch.Tensor,
    keep: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Summed per-row relation KL for one head chunk; the only place `(T, T)` exists."""
    student = student_attn.float()
    teacher = teacher_attn.float()
    student_rel = torch.matmul(student, student.transpose(-1, -2)) * scaling
    teacher_rel = torch.matmul(teacher, teacher.transpose(-1, -2)) * scaling
    if keep is not None:
        student_rel = student_rel.masked_fill(~keep, float("-inf"))
        teacher_rel = teacher_rel.masked_fill(~keep, float("-inf"))

    student_logprobs = F.log_softmax(student_rel, dim=-1)
    teacher_logprobs = F.log_softmax(teacher_rel, dim=-1)
    difference = teacher_logprobs - student_logprobs
    if keep is not None:
        # -inf - -inf on the masked columns is NaN, and its weight is 0 anyway
        difference = difference.masked_fill(~keep, 0.0)
    per_row = (teacher_logprobs.exp() * difference).sum(-1)
    if keep is not None:
        per_row = per_row * keep[:, :, 0, :].to(per_row.dtype)
    return per_row.sum()


def _chunk_terms(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    student_weight: torch.Tensor,
    student_bias: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    teacher_bias: torch.Tensor | None,
    targets: torch.Tensor | None,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed CE and KL for one token chunk; the only place logits exist."""
    student_logits = F.linear(student_hidden, student_weight, student_bias).float()
    if teacher_hidden is None or teacher_weight is None:
        if targets is None:
            return student_logits.new_zeros(()), student_logits.new_zeros(())
        ce = F.cross_entropy(student_logits, targets, reduction="sum")
        return ce, student_logits.new_zeros(())

    with torch.no_grad():
        # the teacher head stays wherever the teacher lives; only this chunk's
        # log-probs cross the device boundary
        teacher_logits = F.linear(teacher_hidden, teacher_weight, teacher_bias).float()
        teacher_logprobs = F.log_softmax(teacher_logits / temperature, dim=-1).to(
            student_logits.device
        )

    if targets is None:
        ce = student_logits.new_zeros(())
    else:
        ce = F.cross_entropy(student_logits, targets, reduction="sum")

    student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
    kd = F.kl_div(student_logprobs, teacher_logprobs, reduction="sum", log_target=True)
    return ce, kd


def _chunked_ce_kd(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    student_weight: torch.Tensor,
    student_bias: torch.Tensor | None,
    teacher_weight: torch.Tensor | None,
    teacher_bias: torch.Tensor | None,
    targets: torch.Tensor | None,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean CE and mean KL over `(tokens, d)` hidden states, one chunk at a time.

    Each chunk is recomputed in the backward pass, so peak memory holds a single
    `(chunk_size, vocab)` logits pair instead of the full `(tokens, vocab)` one.
    A `None` teacher drops the KL term (and the teacher head) entirely.
    """
    # accelerate's mixed-precision wrapper returns fp32 hidden states while the
    # heads keep the model dtype; this loss runs outside autocast, so harmonize
    # here (a no-op when the trainer upcast the whole model to fp32)
    student_hidden = student_hidden.to(student_weight.dtype)
    if teacher_hidden is not None and teacher_weight is not None:
        teacher_hidden = teacher_hidden.to(teacher_weight.dtype)

    tokens = student_hidden.shape[0]
    if not tokens:
        zero = (student_hidden.sum() + student_weight.sum()) * 0.0
        return zero, zero

    recompute = torch.is_grad_enabled() and (
        student_hidden.requires_grad or student_weight.requires_grad
    )
    ce_total: torch.Tensor | None = None
    kd_total: torch.Tensor | None = None
    for start in range(0, tokens, max(chunk_size, 1)):
        stop = min(start + max(chunk_size, 1), tokens)
        arguments = (
            student_hidden[start:stop],
            None if teacher_hidden is None else teacher_hidden[start:stop],
            student_weight,
            student_bias,
            teacher_weight,
            teacher_bias,
            None if targets is None else targets[start:stop],
            temperature,
        )
        if recompute:
            ce, kd = checkpoint(_chunk_terms, *arguments, use_reentrant=False)
        else:
            ce, kd = _chunk_terms(*arguments)
        ce_total = ce if ce_total is None else ce_total + ce
        kd_total = kd if kd_total is None else kd_total + kd

    return ce_total / tokens, kd_total / tokens  # type: ignore[operator]
