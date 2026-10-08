"""Gradient-norm guards: whole-step outlier handling and per-tensor adaptive clipping."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from statistics import median
from typing import Any

import torch
from transformers.trainer_callback import ExportableState, TrainerCallback

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Scale from the median absolute deviation to a normal standard deviation.
_MAD_TO_STD = 1.4826

# Optimizer-state key of a tensor's running average of clipped gradient norms (as in OLMo).
GRAD_NORM_EMA_KEY = "grad_norm_exp_avg"


class StepOutlierState(TrainerCallback, ExportableState):
    """The outlier guard's history, saved in ``trainer_state.json`` with each checkpoint.

    It is an ``ExportableState`` callback, so the Trainer writes it to
    ``TrainerState.stateful_callbacks``; ``GradNormGuardMixin`` restores it on every
    ``resume_from_checkpoint``, whether or not ``restore_callback_states_from_checkpoint``
    is set. The history is global (identical on every rank), so rank 0's copy suffices.
    """

    def __init__(
        self,
        grad_norm_history: list[float] | None = None,
        loss_history: list[float] | None = None,
        outlier_steps: int = 0,
        skipped_steps: int = 0,
    ):
        self.history: dict[str, list[float]] = {
            "grad_norm": list(grad_norm_history or []),
            "loss": list(loss_history or []),
        }
        self.outlier_steps = int(outlier_steps)
        self.skipped_steps = int(skipped_steps)

    def on_step_end(self, args, state, control, **kwargs):
        # keep the TrainerState copy current for every save path, including those that
        # write trainer_state.json without refreshing the stateful callbacks
        state.stateful_callbacks[type(self).__name__] = self.state()

    def state(self) -> dict:
        return {
            "args": {
                "grad_norm_history": list(self.history["grad_norm"]),
                "loss_history": list(self.history["loss"]),
                "outlier_steps": self.outlier_steps,
                "skipped_steps": self.skipped_steps,
            },
            "attributes": {},
        }


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


@dataclass(frozen=True)
class _EmaSpec:
    """Shape of one parameter's running averages in the optimizer state.

    ``experts`` is ``None`` for one average per tensor (a 0-dim tensor), else the
    number of experts in this rank's tensor (a vector). ``ep_mesh`` stores an
    expert-parallel vector as a DTensor over all experts, sharded on that mesh's
    ``ep`` axis and replicated on the others, so checkpoints address every EP rank's
    experts at their own offset instead of under one shared name.
    """

    experts: int | None = None
    ep_mesh: Any = None

    def _placements(self):
        from torch.distributed.tensor import Replicate, Shard

        return [
            Shard(0) if name == "ep" else Replicate()
            for name in self.ep_mesh.mesh_dim_names
        ]

    def adopt(self, value, device) -> torch.Tensor:
        """``value`` as this spec's float32 tensor on ``device`` (NaN if unusable)."""
        sharded = self.ep_mesh is not None and self.experts is not None
        local_shape = () if self.experts is None else (self.experts,)
        local = _local(value) if torch.is_tensor(value) else None
        if local is None or tuple(local.shape) != local_shape:
            local = torch.full(local_shape, math.nan, dtype=torch.float32)
            value = None
        target = device if device is not None else local.device
        if (
            value is not None
            and local.device == target
            and local.dtype == torch.float32
            and hasattr(value, "to_local") == sharded
        ):
            return value
        local = local.detach().to(device=target, dtype=torch.float32)
        if not sharded:
            return local
        from torch.distributed.tensor import DTensor

        size = self.experts * self.ep_mesh["ep"].size()
        return DTensor.from_local(
            local,
            self.ep_mesh,
            self._placements(),
            run_check=False,
            shape=torch.Size((size,)),
            stride=(1,),
        )


def _unwrap_optimizer(optimizer):
    try:
        from accelerate.optimizer import AcceleratedOptimizer
    except ImportError:  # pragma: no cover
        return optimizer
    while isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer
    return optimizer


class _GradNormEmaStore:
    """Per-tensor running averages kept in the optimizer state, as OLMo does.

    ``optimizer.state[p][GRAD_NORM_EMA_KEY]`` is a 0-dim float32 tensor, so it is saved,
    sharded and restored with the rest of the optimizer state. torch optimizers
    initialise a parameter's state when it is empty, so a value is only written into a
    non-empty state (after the first optimizer step); until then it is kept here.
    Optimizer hooks move pending values in before every ``state_dict()``, keep the
    values float32 across ``load_state_dict()`` (which casts state to the parameter
    dtype) and, when an FSDP2 DCP checkpoint holds them, add placeholders to the load
    template so that DCP reads them.
    """

    def __init__(self):
        self.pending: dict[torch.Tensor, torch.Tensor] = {}
        self.optimizer = None
        self.handles: list = []
        self.template_placeholders = False
        self.spec_fn = lambda parameter: _EmaSpec()

    def attach(self, optimizer) -> None:
        optimizer = _unwrap_optimizer(optimizer)
        if optimizer is self.optimizer:
            return
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.optimizer = optimizer
        if optimizer is None or not hasattr(optimizer, "register_state_dict_pre_hook"):
            return
        self.handles = [
            optimizer.register_state_dict_pre_hook(self._before_state_dict),
            optimizer.register_state_dict_post_hook(self._after_state_dict),
            optimizer.register_load_state_dict_pre_hook(self._before_load),
            optimizer.register_load_state_dict_post_hook(self._after_load),
        ]

    def _state(self, parameter) -> dict | None:
        state = getattr(self.optimizer, "state", None)
        return state.get(parameter) if state is not None else None

    def lookup(self, parameters, specs, device) -> list[torch.Tensor]:
        """The running averages of each parameter (NaN where there are none yet).

        ``specs`` gives each parameter's :class:`_EmaSpec`. The stored value is a 0-dim
        tensor, a vector over the parameter's experts, or (under expert parallelism) a
        DTensor over all experts sharded on the ``ep`` axis; the returned tensors are
        the local float32 pieces on ``device``, updated in place by the caller.
        """
        values = []
        for parameter, spec in zip(parameters, specs, strict=True):
            state = self._state(parameter)
            value = state.get(GRAD_NORM_EMA_KEY) if state else None
            if value is None:
                value = self.pending.pop(parameter, None)
            value = spec.adopt(value, device)
            if state:
                state[GRAD_NORM_EMA_KEY] = value
            else:
                self.pending[parameter] = value
            values.append(_local(value))
        return values

    def flush(self) -> None:
        for parameter in list(self.pending):
            state = self._state(parameter)
            if state:
                state[GRAD_NORM_EMA_KEY] = self.pending.pop(parameter)

    def _before_state_dict(self, optimizer) -> None:
        self.flush()

    def _after_state_dict(self, optimizer, state_dict):
        if not self.template_placeholders:
            return None
        state = state_dict["state"]
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        for key, entry in list(state.items()):
            if entry and GRAD_NORM_EMA_KEY not in entry:
                parameter = (
                    parameters[key]
                    if isinstance(key, int) and key < len(parameters)
                    else None
                )
                spec = self.spec_fn(parameter) if parameter is not None else _EmaSpec()
                state[key] = {**entry, GRAD_NORM_EMA_KEY: spec.adopt(None, None)}
        return state_dict

    def _before_load(self, optimizer, state_dict):
        saved = [key for group in state_dict["param_groups"] for key in group["params"]]
        current = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        by_key = dict(zip(saved, current, strict=False))
        loaded = {}
        state = {}
        for key, entry in state_dict["state"].items():
            if isinstance(entry, dict) and GRAD_NORM_EMA_KEY in entry:
                entry = dict(entry)
                value = entry.pop(GRAD_NORM_EMA_KEY)
                if key in by_key and torch.is_tensor(value):
                    loaded[by_key[key]] = value.detach().to(torch.float32)
            state[key] = entry
        self._loaded = loaded
        return {**state_dict, "state": state}

    def _after_load(self, optimizer) -> None:
        for parameter, value in getattr(self, "_loaded", {}).items():
            state = self._state(parameter)
            if state:
                state[GRAD_NORM_EMA_KEY] = value
                self.pending.pop(parameter, None)
            else:
                self.pending[parameter] = value
        self._loaded = {}


def _dcp_checkpoint_has_ema(checkpoint: str) -> bool | None:
    """Whether an FSDP2 DCP optimizer checkpoint holds per-tensor running averages.

    ``None`` when ``checkpoint`` has no DCP optimizer directory.
    """
    if not checkpoint or not os.path.isdir(checkpoint):
        return None
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except ImportError:  # pragma: no cover
        return None
    found = None
    for name in sorted(os.listdir(checkpoint)):
        path = os.path.join(checkpoint, name)
        if not (name.startswith("optimizer") and os.path.isdir(path)):
            continue
        try:
            metadata = FileSystemReader(path).read_metadata()
        except Exception as exc:  # pylint: disable=broad-except
            LOG.debug("could not read DCP metadata from %s: %s", path, exc)
            continue
        found = any(
            key.endswith(f".{GRAD_NORM_EMA_KEY}")
            for key in metadata.state_dict_metadata
        )
        if found:
            return True
    return found


class GradNormGuardMixin:
    """Two optional guards applied after clipping and before the optimizer step.

    **Outlier steps** (``step_outlier_grad_norm_zscore`` and/or
    ``step_outlier_loss_zscore``). Each signal is scored with a robust z-score
    against the previous ``step_outlier_window`` accepted steps, ``(value -
    median) / (1.4826 * MAD)``: the pre-clip global gradient norm, and the
    step's training loss averaged over micro-batches and ranks. When any
    enabled signal is above its threshold, ``scale`` (default) multiplies the
    gradients by the smallest ``threshold / value`` among the triggered
    signals and ``skip`` sets every gradient to ``None`` (torch optimizers
    leave parameters without a gradient untouched, so neither the weights nor
    the optimizer state change). A non-finite gradient norm or loss is always
    skipped. Outlier steps do not enter the history. Both signals are global
    (the gradient norm comes from the trainer's ``_get_grad_norm``, which is
    sharding-aware under FSDP2, TP and EP), so every rank makes the same
    decision. The history is checkpointed in ``trainer_state.json``
    (:class:`StepOutlierState`) and restored on resume.

    **Per-tensor adaptive clipping** (``grad_clip_norm_ratio``), as in OLMo's
    ``max_grad_norm_ratio``. Each trainable tensor's gradient norm is clipped
    to ``ratio`` times an exponential average of that tensor's previous
    clipped norms; the average uses ``grad_clip_norm_ratio_beta`` or, if
    unset, the larger of the optimizer's betas. It acts on the gradients the
    optimizer receives, i.e. after ``max_grad_norm``; set ``max_grad_norm: 0``
    for OLMo's behaviour, where the ratio clip replaces the global clip.

    The norm is that of the whole logical tensor under FSDP2 (incl. HSDP and CPU
    offload) and DTensor tensor parallelism: local pieces are reduced over each
    gradient's sharded mesh axes, one collective per distinct layout, and each rank
    scales only its local shard.

    Fused expert tensors (the experts modules the expert-parallel plugin detects,
    their per-expert biases and expert LoRA) are clipped **per expert**: one norm,
    one coefficient and one running average per expert slice, with or without
    expert parallelism, so turning EP on (at any EP size) does not change training.
    Under EP each rank holds whole experts, so no collective crosses the ``ep`` axis.
    An EP-local tensor whose expert dimension is unknown is clipped as one fused
    tensor over all EP ranks (its norm is reduced over ``ep`` too).

    The averages live in the optimizer state (``grad_norm_exp_avg``, as in OLMo): a
    0-dim tensor per tensor, or a vector over the experts. Under EP that vector is a
    DTensor over all experts, sharded on the ``ep`` axis, so FSDP2 checkpoints keep
    every EP rank's experts (the experts' own weights and Adam moments in an EP
    checkpoint currently keep only EP group 0's experts). Clip counts are logged
    per tensor or expert slice, from rank 0.

    The logged ``grad_norm`` stays the raw pre-clip global norm.
    """

    _step_loss_sum: torch.Tensor | None
    _step_loss_micro: int
    _grad_clip_last_clipped: torch.Tensor | None
    _grad_clip_last_total: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._step_outlier_enabled():
            self._step_outlier_state()

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)  # type: ignore[misc]
        if getattr(self.args, "step_outlier_loss_zscore", None) and torch.is_tensor(  # type: ignore[attr-defined]
            loss
        ):
            previous = getattr(self, "_step_loss_sum", None)
            value = loss.detach().float().reshape(-1).mean()
            self._step_loss_sum = value if previous is None else previous + value
            self._step_loss_micro = getattr(self, "_step_loss_micro", 0) + 1
        return loss

    def _get_grad_norm(self, model, grad_norm=None):
        grad_norm = super()._get_grad_norm(model, grad_norm)  # type: ignore[misc]
        if grad_norm is None:
            return grad_norm
        skipped = self._step_outlier_guard(model, grad_norm)
        if not skipped:
            self._grad_clip_by_ratio(model)
        return grad_norm

    # checkpointing

    def _load_callback_state(self) -> None:
        super()._load_callback_state()  # type: ignore[misc]
        self._restore_step_outlier_state(
            getattr(self.state, "stateful_callbacks", None) or {}  # type: ignore[attr-defined]
        )

    def _restore_step_outlier_state(self, stateful_callbacks: dict) -> None:
        data = stateful_callbacks.get(StepOutlierState.__name__)
        if isinstance(data, list):
            data = data[-1] if data else None
        if not data or not self._step_outlier_enabled():
            return
        restored = StepOutlierState(**(data.get("args") or {}))
        handler = getattr(self, "callback_handler", None)
        if handler is not None:
            for callback in list(handler.callbacks):
                if isinstance(callback, StepOutlierState):
                    handler.remove_callback(callback)
            handler.add_callback(restored)
        self._step_outlier = restored
        self._trim_step_outlier_history()

    def _load_optimizer_and_scheduler(self, checkpoint):
        store = self._grad_clip_store()
        if store is not None:
            store.attach(getattr(self, "optimizer", None))
            model = getattr(self, "model", None)
            if model is not None:
                layouts, ep_local, ep_mesh = self._grad_clip_layout(model)
                store.spec_fn = lambda parameter: self._grad_clip_spec(
                    parameter, layouts, ep_local, ep_mesh
                )
            # FSDP2 loads into a template built from the fresh optimizer's state_dict():
            # DCP reads only the template's keys (and fails on keys the checkpoint
            # lacks), and the full-state-dict path fails on 0-dim keys missing from it
            store.template_placeholders = (
                _dcp_checkpoint_has_ema(checkpoint) is not False
            )
        try:
            return super()._load_optimizer_and_scheduler(checkpoint)  # type: ignore[misc]
        finally:
            if store is not None:
                store.template_placeholders = False

    # outlier steps

    def _step_outlier_enabled(self) -> bool:
        args = getattr(self, "args", None)
        return bool(
            getattr(args, "step_outlier_grad_norm_zscore", None)
            or getattr(args, "step_outlier_loss_zscore", None)
        )

    def _step_outlier_window(self) -> int:
        return int(getattr(self.args, "step_outlier_window", None) or 100)  # type: ignore[attr-defined]

    def _step_outlier_state(self) -> StepOutlierState:
        state = getattr(self, "_step_outlier", None)
        if state is None:
            state = StepOutlierState()
            self._step_outlier = state
            handler = getattr(self, "callback_handler", None)
            if handler is not None:
                handler.add_callback(state)
        return state

    def _trim_step_outlier_history(self) -> None:
        window = self._step_outlier_window()
        for history in self._step_outlier.history.values():
            del history[:-window]

    @property
    def _step_outlier_history(self) -> dict[str, list[float]]:
        return self._step_outlier.history

    @property
    def _step_outlier_count(self) -> int:
        return self._step_outlier.outlier_steps

    @property
    def _step_outlier_skipped(self) -> int:
        return self._step_outlier.skipped_steps

    def _consume_step_loss(self) -> float | None:
        total = getattr(self, "_step_loss_sum", None)
        count = getattr(self, "_step_loss_micro", 0)
        self._step_loss_sum = None
        self._step_loss_micro = 0
        if total is None or count == 0:
            return None
        mean = total / count
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
            mean = accelerator.reduce(mean, reduction="mean")
        return float(mean.item())

    def _step_outlier_guard(self, model, grad_norm) -> bool:
        args = self.args  # type: ignore[attr-defined]
        thresholds = {
            "grad_norm": getattr(args, "step_outlier_grad_norm_zscore", None),
            "loss": getattr(args, "step_outlier_loss_zscore", None),
        }
        if not any(thresholds.values()):
            return False
        window = self._step_outlier_window()
        state = self._step_outlier_state()

        values: dict[str, float] = {}
        if thresholds["grad_norm"]:
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            values["grad_norm"] = float(
                grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
            )
        if thresholds["loss"]:
            loss = self._consume_step_loss()
            if loss is not None:
                values["loss"] = loss
        step = self.state.global_step + 1  # type: ignore[attr-defined]

        bad = [name for name, value in values.items() if not math.isfinite(value)]
        if bad:
            self._apply_step_outlier(model, "skip", 0.0)
            LOG.warning(
                "step %s: non-finite %s; optimizer step skipped",
                step,
                ", ".join(f"{name} {values[name]}" for name in bad),
            )
            return True

        triggered: list[str] = []
        factor = 1.0
        for name, value in values.items():
            zscore = thresholds[name]
            history = state.history[name][-window:]
            if zscore is None or len(history) < window:
                continue
            center = median(history)
            spread = _MAD_TO_STD * median(abs(x - center) for x in history)
            if spread <= 0:
                continue
            threshold = center + float(zscore) * spread
            if value > threshold:
                factor = min(factor, threshold / value)
                triggered.append(
                    f"{name} {value:.4g} is {(value - center) / spread:.1f} robust "
                    f"sigma above the median {center:.4g}"
                )
        if not triggered:
            for name, value in values.items():
                state.history[name].append(value)
            self._trim_step_outlier_history()
            return False

        action = getattr(args, "step_outlier_action", None) or "scale"
        self._apply_step_outlier(model, action, factor)
        LOG.warning(
            "step %s: %s (last %s steps); %s",
            step,
            "; ".join(triggered),
            window,
            "optimizer step skipped"
            if action == "skip"
            else f"gradients scaled by {factor:.3g}",
        )
        return action == "skip"

    def _apply_step_outlier(self, model, action: str, factor: float) -> None:
        state = self._step_outlier_state()
        state.outlier_steps += 1
        if action == "skip":
            state.skipped_steps += 1
            for param in model.parameters():
                param.grad = None
            return
        grads = [
            p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            for p in model.parameters()
            if p.grad is not None
        ]
        if grads:
            torch._foreach_mul_([g.detach() for g in grads], factor)

    # per-tensor adaptive clipping

    def _grad_clip_beta(self) -> float:
        beta = getattr(self.args, "grad_clip_norm_ratio_beta", None)  # type: ignore[attr-defined]
        if beta is not None:
            return float(beta)
        optimizer = getattr(self, "optimizer", None)
        betas = None
        if optimizer is not None and optimizer.param_groups:
            betas = optimizer.param_groups[0].get("betas")
        return float(max(betas)) if betas else 0.99

    def _grad_clip_store(self) -> _GradNormEmaStore | None:
        if not getattr(getattr(self, "args", None), "grad_clip_norm_ratio", None):
            return None
        store = getattr(self, "_grad_clip_ema_store", None)
        if store is None:
            store = _GradNormEmaStore()
            self._grad_clip_ema_store = store
        return store

    def _grad_clip_ep_mesh(self, model=None):
        """The device mesh with the ``ep`` axis, or ``None`` without expert parallelism.

        Pure EP builds no global mesh; there the experts' own (per-rank) root mesh
        carries the ``ep`` axis.
        """
        enabled = getattr(self, "_expert_parallel_enabled", None)
        if not (callable(enabled) and enabled()):
            return None
        meshes = [self._global_mesh()]  # type: ignore[attr-defined]
        try:
            from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

            meshes.append(getattr(ExpertParallelPlugin, "_device_mesh", None))
        except ImportError:  # pragma: no cover
            pass
        if model is not None:
            from axolotl.utils.gradient_clipping import ep_local_parameter_ids

            ep_local = ep_local_parameter_ids(model)
            for parameter in model.parameters():
                mesh = getattr(parameter, "device_mesh", None)
                if id(parameter) in ep_local and mesh is not None:
                    meshes.append(mesh._get_root_mesh())
        return next(
            (
                mesh
                for mesh in meshes
                if mesh is not None and "ep" in (mesh.mesh_dim_names or ())
            ),
            None,
        )

    def _grad_clip_layout(self, model):
        """Expert layouts, EP-local parameter ids and the EP mesh, computed once per model."""
        cached = getattr(self, "_grad_clip_model_layout", None)
        if cached is None or cached[0] is not model:
            from axolotl.utils.gradient_clipping import (
                ep_local_parameter_ids,
                expert_parameter_layouts,
            )

            ep_mesh = self._grad_clip_ep_mesh(model)
            cached = (
                model,
                expert_parameter_layouts(model),
                frozenset(ep_local_parameter_ids(model))
                if ep_mesh is not None
                else frozenset(),
                ep_mesh,
            )
            self._grad_clip_model_layout = cached
        return cached[1:]

    def _grad_clip_spec(self, parameter, layouts, ep_local, ep_mesh) -> _EmaSpec:
        layout = layouts.get(id(parameter))
        if layout is None:
            return _EmaSpec()
        return _EmaSpec(
            layout.num_experts,
            ep_mesh if id(parameter) in ep_local else None,
        )

    def _grad_clip_by_ratio(self, model) -> None:
        store = self._grad_clip_store()
        if store is None:
            return
        from axolotl.utils.gradient_clipping import (
            get_grad_norms_per_tensor_,
            scale_grads_per_tensor_,
        )

        ratio = float(self.args.grad_clip_norm_ratio)  # type: ignore[attr-defined]
        layouts, ep_local, ep_mesh = self._grad_clip_layout(model)
        result = get_grad_norms_per_tensor_(
            model.parameters(),
            ep_local_parameters=ep_local,
            global_mesh=ep_mesh,
            expert_layouts=layouts,
        )
        if not result.parameters:
            return
        norms = result.norms
        store.attach(getattr(self, "optimizer", None))
        averages = store.lookup(
            result.parameters,
            [
                self._grad_clip_spec(p, layouts, ep_local, ep_mesh)
                for p in result.parameters
            ],
            norms.device,
        )
        ema = torch.cat([average.reshape(-1) for average in averages])
        # a tensor's (or expert's) first step, or a non-finite average, starts from its norm
        ema = torch.where(torch.isfinite(ema), ema, norms)
        coef = torch.clamp(ratio * ema / (norms + 1e-6), max=1.0)
        scale_grads_per_tensor_(result, coef)
        ema = ema.lerp(norms * coef, 1.0 - self._grad_clip_beta())
        torch._foreach_copy_(
            averages,
            [
                piece.view_as(average)
                for piece, average in zip(
                    ema.split(result.sizes), averages, strict=True
                )
            ],
        )
        self._grad_clip_last_clipped = (coef < 1.0).sum()
        self._grad_clip_last_total = int(coef.numel())

    def _grad_clip_averages(self, params) -> torch.Tensor:
        """This rank's running averages of ``params``, flattened (for inspection and tests)."""
        store = self._grad_clip_store()
        assert store is not None
        values = []
        for param in params:
            state = store._state(param)
            value = state.get(GRAD_NORM_EMA_KEY) if state else None
            if value is None:
                value = store.pending.get(param)
            values.append(
                torch.tensor([math.nan])
                if value is None
                else _local(value).detach().cpu().reshape(-1)
            )
        return torch.cat(values)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            state = getattr(self, "_step_outlier", None)
            if state is not None:
                logs["step_outlier/steps"] = state.outlier_steps
                logs["step_outlier/skipped"] = state.skipped_steps
            clipped = getattr(self, "_grad_clip_last_clipped", None)
            if clipped is not None:
                count = int(clipped.item())
                logs["grad_clip_ratio/clipped_tensors"] = count
                logs["grad_clip_ratio/clipping_rate"] = count / max(
                    1, self._grad_clip_last_total
                )
        super().log(logs, start_time)  # type: ignore[misc]
