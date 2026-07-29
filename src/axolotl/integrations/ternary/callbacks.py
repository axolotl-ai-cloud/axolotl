"""Training callbacks: λ warmup, KD anchor, weight-decay anneal, code monitoring."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.logging import get_logger

from . import aux_modules
from .args import LambdaSchedule
from .modules import (
    TernaryEmbedding,
    TernaryLinear,
    as_local,
    iter_quantized_modules,
    iter_ternary_modules,
)
from .quant import flip_count, zero_fraction
from .swap import module_family

LOG = get_logger(__name__)

# steepness of the normalized sigmoid ramp; ~0 and ~1 at the window edges
SIGMOID_STEEPNESS: float = 12.0

# width of the KD anchor ramp, as a fraction of max_steps
ANCHOR_RAMP_FRACTION: float = 0.02

# optimizer steps a targeted module may go without a gradient before it is called out
ZERO_GRAD_MONITORED_STEPS: int = 50

# module names spelled out in the zero-grad warning before it truncates
_MAX_REPORTED_MODULES: int = 5


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class LambdaScheduleCallback(TrainerCallback):
    """Ramps the quantization strength λ from 0 to 1 across the warmup window.

    A linear ramp is the only schedule with published evidence; step jumps to λ=1
    destroy the model. λ is pushed onto every `TernaryLinear` at each step start.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: LambdaSchedule = "linear",
        warmup_steps: int | float = 1000,
    ) -> None:
        """Store the model and schedule.

        Args:
            model: The already-swapped model.
            schedule: `linear`, `sigmoid` or `none`.
            warmup_steps: An `int` number of steps, or a `float` fraction of
                `max_steps` — the type decides, so `1` is one step and `1.0` is the
                whole run.
        """
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.ternary_modules: list[TernaryLinear | TernaryEmbedding] = [
            module for _, module in iter_quantized_modules(model)
        ]
        self.current_lambda: float = -1.0

    @staticmethod
    def lambda_at(step: int, warmup_steps: int, schedule: LambdaSchedule) -> float:
        """Return λ in `[0, 1]` for a global step under `schedule`."""
        if schedule == "none" or warmup_steps <= 0:
            return 1.0
        progress = min(max(step / warmup_steps, 0.0), 1.0)
        if schedule == "sigmoid":
            low = _sigmoid(-SIGMOID_STEEPNESS / 2)
            high = _sigmoid(SIGMOID_STEEPNESS / 2)
            centered = _sigmoid(SIGMOID_STEEPNESS * (progress - 0.5))
            return (centered - low) / (high - low)
        return progress

    def _resolve_warmup(self, state: TrainerState) -> int:
        steps = self.warmup_steps
        # the type disambiguates, not the value: `1.0` is the whole run, `1` is one step
        if isinstance(steps, float) and not isinstance(steps, bool):
            return max(1, int(steps * state.max_steps))
        return int(steps)

    def _apply(self, state: TrainerState) -> None:
        if not self.ternary_modules:
            return
        value = self.lambda_at(
            state.global_step, self._resolve_warmup(state), self.schedule
        )
        if value == self.current_lambda:
            return
        for module in self.ternary_modules:
            module.set_lambda(value)
        if value >= 1.0 > self.current_lambda:
            LOG.info("ternary: lambda reached 1.0 at step %d", state.global_step)
        self.current_lambda = value

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Resolve a fractional warmup against `state.max_steps` and set the initial λ."""
        self._apply(state)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Set λ on every ternary module for this step."""
        self._apply(state)


class DistillAnchorCallback(TrainerCallback):
    """Holds the distillation terms at 0 until the LR-decay tail, then ramps them in.

    `distill.schedule: anchored` spends the bulk of the heal on pure CE and anchors the
    student to the teacher only over the last `1 - anchor_start` of the steps, where the
    LR has already decayed. The switch is a short linear ramp rather than a step: an
    abrupt change in loss composition shocks the optimizer's moment estimates, the same
    failure mode that made discrete λ jumps unusable in the BitNet warmup ablations.
    A multiplier of exactly 0 also keeps the teacher out of memory: the trainer skips
    the teacher forward — and its load — on those steps.
    """

    def __init__(
        self,
        trainer: Any = None,
        anchor_start: float = 0.9,
        ramp_fraction: float = ANCHOR_RAMP_FRACTION,
    ) -> None:
        """Store the trainer the multiplier is pushed onto and the ramp geometry.

        Args:
            trainer: The trainer exposing `set_distill_multiplier`.
            anchor_start: Fraction of `max_steps` after which the KD terms turn on.
            ramp_fraction: Width of the linear ramp, as a fraction of `max_steps`.
        """
        self.trainer = trainer
        self.anchor_start = anchor_start
        self.ramp_fraction = ramp_fraction
        self.multiplier: float = -1.0
        self.warned = False

    @staticmethod
    def multiplier_at(
        step: int,
        max_steps: int,
        anchor_start: float,
        ramp_fraction: float = ANCHOR_RAMP_FRACTION,
    ) -> float:
        """Return the KD multiplier in `[0, 1]` for a global step.

        An unknown horizon (`max_steps <= 0`) has no anchor to resolve against and
        falls back to the constant blend.
        """
        if max_steps <= 0:
            return 1.0
        ramp = max(1, round(ramp_fraction * max_steps))
        # the anchor is pulled in far enough for the ramp to finish before the last
        # step, so a short run still gets its KD phase at full weight; steps are
        # 0-based, so the anchor is the first KD step rather than the last CE one
        anchor = min(int(anchor_start * max_steps), max(max_steps - ramp, 0))
        if step < anchor:
            return 0.0
        return min((step - anchor + 1) / ramp, 1.0)

    def _apply(self, state: TrainerState) -> None:
        if state.max_steps <= 0 and not self.warned:
            LOG.warning(
                "ternary: distill.schedule: anchored needs a known max_steps; the KD "
                "terms stay on for the whole run"
            )
            self.warned = True
        value = self.multiplier_at(
            state.global_step, state.max_steps, self.anchor_start, self.ramp_fraction
        )
        if value == self.multiplier:
            return
        setter = getattr(self.trainer, "set_distill_multiplier", None)
        if setter is not None:
            setter(value)
        if self.multiplier <= 0.0 < value:
            LOG.info("ternary: KD anchor ramp opened at step %d", state.global_step)
        if value >= 1.0 > self.multiplier:
            LOG.info("ternary: KD anchor at full weight at step %d", state.global_step)
        self.multiplier = value

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Set the initial multiplier, resolving the anchor against `state.max_steps`."""
        self._apply(state)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Push this step's multiplier onto the trainer."""
        self._apply(state)


class WdAnnealCallback(TrainerCallback):
    """Anneals weight decay to 0 partway through training.

    Weight decay keeps churning ternary code assignments; dropping it mid-run lets
    the codes settle.
    """

    def __init__(self, zero_at: float = 0.5, model: nn.Module | None = None) -> None:
        """Store the anneal point and the model whose latents scope the anneal.

        Args:
            zero_at: Fraction of training after which weight decay is set to 0.
            model: The swapped model; defaults to the one the trainer passes to the
                callback, and only the param groups holding ternary latents are
                annealed. Without any model every group is annealed.
        """
        self.zero_at = zero_at
        self.model = model
        self.annealed = False
        self.latent_ids: set[int] | None = None

    def _resolve_latent_ids(self, model: nn.Module | None) -> set[int] | None:
        """Return the ids of the ternary latent params, or `None` if no model is known."""
        if self.latent_ids is None:
            model = self.model if self.model is not None else model
            if model is None:
                return None
            self.latent_ids = {
                id(param)
                for _, module in iter_ternary_modules(model)
                for param in module.parameters()
            }
        return self.latent_ids

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        optimizer=None,
        model: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Zero the `weight_decay` of the ternary optimizer param groups past the threshold."""
        if self.annealed or optimizer is None or state.max_steps <= 0:
            return
        if state.global_step < int(self.zero_at * state.max_steps):
            return
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return

        latent_ids = self._resolve_latent_ids(model)
        if latent_ids is not None and not latent_ids:
            self.annealed = True
            return

        groups = param_groups
        if latent_ids:
            groups = [
                group
                for group in param_groups
                if any(id(param) in latent_ids for param in group.get("params", ()))
            ]
            if not groups:
                # sharding can replace the latents with flat params the ids miss
                LOG.warning(
                    "ternary: no optimizer param group holds a ternary latent; "
                    "annealing weight decay for every group"
                )
                groups = param_groups

        self.annealed = True
        zeroed = 0
        for group in groups:
            if group.get("weight_decay"):
                group["weight_decay"] = 0.0
                zeroed += 1
        LOG.info(
            "ternary: weight decay annealed to 0 for %d param group(s) at step %d",
            zeroed,
            state.global_step,
        )


class TernaryMonitorCallback(TrainerCallback):
    """Logs ternary code-flip rate and zero fraction per module family.

    Flip rate is computed against a packed 2-bit snapshot taken at the previous
    monitoring step, so the memory cost is `numel / 4` bytes per swapped module.
    Under FSDP2 every module contributes its local shard, making the numbers
    per-rank rather than global.

    The metrics are pushed through the trainer's own metric store when one is
    available: `on_log` fires *after* the reporting callbacks have consumed `logs`
    and after `Trainer.log` has snapshotted it into `state.log_history`, so mutating
    the dict there would keep them on the console only.
    """

    def __init__(
        self, model: nn.Module, every_n_steps: int = 100, trainer=None
    ) -> None:
        """Args: model: the swapped model; every_n_steps: logging period, 0 disables;
        trainer: the trainer whose `store_metrics` carries the metrics into the logs.
        """
        self.every_n_steps = every_n_steps
        self.trainer = trainer
        self.ternary_modules: list[tuple[str, TernaryLinear]] = list(
            iter_ternary_modules(model)
        )
        self.metrics: dict[str, float] = {}
        self.snapshots: dict[str, torch.Tensor] = {}

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Compute and log `ternary/flip_rate` and `ternary/zero_frac` per family."""
        if not self.every_n_steps or not self.ternary_modules:
            return
        if state.global_step % self.every_n_steps:
            return

        codes: dict[str, float] = defaultdict(float)
        zeros: dict[str, float] = defaultdict(float)
        compared: dict[str, float] = defaultdict(float)
        flips: dict[str, float] = defaultdict(float)
        for name, module in self.ternary_modules:
            # snapshots live on CPU so monitoring never grows the training footprint
            packed = module.code_snapshot().cpu()
            numel = _code_count(module)
            family = module_family(name)
            codes[family] += numel
            zeros[family] += float(zero_fraction(packed, numel)) * numel
            previous = self.snapshots.get(name)
            if previous is not None:
                compared[family] += numel
                flips[family] += float(flip_count(previous, packed))
            self.snapshots[name] = packed

        metrics: dict[str, float] = {}
        for family, total in codes.items():
            metrics[f"ternary/zero_frac/{family}"] = zeros[family] / total
            if compared[family]:
                metrics[f"ternary/flip_rate/{family}"] = (
                    flips[family] / compared[family]
                )
        metrics["ternary/zero_frac"] = sum(zeros.values()) / sum(codes.values())
        total_compared = sum(compared.values())
        if total_compared:
            metrics["ternary/flip_rate"] = sum(flips.values()) / total_compared
        self.metrics = {key: round(value, 6) for key, value in metrics.items()}

        LOG.info(
            "ternary: step %d zero_frac %.4f flip_rate %.6f",
            state.global_step,
            self.metrics["ternary/zero_frac"],
            self.metrics.get("ternary/flip_rate", float("nan")),
        )
        self._store(self.metrics)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        """Merge the metrics into `logs` when no trainer metric store took them."""
        if logs is None or not self.metrics:
            return
        logs.update(self.metrics)
        self.metrics = {}

    def _store(self, metrics: dict[str, float]) -> bool:
        """Hand the metrics to the trainer's metric store; report whether it took them."""
        store = getattr(self.trainer, "store_metrics", None)
        if store is None:
            return False
        store(dict(metrics), train_eval="train")
        self.metrics = {}
        return True


class ZeroGradWarningCallback(TrainerCallback):
    """Warns once about targeted modules that never receive a gradient.

    Quantizing a module the run's loss does not reach is the worst trade the plugin
    can make: the damage is taken in full and no gradient exists to heal any of it.
    It is also invisible — the loss curve of a heal that skipped an MTP head looks
    exactly like one that healed it, because nothing in the objective ever touched it.

    The usual causes are structural rather than accidental: an MTP head under a run
    with no MTP loss, a draft head, a reward head on a plain SFT objective, a vision
    tower that no image in the corpus activates, or a cold MoE expert. The first four
    are the `aux_modules` families, so the warning can name the likely cause instead
    of only the module. A cold expert is different — it is a data property, not a
    wiring mistake — so the two are reported apart.

    The evidence is `weight.grad`, which not every engine keeps there: DeepSpeed ZeRO
    reduces into a flat partition and clears the parameter's gradient before
    `optimizer.step()`, and FSDP1 without `use_orig_params` leaves the module holding
    a view of a FlatParameter that has none. Both make every watched module look
    unreached, so both are detected and the watch stands down rather than accusing the
    whole model. FSDP2's DTensor latents keep a real per-rank gradient and are read
    through `as_local`.
    """

    def __init__(
        self,
        model: nn.Module,
        monitored_steps: int = ZERO_GRAD_MONITORED_STEPS,
    ) -> None:
        """Store the swapped modules to watch and how long to watch them.

        Args:
            model: The already-swapped model.
            monitored_steps: Optimizer steps from the start of training over which a
                module must stay gradient-free to be reported. Monitoring stops after
                the window, so the check costs nothing for the rest of the run.
        """
        self.monitored_steps = monitored_steps
        self.ternary_modules: list[tuple[str, TernaryLinear]] = list(
            iter_ternary_modules(model)
        )
        self.seen_grad: set[str] = set()
        self.warned = False
        self.monitoring = True
        # a trainer that never fires `on_pre_optimizer_step` would otherwise look
        # exactly like a run whose loss reaches nothing
        self.observed_steps = 0

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Record which targeted latents carry a gradient this step.

        This is the only hook where the answer exists: gradients are allocated by
        backward and gone again after `optimizer.zero_grad()`. Under FSDP2 the
        latent is a DTensor and only this rank's shard is inspected.
        """
        if not self.monitoring:
            return
        if self._gradients_are_partitioned(args):
            self.monitoring = False
            LOG.debug(
                "ternary: this run keeps its gradients off the parameters, so the "
                "zero-gradient watch has nothing to read and stands down"
            )
            return
        self.observed_steps += 1
        for name, module in self.ternary_modules:
            if name in self.seen_grad:
                continue
            grad = module.weight.grad
            # sharded runs allocate a zero-filled gradient for an unused parameter,
            # so presence alone answers a different question than the one asked
            if grad is not None and bool(as_local(grad).any()):
                self.seen_grad.add(name)
        if len(self.seen_grad) == len(self.ternary_modules):
            self.monitoring = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Emit the warning once the monitoring window closes, then stop watching."""
        if not self.monitoring or state.global_step < self.monitored_steps:
            return
        self._report(f"first {self.monitored_steps} optimizer steps")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Report on a run that ended before the monitoring window ever closed.

        A short heal — an ablation, a smoke run, `max_steps` under the window — is
        where a mis-wired target list is cheapest to catch, and it is exactly the run
        `on_step_end` never reports on.
        """
        if not self.monitoring:
            return
        self._report(f"run's {self.observed_steps} optimizer step(s)")

    def _report(self, window: str) -> None:
        """Warn about the modules still gradient-free, then stop watching."""
        self.monitoring = False
        if not self.observed_steps:
            return
        missing = [
            name for name, _ in self.ternary_modules if name not in self.seen_grad
        ]
        if not missing:
            return
        if len(missing) == len(self.ternary_modules) and len(missing) > 1:
            # every watched module at once is the signature of gradients living
            # somewhere this hook cannot see, not of an objective reaching nothing
            LOG.debug(
                "ternary: no targeted module exposed a gradient at all; reading the "
                "watch as wrong plumbing rather than as an unreachable model"
            )
            return
        LOG.warning(self._message(missing, window))
        self.warned = True

    def _gradients_are_partitioned(self, args: TrainingArguments) -> bool:
        """Whether this run keeps its gradients somewhere `weight.grad` is not.

        DeepSpeed ZeRO-2 (with `contiguous_gradients`) and ZeRO-3 reduce into a flat
        partition and clear the parameter's gradient before the optimizer steps, which
        is where this hook fires; FSDP1 without `use_orig_params` replaces the module's
        weight with a plain view of a FlatParameter. Either way the presence check
        answers a question about the engine rather than about the objective.
        """
        if getattr(args, "deepspeed", None) or getattr(args, "deepspeed_plugin", None):
            return True
        return any(
            hasattr(module.weight, "ds_id")
            or not isinstance(module.weight, nn.Parameter)
            for _, module in self.ternary_modules
        )

    def _message(self, names: list[str], window: str | None = None) -> str:
        """Return the one-shot warning: the modules, and the cause each one suggests.

        Args:
            names: Targeted module names that saw no gradient in the window.
            window: How to describe the window that saw none, as it is not always the
                configured one — a run can end before it closes.

        Returns:
            The log line, with the `aux_modules` families spelled out by name and
            rationale and everything else grouped as unreached modules.
        """
        window = window or f"first {self.monitored_steps} optimizer steps"
        lines = [
            f"ternary: {len(names)} targeted module(s) received no gradient in the "
            f"{window}. They take the "
            "quantization damage in full and this run's objective has no gradient to "
            "heal any of it — a loss curve cannot show you this."
        ]
        explanation = aux_modules.explain(names)
        if explanation:
            lines.append(explanation)
        claimed = {
            name
            for matched in aux_modules.group_by_family(names).values()
            for name in matched
        }
        rest = [name for name in names if name not in claimed]
        if rest:
            head = ", ".join(rest[:_MAX_REPORTED_MODULES])
            extra = len(rest) - _MAX_REPORTED_MODULES
            lines.append(
                f"{len(rest)} belong to no auxiliary-module family: "
                + (f"{head} (+{extra} more)" if extra > 0 else head)
            )
            lines.append(
                "  a cold MoE expert no batch has routed to yet is the benign case — "
                "that is a property of the data, not of the wiring, and it heals as "
                "soon as it is routed. A module whose labels are entirely masked, or "
                "one left out of the optimizer's parameter groups, is not"
            )
        return "\n".join(lines)


def _code_count(module: nn.Module) -> int:
    counter = getattr(module, "code_count", None)
    return counter() if callable(counter) else module.weight.numel()
