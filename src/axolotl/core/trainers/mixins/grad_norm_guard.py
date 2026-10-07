"""Gradient-norm guards: whole-step outlier handling and per-tensor adaptive clipping."""

from __future__ import annotations

import math
from collections import deque
from statistics import median

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Scale from the median absolute deviation to a normal standard deviation.
_MAD_TO_STD = 1.4826


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
    skipped. Outlier steps do not enter the history. Both signals are global,
    so every rank makes the same decision.

    **Per-tensor adaptive clipping** (``grad_clip_norm_ratio``), as in OLMo's
    ``max_grad_norm_ratio``. Each trainable tensor's gradient norm is clipped
    to ``ratio`` times an exponential average of that tensor's previous
    clipped norms; the average uses ``grad_clip_norm_ratio_beta`` or, if
    unset, the larger of the optimizer's betas. It acts on the gradients the
    optimizer receives, i.e. after ``max_grad_norm``; set ``max_grad_norm: 0``
    for OLMo's behaviour, where the ratio clip replaces the global clip. The
    averages live on the trainer and restart on resume.

    The logged ``grad_norm`` stays the raw pre-clip global norm.
    """

    _step_outlier_history: dict[str, deque[float]]
    _step_outlier_count: int
    _step_outlier_skipped: int
    _step_loss_sum: torch.Tensor | None
    _step_loss_micro: int
    _grad_clip_ema: torch.Tensor | None
    _grad_clip_params: tuple[int, ...] | None
    _grad_clip_last_clipped: torch.Tensor | None
    _grad_clip_last_total: int

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

    # outlier steps

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
        window = int(getattr(args, "step_outlier_window", None) or 100)
        action = getattr(args, "step_outlier_action", None) or "scale"
        if not hasattr(self, "_step_outlier_history"):
            self._step_outlier_history = {
                name: deque(maxlen=window) for name in thresholds
            }
            self._step_outlier_count = 0
            self._step_outlier_skipped = 0

        values: dict[str, float] = {}
        if thresholds["grad_norm"]:
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
            history = self._step_outlier_history[name]
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
                self._step_outlier_history[name].append(value)
            return False

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
        self._step_outlier_count += 1
        if action == "skip":
            self._step_outlier_skipped += 1
            for param in model.parameters():
                param.grad = None
            return
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            torch._foreach_mul_(grads, factor)

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

    def _grad_clip_by_ratio(self, model) -> None:
        ratio = getattr(self.args, "grad_clip_norm_ratio", None)  # type: ignore[attr-defined]
        if not ratio:
            return
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]
        device = grads[0].device
        norms = torch.stack(
            [
                n.to(device=device, dtype=torch.float32)
                for n in torch._foreach_norm(grads)
            ]
        )
        key = tuple(id(p) for p in params)
        ema = getattr(self, "_grad_clip_ema", None)
        if ema is None or getattr(self, "_grad_clip_params", None) != key:
            ema = norms.clone()
            self._grad_clip_params = key
        coef = torch.clamp(float(ratio) * ema / (norms + 1e-6), max=1.0)
        torch._foreach_mul_(
            grads, [c.to(g.device) for c, g in zip(coef.unbind(), grads, strict=True)]
        )
        ema.lerp_(norms * coef, 1.0 - self._grad_clip_beta())
        self._grad_clip_ema = ema
        self._grad_clip_last_clipped = (coef < 1.0).sum()
        self._grad_clip_last_total = len(params)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            if hasattr(self, "_step_outlier_count"):
                logs["step_outlier/steps"] = self._step_outlier_count
                logs["step_outlier/skipped"] = self._step_outlier_skipped
            clipped = getattr(self, "_grad_clip_last_clipped", None)
            if clipped is not None:
                count = int(clipped.item())
                logs["grad_clip_ratio/clipped_tensors"] = count
                logs["grad_clip_ratio/clipping_rate"] = count / max(
                    1, self._grad_clip_last_total
                )
        super().log(logs, start_time)  # type: ignore[misc]
