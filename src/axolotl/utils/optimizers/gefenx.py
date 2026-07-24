"""Factory for the whole-model Gefen-X Muon hybrid optimizer."""

from __future__ import annotations

from typing import Any

from axolotl.integrations.base import BaseOptimizerFactory


def coerce_optim_arg(value: Any) -> Any:
    # String-form optim_args (key=value) arrive as strings; restore native types.
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


class GefenXMuonHybridOptimizerFactory(BaseOptimizerFactory):
    # A factory, not optimizer_cls: GefenMuonHybrid needs the model to split params.
    def __call__(self, opt_model, training_args=None, **optimizer_kwargs):
        from gefen import GefenMuonHybrid

        optimizer_kwargs = {k: coerce_optim_arg(v) for k, v in optimizer_kwargs.items()}

        lr = optimizer_kwargs.pop("lr")
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)

        # Gefen-X recommended recipe; all overridable via optim_args.
        optimizer_kwargs.setdefault("backup_1d_period_one", True)
        optimizer_kwargs.setdefault("adjust_lr_fn", "match_rms_adamw")
        optimizer_kwargs.setdefault("fused", True)
        if "backup_lr" not in optimizer_kwargs:
            optimizer_kwargs["backup_lr"] = 0.5 * lr

        return GefenMuonHybrid(
            opt_model,
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        )
