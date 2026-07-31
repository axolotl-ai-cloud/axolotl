"""Ternary (1.58-bit) conversion plugin: swap FP linears to {-1, 0, +1}, heal, export."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

from transformers import PreTrainedModel, Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .args import (
    CONFLICTING_KEYS,
    TernaryArgs,
    TernaryConfig,
    resolve_ternary_config,
)

if TYPE_CHECKING:
    from .swap import SwapManifest

LOG = get_logger(__name__)

__all__ = [
    "TernaryArgs",
    "TernaryConfig",
    "TernaryPlugin",
    "is_export_rank",
    "resolve_ternary_config",
]


def is_export_rank() -> bool:
    """Whether this process should write the exports.

    `post_train_unload` runs after `train.py` has already torn the process group
    down, so `is_main_process()` — which still sees a populated `PartialState` and
    falls through to `dist.get_rank()` — would raise instead of answering.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    for key in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value:
            return value == "0"
    return True


class TernaryPlugin(BasePlugin):
    """Converts a full-precision checkpoint to ternary weights and heals it in-place.

    The plugin swaps the enumerated transformer-block Linears for `TernaryLinear`
    right after the model is built, installs the λ-warmup / weight-decay-anneal /
    monitoring callbacks, optionally routes training through an in-process
    distillation trainer, and runs the export pipeline once training is done.
    """

    def __init__(self):
        super().__init__()
        self.manifest: "SwapManifest | None" = None

    def get_input_args(self) -> str:
        """Returns the pydantic model for the `ternary` config block."""
        return "axolotl.integrations.ternary.args.TernaryArgs"

    def get_training_args_mixin(self) -> str:
        """Returns the dataclass carrying the distillation knobs onto `TrainingArguments`."""
        return "axolotl.integrations.ternary.distill.TernaryDistillTrainingArgsMixin"

    def register(self, cfg: dict) -> None:
        """Cross-checks the whole config, which the plugin schema only partly sees.

        Rejects adapters and any co-set quantization path (`use_onebitllms`, `qat`,
        `quantization`, `load_in_*bit`), requires torch >= 2.6 (the QAT floor),
        keeps the teacher losses off the RL path, and recommends `save_only_model:
        true` so optimizer state is not written beside the baked master.
        """
        if cfg.get("adapter"):
            raise ValueError(
                "ternary conversion is full-finetune only; remove `adapter:`"
            )
        for key in CONFLICTING_KEYS:
            if cfg.get(key):
                raise ValueError(
                    f"`{key}:` cannot be combined with the ternary plugin; the ternary "
                    "quantizer owns the weight representation for the whole run"
                )

        import torch
        from packaging import version

        torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
        if version.parse(torch_version) < version.parse("2.6.0"):
            raise ValueError(
                f"ternary conversion requires torch >= 2.6.0, found {torch_version}"
            )

        distill_mode = resolve_ternary_config(cfg).distill.mode
        if distill_mode and cfg.get("rl"):
            raise ValueError(
                f"ternary.distill.mode: {distill_mode} heals a causal-LM next-token "
                f"objective and cannot run under `rl: {cfg.get('rl')}`; drop "
                "ternary.distill (pure QAT works on the RL path) or drop `rl:`"
            )

        if cfg.get("torch_compile"):
            LOG.warning(
                "ternary: `torch_compile` is untested against the fake-quant forward; "
                "set ternary.fused_fake_quant: false if compilation fails"
            )
        if not cfg.get("save_only_model"):
            LOG.warning(
                "ternary: `save_only_model: true` is recommended so checkpoints hold "
                "only the baked master weights"
            )

    def post_model_build(self, cfg: DictDefault, model: PreTrainedModel) -> None:
        """Swaps the enumerated Linears for `TernaryLinear` before any parallel wrapping.

        Swapped modules start at λ=1 (fully quantized), the steady state every
        eval/inference entry point must see. `LambdaScheduleCallback` walks λ back
        down to the schedule's step-0 value when a training loop actually starts.
        """
        from .modules import set_act_quant_sharing
        from .swap import convert_model

        ternary_cfg = resolve_ternary_config(cfg)
        if ternary_cfg.smoothing:
            raise NotImplementedError(
                "ternary.smoothing is accepted by the schema but not implemented yet"
            )

        self.manifest = convert_model(model, cfg)
        if ternary_cfg.share_act_quant:
            set_act_quant_sharing(model)

    def add_callbacks_pre_trainer(
        self, cfg: DictDefault, model: PreTrainedModel
    ) -> list[Callable]:
        """Builds the λ-schedule, weight-decay-anneal and zero-gradient callbacks."""
        from .callbacks import (
            LambdaScheduleCallback,
            WdAnnealCallback,
            ZeroGradWarningCallback,
        )

        ternary_cfg = resolve_ternary_config(cfg)
        callbacks: list[Callable] = [ZeroGradWarningCallback(model)]
        if ternary_cfg.lambda_schedule != "none":
            callbacks.append(
                LambdaScheduleCallback(
                    model,
                    schedule=ternary_cfg.lambda_schedule,
                    warmup_steps=ternary_cfg.lambda_warmup_steps,
                )
            )
        if ternary_cfg.weight_decay_zero_at < 1.0 and cfg.get("weight_decay"):
            callbacks.append(
                WdAnnealCallback(zero_at=ternary_cfg.weight_decay_zero_at, model=model)
            )
        return callbacks

    def add_callbacks_post_trainer(
        self, cfg: DictDefault, trainer: Trainer
    ) -> list[Callable]:
        """Builds the monitoring callback, which logs through the trainer's metric store."""
        from .callbacks import TernaryMonitorCallback

        ternary_cfg = resolve_ternary_config(cfg)
        if not ternary_cfg.log_code_flip_every:
            return []
        return [
            TernaryMonitorCallback(
                trainer.model,
                every_n_steps=ternary_cfg.log_code_flip_every,
                trainer=trainer,
            )
        ]

    def get_trainer_cls(self, cfg: DictDefault) -> type[Trainer] | None:
        """Returns the in-process distillation trainer when that mode is configured."""
        if resolve_ternary_config(cfg).distill.mode != "inprocess":
            return None
        if cfg.get("rl"):
            # rl.py asks the plugin manager first; an AxolotlTrainer subclass would be
            # built for a preference-tuning run
            return None

        from .distill import TernaryDistillTrainer

        return TernaryDistillTrainer

    def get_training_args(self, cfg: DictDefault) -> dict:
        """Surfaces the in-process distillation knobs on `TrainingArguments`."""
        distill_cfg = resolve_ternary_config(cfg).distill
        if distill_cfg.mode != "inprocess":
            return {}
        return {
            "ternary_distill_teacher_model": distill_cfg.teacher_model
            or cfg.get("base_model"),
            "ternary_distill_logits_weight": distill_cfg.logits_weight,
            "ternary_distill_logits_temperature": distill_cfg.logits_temperature,
            "ternary_distill_hidden_weight": distill_cfg.hidden_weight,
            "ternary_distill_hidden_loss": distill_cfg.hidden_loss,
            "ternary_distill_hidden_huber_delta": distill_cfg.hidden_huber_delta,
            "ternary_distill_attn_relation_layer": distill_cfg.attn_relation_layer,
            "ternary_distill_teacher_device_map": distill_cfg.teacher_device_map,
            "ternary_distill_prefetch_teacher": distill_cfg.prefetch_teacher,
            "ternary_distill_logprob_prefetch": distill_cfg.logprob_prefetch,
            "ternary_distill_logprob_prefetch_depth": distill_cfg.logprob_prefetch_depth,
            "ternary_distill_teacher_prefetch_depth": distill_cfg.teacher_prefetch_depth,
            "ternary_distill_teacher_endpoint": distill_cfg.teacher_endpoint,
            "ternary_distill_schedule": distill_cfg.schedule,
            "ternary_distill_anchor_start": distill_cfg.anchor_start,
        }

    def post_train_unload(self, cfg: DictDefault) -> None:
        """Packs the baked master into the configured deployment formats."""
        export_cfg = resolve_ternary_config(cfg).export
        # packing is single-process CPU work against the saved master
        if not export_cfg.formats or not is_export_rank():
            return

        from .export import run_export

        artifacts = run_export(cfg, formats=export_cfg.formats, manifest=self.manifest)
        for fmt, path in (artifacts or {}).items():
            LOG.info(f"ternary: wrote {fmt} export to {path}")
