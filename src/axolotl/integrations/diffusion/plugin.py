"""Deprecated diffusion LM plugin compatibility shim."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from peft import PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.diffusion import DiffusionConfig, DiffusionLMConfig

LOG = get_logger(__name__)
_LEGACY_WARNING_EMITTED = False


class DiffusionPlugin(BasePlugin):
    """
    Plugin for diffusion language model training.

    This plugin enables diffusion-based training using the LLaDA approach, which uses
    random masking and bidirectional attention to train language models.
    """

    def __init__(self):
        super().__init__()
        self.cfg = None

    def register(self, cfg: dict):
        """Translate the legacy block into the canonical core configuration."""
        global _LEGACY_WARNING_EMITTED  # pylint: disable=global-statement
        legacy = cfg.get("diffusion")
        canonical = cfg.get("diffusion_lm")

        normalization_messages: list[str] = []
        if legacy is not None or canonical is None:
            if (
                cfg.get("save_strategy") == "best"
                and cfg.get("metric_for_best_model") is None
            ):
                cfg["metric_for_best_model"] = "eval_loss"
                normalization_messages.append(
                    "save_strategy: best now uses metric_for_best_model: eval_loss"
                )

            warmup_steps = cfg.get("warmup_steps")
            if isinstance(warmup_steps, float) and 0 < warmup_steps < 1:
                warmup_ratio = cfg.get("warmup_ratio")
                if warmup_ratio is not None and warmup_ratio != warmup_steps:
                    raise ValueError(
                        "Legacy diffusion fractional warmup_steps conflicts with "
                        "an explicit warmup_ratio. Use only warmup_ratio."
                    )
                cfg["warmup_ratio"] = warmup_steps
                cfg["warmup_steps"] = None
                normalization_messages.append(
                    f"fractional warmup_steps: {warmup_steps} now uses warmup_ratio"
                )

        for message in normalization_messages:
            LOG.warning("Legacy diffusion configuration normalization: %s", message)

        def as_dict(value: Any) -> dict[str, Any]:
            if value is None:
                return {}
            if isinstance(value, Mapping):
                return dict(value)
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                return model_dump(exclude_none=False)
            return dict(value)

        if legacy is None and canonical is not None:
            if not DiffusionLMConfig(**as_dict(canonical)).from_causal_lm:
                raise ValueError(
                    "The legacy diffusion plugin supports only "
                    "`diffusion_lm.from_causal_lm: true`; remove the plugin for "
                    "native diffusion_lm configuration."
                )
            return

        legacy_values = as_dict(legacy)
        translated = DiffusionLMConfig(
            **DiffusionConfig(**legacy_values).model_dump(), from_causal_lm=True
        ).model_dump()
        if canonical is not None:
            canonical_values = DiffusionLMConfig(**as_dict(canonical)).model_dump()
            if canonical_values != translated:
                raise ValueError(
                    "Configure only one of `diffusion` and `diffusion_lm`; "
                    "the legacy diffusion plugin cannot translate conflicting blocks."
                )
        else:
            cfg["diffusion_lm"] = DictDefault(translated)

        if not _LEGACY_WARNING_EMITTED:
            warnings.warn(
                "`diffusion` is deprecated; use `diffusion_lm` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _LEGACY_WARNING_EMITTED = True

    def get_input_args(self) -> str:
        """Returns the pydantic model for LLaDA plugin arguments."""
        return "axolotl.integrations.diffusion.DiffusionArgs"

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Perform actions after model is loaded."""
        self.cfg = cfg

    def get_trainer_cls(self, cfg: DictDefault) -> type | None:
        """Return custom trainer class for diffusion training."""
        from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer

        return AxolotlDiffusionTrainer

    def post_trainer_create(self, cfg: DictDefault, trainer: Any):
        """Configure trainer after creation."""
        if hasattr(trainer, "axolotl_cfg"):
            trainer.axolotl_cfg = cfg
        trainer.post_set_axolotl_cfg()
