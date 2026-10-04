"""Native diffusion training and causal-LM diffusion compatibility plugin."""

from __future__ import annotations

import warnings
from typing import Any

from peft import PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.diffusion.args import normalize_diffusion_blocks
from axolotl.integrations.diffusion.schema import DiffusionLMConfig
from axolotl.model_support import (
    DiffusionLayout,
    Supported,
    get_model_support_for_cfg,
    resolve_model_support,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
_ALIAS_WARNING_EMITTED = False


class DiffusionPlugin(BasePlugin):
    """
    Plugin for diffusion language model training.

    Native Nemotron models use their own bidirectional objective. Existing
    causal-LM diffusion recipes retain random-mask training through this plugin.
    """

    def __init__(self):
        super().__init__()
        self.cfg = None

    def register(self, cfg: dict):
        """Normalize old and new input spellings to one runtime block."""
        global _ALIAS_WARNING_EMITTED  # pylint: disable=global-statement
        alias = cfg.get("diffusion_lm")
        normalized = normalize_diffusion_blocks(cfg)
        settings = normalized.get("diffusion")
        if settings is None:
            settings = {"from_causal_lm": True}
        legacy_mode = alias is None and settings["from_causal_lm"]

        normalization_messages: list[str] = []
        if legacy_mode:
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

        cfg.pop("diffusion_lm", None)
        cfg["diffusion"] = DictDefault(DiffusionLMConfig(**settings).model_dump())

        if alias is not None and not _ALIAS_WARNING_EMITTED:
            warnings.warn(
                "`diffusion_lm` is deprecated; use `diffusion` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _ALIAS_WARNING_EMITTED = True

    def get_input_args(self) -> str:
        """Return the schema for diffusion settings."""
        return "axolotl.integrations.diffusion.DiffusionArgs"

    @staticmethod
    def _native(cfg) -> bool:
        diffusion = getattr(cfg, "diffusion", None)
        return (
            diffusion is not None
            and not getattr(diffusion, "from_causal_lm", False)
            and getattr(cfg, "decision", None) is None
        )

    @staticmethod
    def _native_profile(cfg):
        if cfg.model_config_type != "nemotron_labs_diffusion":
            raise ValueError(
                "native diffusion currently supports Nemotron Labs Diffusion"
            )
        profile = resolve_model_support(get_model_support_for_cfg(cfg))
        if profile is None or profile.diffusion is None:
            raise ValueError("native diffusion requires a supported diffusion model")
        if cfg.attn_implementation == "varlen":
            if (
                profile.diffusion.layout is not DiffusionLayout.FULL_SEQUENCE
                or not isinstance(
                    profile.capabilities.get("diffusion_varlen"), Supported
                )
            ):
                raise ValueError(
                    "attn_implementation: varlen requires native full-sequence diffusion support"
                )
        if profile.is_multimodal or cfg.is_multimodal:
            raise ValueError("native diffusion requires a text-only model")
        return profile

    def load_datasets(self, cfg, preprocess: bool = False):
        if not self._native(cfg):
            return None
        self._native_profile(cfg)
        from .datasets import load_native_datasets

        return load_native_datasets(cfg, preprocess=preprocess)

    def get_training_args(self, cfg):
        if self._native(cfg):
            return {"remove_unused_columns": False}
        return None

    def get_collator_cls_and_kwargs(self, cfg, is_eval: bool = False):
        if not self._native(cfg):
            return None
        profile = self._native_profile(cfg)
        diffusion = cfg.diffusion
        spec = profile.diffusion
        from .lm.collator import NativeDiffusionPluginCollator
        from .lm.sampling import resolve_native_packing_budget

        budget = resolve_native_packing_budget(
            cfg,
            batch_size=(
                cfg.micro_batch_size
                if cfg.sample_packing
                and (not is_eval or cfg.eval_sample_packing is not False)
                else cfg.eval_batch_size
                if is_eval
                else cfg.micro_batch_size
            ),
        )

        return NativeDiffusionPluginCollator, {
            "layout": spec.layout.value,
            "logical_sequence_length": cfg.sequence_len,
            "physical_pack_budget": None if budget is None else budget.payload_capacity,
            "eos_tail": diffusion.eos_tail,
            "eos_token_id": getattr(cfg, "eos_token_id", None),
            "overflow_policy": diffusion.overflow_policy,
        }

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Perform actions after model is loaded."""
        self.cfg = cfg

    def get_trainer_cls(self, cfg: DictDefault) -> type | None:
        """Return custom trainer class for diffusion training."""
        if getattr(cfg, "decision", None) is not None:
            return None
        if self._native(cfg):
            self._native_profile(cfg)
        from axolotl.integrations.diffusion.lm.trainer import AxolotlDiffusionTrainer

        return AxolotlDiffusionTrainer

    def post_trainer_create(self, cfg: DictDefault, trainer: Any):
        """Configure trainer after creation."""
        if getattr(cfg, "decision", None) is not None:
            return
        if hasattr(trainer, "axolotl_cfg"):
            trainer.axolotl_cfg = cfg
        trainer.post_set_axolotl_cfg()
