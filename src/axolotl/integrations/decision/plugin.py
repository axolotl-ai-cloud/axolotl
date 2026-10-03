"""Axolotl registration for typed diffusion-decision training."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from axolotl.integrations.base import BasePlugin

from .args import DecisionConfig
from .manifest import (
    DecisionManifest,
    DecisionManifestCheckpointCallback,
    build_decision_manifest,
)


def _value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _output_dir(cfg: Any) -> str | Path:
    output_dir: str | Path | None = (
        cfg.get("output_dir") if isinstance(cfg, Mapping) else cfg.output_dir
    )
    if not isinstance(output_dir, (str, Path)):
        raise ValueError("decision manifest requires output_dir")
    return output_dir


class DecisionPlugin(BasePlugin):
    """Register decision settings and defer layout handling to ``DiffusionSpec``."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.decision.args.DecisionArgs"

    @staticmethod
    def _decision_config(cfg: Any) -> DecisionConfig | None:
        value = (
            cfg.get("decision")
            if isinstance(cfg, Mapping)
            else getattr(cfg, "decision", None)
        )
        if value is None or isinstance(value, DecisionConfig):
            return value
        if isinstance(value, Mapping):
            return DecisionConfig.model_validate(value)
        raise TypeError("decision must be a mapping")

    def register(self, cfg: dict) -> None:
        if self._decision_config(cfg) is None:
            return
        diffusion = cfg.get("diffusion")
        legacy_alias = cfg.get("diffusion_lm")
        if "diffusion" in cfg and "diffusion_lm" in cfg:
            raise ValueError("Configure only one of `diffusion` and `diffusion_lm`.")
        diffusion = diffusion if diffusion is not None else legacy_alias
        if diffusion is None:
            raise ValueError("decision requires native diffusion settings")
        from_causal_lm = (
            diffusion.get("from_causal_lm", legacy_alias is None)
            if isinstance(diffusion, Mapping)
            else getattr(diffusion, "from_causal_lm", legacy_alias is None)
        )
        if from_causal_lm:
            raise ValueError("decision does not support diffusion.from_causal_lm")
        self._validate_fixed_logical_batch(cfg)

    def _require_runtime(self, cfg: Any) -> DecisionConfig | None:
        decision = self._decision_config(cfg)
        if decision is None:
            return None
        self._validate_fixed_logical_batch(cfg)
        return decision

    @staticmethod
    def _validate_fixed_logical_batch(cfg: Any) -> None:
        batch_flattening = (
            cfg.get("batch_flattening")
            if isinstance(cfg, Mapping)
            else getattr(cfg, "batch_flattening", False)
        )
        micro_batch_size = _value(cfg, "micro_batch_size")
        sample_packing = bool(_value(cfg, "sample_packing", False))
        logical_length = _value(cfg, "sequence_len")
        if sample_packing:
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in (logical_length, micro_batch_size)
            ):
                raise ValueError(
                    "decision sample_packing requires positive sequence_len and micro_batch_size"
                )
            return
        if (
            batch_flattening is False
            and isinstance(micro_batch_size, int)
            and (micro_batch_size > 1)
        ):
            raise ValueError(
                "decision fixed logical batches require batch_flattening: true when micro_batch_size is greater than one"
            )
        if batch_flattening is not True:
            return
        physical_budget = (
            logical_length * micro_batch_size
            if isinstance(logical_length, int) and isinstance(micro_batch_size, int)
            else None
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (logical_length, physical_budget, micro_batch_size)
        ):
            raise ValueError(
                "decision fixed logical batches require positive logical length, physical budget, and micro batch size"
            )

    def load_datasets(self, cfg, preprocess: bool = False):
        if self._require_runtime(cfg) is None:
            return None
        try:
            from .datasets import load_decision_datasets
        except ImportError as exc:
            raise RuntimeError("decision dataset integration is not installed") from exc
        return load_decision_datasets(cfg, preprocess=preprocess)

    def pre_lora_load(self, cfg, model) -> None:
        self._require_runtime(cfg)

    def get_trainer_cls(self, cfg):
        if self._require_runtime(cfg) is None:
            return None
        try:
            from .trainer import DecisionTrainer
        except ImportError as exc:
            raise RuntimeError("decision trainer integration is not installed") from exc
        return DecisionTrainer

    def post_trainer_create(self, cfg, trainer) -> None:
        if self._require_runtime(cfg) is None:
            return
        tokenizer = getattr(trainer, "processing_class", None)
        if tokenizer is None:
            tokenizer = getattr(trainer, "tokenizer", None)
        self._decision_manifest = build_decision_manifest(
            cfg,
            tokenizer=tokenizer,
            model=getattr(trainer, "model", None),
        )
        from axolotl.utils.distributed import is_main_process

        if is_main_process():
            self._decision_manifest.write_to(_output_dir(cfg))
        trainer.add_callback(
            DecisionManifestCheckpointCallback(self._decision_manifest)
        )

    def post_train(self, cfg, model) -> None:
        if self._require_runtime(cfg) is None:
            return
        from axolotl.utils.distributed import is_main_process

        if not is_main_process():
            return
        manifest = getattr(self, "_decision_manifest", None)
        if not isinstance(manifest, DecisionManifest):
            manifest = build_decision_manifest(cfg, model=model)
        manifest.write_to(_output_dir(cfg))

    def post_lora_merge(self, cfg, adapter_path: str, output_path: str) -> None:
        source = Path(adapter_path) / "diffusion_decision_manifest.json"
        if not source.exists():
            return
        DecisionManifest.from_path(source)
        shutil.copy2(source, Path(output_path) / source.name)

    def get_collator_cls_and_kwargs(self, cfg, is_eval: bool = False):
        if self._require_runtime(cfg) is None:
            return None
        try:
            from .training_collator import decision_collator_for_config
        except ImportError as exc:
            raise RuntimeError(
                "decision collator integration is not installed"
            ) from exc
        return decision_collator_for_config(cfg, is_eval=is_eval)
