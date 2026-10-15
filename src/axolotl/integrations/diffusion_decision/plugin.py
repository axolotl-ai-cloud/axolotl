"""Axolotl registration for typed diffusion-decision training."""

from __future__ import annotations

__ci_config_keys__ = ("diffusion_decision",)

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from axolotl.integrations.base import BasePlugin

from ._util import _value
from .args import DiffusionDecisionConfig
from .manifest import (
    MANIFEST_FILENAME,
    DecisionManifest,
    DecisionManifestCheckpointCallback,
    build_decision_manifest,
)


def _carries_images(*datasets: Any) -> bool:
    for dataset in datasets:
        for row in getattr(dataset, "_rows", ()):
            for item in (row,) if isinstance(row, Mapping) else row:
                if getattr(item.get("canvas"), "image_refs", ()):
                    return True
    return False


def _output_dir(cfg: Any) -> str | Path:
    output_dir: str | Path | None = (
        cfg.get("output_dir") if isinstance(cfg, Mapping) else cfg.output_dir
    )
    if not isinstance(output_dir, (str, Path)):
        raise ValueError("diffusion_decision manifest requires output_dir")
    return output_dir


class DiffusionDecisionPlugin(BasePlugin):
    """Register decision settings and defer layout handling to ``DiffusionSpec``."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.diffusion_decision.args.DiffusionDecisionArgs"

    @staticmethod
    def _decision_config(cfg: Any) -> DiffusionDecisionConfig | None:
        value = _value(cfg, "diffusion_decision")
        if value is None or isinstance(value, DiffusionDecisionConfig):
            return value
        if isinstance(value, Mapping):
            return DiffusionDecisionConfig.model_validate(value)
        raise TypeError("diffusion_decision must be a mapping")

    def register(self, cfg: dict) -> None:
        if self._decision_config(cfg) is None:
            return
        diffusion = cfg.get("diffusion_lm")
        if diffusion is None:
            raise ValueError("diffusion_decision requires native diffusion_lm settings")
        if _value(diffusion, "from_causal_lm", False):
            raise ValueError(
                "diffusion_decision does not support diffusion_lm.from_causal_lm"
            )
        self._validate_fixed_logical_batch(cfg)

    def _require_runtime(self, cfg: Any) -> DiffusionDecisionConfig | None:
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
                    "diffusion_decision sample_packing requires positive sequence_len and micro_batch_size"
                )
            return
        if (
            batch_flattening is False
            and isinstance(micro_batch_size, int)
            and (micro_batch_size > 1)
        ):
            raise ValueError(
                "diffusion_decision fixed logical batches require batch_flattening: true when micro_batch_size is greater than one"
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
                "diffusion_decision fixed logical batches require positive logical length, physical budget, and micro batch size"
            )

    def load_datasets(self, cfg, preprocess: bool = False):
        if self._require_runtime(cfg) is None:
            return None
        try:
            from .datasets import load_decision_datasets
        except ImportError as exc:
            raise RuntimeError(
                "diffusion_decision dataset integration is not installed"
            ) from exc
        return load_decision_datasets(cfg, preprocess=preprocess)

    def pre_lora_load(self, cfg, model) -> None:
        decision = self._require_runtime(cfg)
        if decision is None:
            return
        from .slot_runtime import (
            merge_trainable_slot_indices,
            resolve_trainable_slot_runtime,
        )

        runtime = resolve_trainable_slot_runtime(cfg, model, decision)
        if runtime is not None:
            merge_trainable_slot_indices(cfg, runtime)

    def get_trainer_cls(self, cfg):
        if self._require_runtime(cfg) is None:
            return None
        try:
            from .trainer import DiffusionDecisionTrainer
        except ImportError as exc:
            raise RuntimeError(
                "diffusion_decision trainer integration is not installed"
            ) from exc
        return DiffusionDecisionTrainer

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
            has_images=_carries_images(
                getattr(trainer, "train_dataset", None),
                getattr(trainer, "eval_dataset", None),
            ),
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
        if self._decision_config(cfg) is None:
            return
        source = Path(adapter_path) / MANIFEST_FILENAME
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
                "diffusion_decision collator integration is not installed"
            ) from exc
        return decision_collator_for_config(cfg, is_eval=is_eval)
