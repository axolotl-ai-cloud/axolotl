# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Axolotl plugin that routes training to a remote Hatchery/Tinker API."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoConfig, PreTrainedModel, Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class HatcheryPlugin(BasePlugin):
    """Plugin that replaces local training with remote API calls.

    Activated by adding to the axolotl YAML:

        plugins:
          - axolotl.integrations.hatchery.HatcheryPlugin

        hatchery:
          backend: tinker  # or "hatchery"
          lora_rank: 32
          learning_rate: 1e-4
          loss_fn: cross_entropy
          # ... see HatcheryArgs for full options
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.hatchery.args.HatcheryArgs"

    def register(self, cfg: dict):
        """Auto-set config values needed for remote training."""
        if cfg.get("remove_unused_columns") is None:
            cfg["remove_unused_columns"] = False

    def pre_model_load(self, cfg: DictDefault):
        """Replace model loading with a tiny stub."""
        hcfg = cfg.hatchery or {}
        backend = (
            hcfg.get("backend", "tinker")
            if isinstance(hcfg, dict)
            else getattr(hcfg, "backend", "tinker")
        )
        LOG.info(
            f"Hatchery plugin active: training dispatched to remote "
            f"{backend} API. Skipping local model weight loading."
        )

        from axolotl.loaders import ModelLoader

        def _stub_build_model(loader_self) -> bool:
            base_model = loader_self.cfg.base_model
            LOG.info(f"Skipping model weight loading for: {base_model}")

            config = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=loader_self.cfg.get("trust_remote_code", False),
            )

            class _Stub(PreTrainedModel):
                config_class = type(config)
                _no_split_modules: list[str] = []
                supports_gradient_checkpointing = False

                def __init__(self, cfg):
                    super().__init__(cfg)
                    vocab_size = getattr(cfg, "vocab_size", 32000)
                    self.embed_tokens = torch.nn.Embedding(vocab_size, 1)

                def get_input_embeddings(self):
                    return self.embed_tokens

                def set_input_embeddings(self, value):
                    pass

                def get_output_embeddings(self):
                    return None

            loader_self.model = _Stub(config)
            return True

        ModelLoader._build_model = _stub_build_model  # type: ignore[method-assign,assignment]

    def get_trainer_cls(self, cfg: DictDefault) -> type[Trainer] | None:
        """Return the appropriate remote trainer class."""
        hcfg = cfg.hatchery
        loss_fn = getattr(hcfg, "loss_fn", "cross_entropy") if hcfg else "cross_entropy"

        if loss_fn in ("importance_sampling", "ppo", "cispo", "dro"):
            from .rl_trainer import HatcheryRLTrainer

            return HatcheryRLTrainer

        from .trainer import HatcheryTrainer

        return HatcheryTrainer

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        model._hatchery_remote = True

    def post_train(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        LOG.info(
            "Hatchery: skipping local model save (weights are on remote API). "
            "Use `tinker checkpoint download` or hatchery CLI to retrieve."
        )

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer):
        """Inject hatchery config + axolotl training params into the trainer."""
        from .args import HatcheryConfig
        from .rl_trainer import HatcheryRLTrainer
        from .trainer import HatcheryTrainer

        if not isinstance(trainer, (HatcheryTrainer, HatcheryRLTrainer)):
            return

        hcfg = cfg.hatchery
        if isinstance(hcfg, dict):
            hatchery_config = HatcheryConfig(**hcfg)
        elif hcfg is None:
            hatchery_config = HatcheryConfig()
        else:
            hatchery_config = hcfg

        trainer.hatchery_args = hatchery_config
        trainer._base_model_name = cfg.base_model

        # Pull standard training params from axolotl config so they
        # don't need to be duplicated under hatchery:
        trainer._optim_params = {
            "learning_rate": cfg.learning_rate or 1e-4,
            "beta1": cfg.adam_beta1 or 0.9,
            "beta2": cfg.adam_beta2 or 0.95,
            "eps": cfg.adam_epsilon or 1e-12,
            "weight_decay": cfg.weight_decay or 0.0,
            "grad_clip_norm": cfg.max_grad_norm or 0.0,
        }
