"""Axolotl registration for decision-head training on causal LMs."""

from __future__ import annotations

__ci_config_keys__ = ("causal_decision",)

from axolotl.integrations.base import BasePlugin


class CausalDecisionPlugin(BasePlugin):
    """Train a letter-logit decision head on ``diffusion_decision`` records."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.causal_decision.args.CausalDecisionArgs"

    def register(self, cfg: dict) -> None:
        if cfg.get("diffusion_lm"):
            raise ValueError(
                "causal_decision is for causal LMs; use diffusion_decision"
            )
        if cfg.get("sample_packing"):
            raise ValueError("causal_decision does not support sample_packing")

    def load_datasets(self, cfg, preprocess: bool = False):
        from .data import load_causal_decision_datasets

        return load_causal_decision_datasets(cfg)

    def get_trainer_cls(self, cfg):
        from .trainer import CausalDecisionTrainer

        return CausalDecisionTrainer

    def get_collator_cls_and_kwargs(self, cfg, is_eval: bool = False):
        from .data import CausalDecisionCollator

        return CausalDecisionCollator, {}
