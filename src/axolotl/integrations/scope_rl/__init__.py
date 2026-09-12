"""SCOPE-RL plugin: entropy control for async GRPO (arXiv:2510.08141)."""

from axolotl.integrations.base import BasePlugin

from .args import SCOPE_KEYS, TRAINER_CLS, ScopeRLArgs, ScopeRLTrainingArgsMixin

__all__ = ["ScopeRLArgs", "ScopeRLPlugin", "ScopeRLTrainingArgsMixin"]


class ScopeRLPlugin(BasePlugin):
    """Routes GRPO through the SCOPE-RL trainer and forwards its arguments."""

    def get_input_args(self):
        return "axolotl.integrations.scope_rl.args.ScopeRLArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.scope_rl.args.ScopeRLTrainingArgsMixin"

    def register(self, cfg):
        # The RL builder drops the model and reward funcs for plugin-supplied trainer
        # classes, so route through `trainer_cls`, which is applied after they are built.
        if cfg.get("scope_rl") and not cfg.get("trainer_cls"):
            cfg["trainer_cls"] = TRAINER_CLS

    def get_training_args(self, cfg):
        if not cfg.scope_rl:
            return {}
        return {
            key: getattr(cfg, key)
            for key in SCOPE_KEYS
            if getattr(cfg, key, None) is not None
        }
