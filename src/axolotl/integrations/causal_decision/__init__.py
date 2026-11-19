"""Decision-head training for causal language models."""

__ci_config_keys__ = ("causal_decision",)

from .args import CausalDecisionArgs, CausalDecisionConfig
from .plugin import CausalDecisionPlugin

__all__ = ["CausalDecisionArgs", "CausalDecisionConfig", "CausalDecisionPlugin"]
