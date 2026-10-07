"""Typed decision training for native diffusion language models."""

__ci_config_keys__ = ("diffusion_decision",)

from .args import DiffusionDecisionArgs, DiffusionDecisionConfig
from .plugin import DiffusionDecisionPlugin
from .records import DecisionCanvas
from .slots import SlotInit

__all__ = [
    "DecisionCanvas",
    "DiffusionDecisionArgs",
    "DiffusionDecisionConfig",
    "DiffusionDecisionPlugin",
    "SlotInit",
]
