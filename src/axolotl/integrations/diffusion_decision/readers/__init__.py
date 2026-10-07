"""In-process readers for structured diffusion decisions."""

__ci_config_keys__ = ("diffusion_decision",)

from .base import DecisionRead, ReadDiagnostics

__all__ = (
    "DecisionRead",
    "ReadDiagnostics",
)
