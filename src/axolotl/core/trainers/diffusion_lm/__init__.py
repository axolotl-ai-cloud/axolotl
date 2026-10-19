"""Core diffusion LM interfaces.

Implementations stay lazily imported to avoid local configuration and trainer
import cycles.
"""

__ci_config_keys__ = ("diffusion", "diffusion_lm")

from .config import get_diffusion_config

__all__ = ["get_diffusion_config"]
