"""Core diffusion LM interfaces.

Implementations stay lazily imported to avoid local configuration and trainer
import cycles.
"""

from .config import get_diffusion_config

__all__ = ["get_diffusion_config"]
