"""Diffusion configuration access shared by native and compatibility paths."""

from __future__ import annotations

from typing import Any


def get_diffusion_config(cfg: Any) -> Any:
    """Return canonical diffusion settings, falling back to legacy settings."""
    if isinstance(cfg, dict):
        diffusion_lm = cfg.get("diffusion_lm")
        return diffusion_lm if diffusion_lm is not None else cfg.get("diffusion")

    diffusion_lm = getattr(cfg, "diffusion_lm", None)
    return diffusion_lm if diffusion_lm is not None else getattr(cfg, "diffusion", None)
