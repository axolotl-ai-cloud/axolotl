"""Diffusion configuration access shared by native and compatibility paths."""

from __future__ import annotations

from typing import Any


def get_diffusion_config(cfg: Any) -> Any:
    """Return canonical diffusion settings."""
    if isinstance(cfg, dict):
        return cfg.get("diffusion")

    return getattr(cfg, "diffusion", None)
