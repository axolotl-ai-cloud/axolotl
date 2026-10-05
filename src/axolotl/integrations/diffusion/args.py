"""Deprecated plugin arguments for diffusion LM training."""

from pydantic import BaseModel, Field

from axolotl.utils.schemas.diffusion import DiffusionConfig


class DiffusionArgs(BaseModel):
    """Plugin entry that preserves the legacy ``diffusion`` block."""

    diffusion: DiffusionConfig = Field(
        default_factory=DiffusionConfig,
        description="Diffusion training configuration. Only nested block is supported.",
    )


__all__ = ["DiffusionArgs", "DiffusionConfig"]
