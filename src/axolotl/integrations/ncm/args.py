"""Config block for the NCM plugin. Model architecture goes in ``model_config``."""

from typing import Literal

from pydantic import BaseModel, Field


class NCMConfig(BaseModel):
    stage: Literal["tokenizer", "generator"] = Field(
        description=(
            "tokenizer: train the multiscale VQ-VAE from scratch (or continue a stage-1 "
            "checkpoint). generator: train the next-concept transformer over the frozen "
            "tokenizer at base_model (a stage-1 output or an existing NCM generator)."
        )
    )
    sample_every: int = Field(
        default=0,
        description="Generator stage: log unconditional samples every N steps (0 disables).",
    )
    num_samples: int = Field(default=4, description="Samples per sampling round.")
    sample_temperature: float = 1.0


class NCMArgs(BaseModel):
    ncm: NCMConfig | None = None
