"""Config args for diffusion LM training (nested under `diffusion:`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiffusionConfig(BaseModel):
    """Nested diffusion configuration available under the `diffusion` key."""

    # Noise schedule config
    noise_schedule: Literal["linear", "cosine"] = Field(
        default="linear", description="Type of noise schedule for diffusion training"
    )
    min_mask_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum masking ratio for diffusion noise schedule",
    )
    max_mask_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Maximum masking ratio for diffusion noise schedule",
    )
    num_diffusion_steps: int = Field(
        default=128, ge=1, description="Number of diffusion timesteps"
    )
    eps: float = Field(
        default=1e-3,
        ge=0.0,
        le=1.0,
        description="Epsilon value for minimum masking probability in forward process",
    )

    # Training config
    importance_weighting: bool = Field(
        default=True,
        description="Apply importance weighting to loss based on masking probability",
    )
    mask_token_id: int | None = Field(
        default=None,
        description=(
            "Token ID to use for masking. Unset by default; can use one of the "
            "tokenizer's special tokens here."
        ),
    )
    mask_token_str: str | None = Field(
        default=None,
        description=(
            "Token string to use as a mask. If `mask_token_id` is invalid or unset, "
            "this token will be ensured to exist as an additional special token and "
            "used. If absent, a default '<|diffusion_mask|>' will be added."
        ),
    )

    # Sample generation config
    generate_samples: bool = Field(
        default=True, description="Enable sample generation during training"
    )
    generation_interval: int = Field(
        default=100, ge=1, description="Generate samples every N steps"
    )
    num_generation_samples: int = Field(
        default=3, ge=1, description="Number of samples to generate each time"
    )
    generation_steps: int = Field(
        default=128, ge=1, description="Number of diffusion steps for generation"
    )
    generation_temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Temperature for generation sampling (0.0 = deterministic)",
    )
    generation_max_length: int = Field(
        default=100, ge=1, description="Maximum sequence length for generation"
    )

    @model_validator(mode="after")
    def _validate_mask_ratios(self) -> "DiffusionConfig":
        if self.min_mask_ratio > self.max_mask_ratio:
            raise ValueError("min_mask_ratio must be ≤ max_mask_ratio")
        return self


class DiffusionLMConfig(DiffusionConfig):
    """Canonical diffusion configuration available under ``diffusion``."""

    model_config = ConfigDict(extra="forbid")

    from_causal_lm: bool = Field(
        default=False,
        description="Use the causal-LM compatibility backend for legacy diffusion runs.",
    )
    canvas_width: int | None = Field(
        default=None,
        ge=1,
        description="Decision-read canvas width; native full-sequence training ignores it.",
    )
    t_eps: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Native diffusion-time lower bound; unset resolves from the model spec.",
    )
    time_weighting: Literal["none", "inv_t", "linear"] | None = Field(
        default=None,
        description="Named native objective weighting override.",
    )
    objective_reduction: (
        Literal["supervised_token_mean", "masked_token_mean", "example_mean"] | None
    ) = Field(
        default=None,
        description="Objective reduction override; unset keeps the model spec's reference reduction.",
    )
    treat_eos_as_one: bool | None = Field(
        default=None,
        description="Group a trailing EOS run into one absorbing-noise draw.",
    )
    eos_tail: Literal["none", "visible_supervised"] | None = Field(
        default=None,
        description="Logical EOS-tail layout policy, independent of EOS corruption.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_length_fields(cls, value):
        if isinstance(value, Mapping) and {
            "logical_sequence_length",
            "physical_pack_budget",
        }.intersection(value):
            raise ValueError(
                "diffusion.logical_sequence_length and "
                "diffusion.physical_pack_budget were removed; use top-level "
                "sequence_len and micro_batch_size"
            )
        return value

    overflow_policy: Literal["error", "drop"] = Field(
        default="error",
        description="Handling for an example that exceeds its logical layout budget.",
    )
    allow_native_vocab_resize: Literal[False] = Field(
        default=False,
        description="Native diffusion vocab resizing is disabled until explicitly validated.",
    )
    unroll: "DiffusionUnrollConfig | None" = Field(
        default=None,
        description="K-step denoising and gradient-through-step policy.",
    )


class DiffusionUnrollConfig(BaseModel):
    """Denoising unroll and gradient-retention settings."""

    k_max: int = Field(default=1, ge=1)
    grad_through_steps: bool = False
