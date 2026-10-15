"""Config args for diffusion LM training (nested under `diffusion:`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    """Canonical diffusion configuration available under ``diffusion_lm``."""

    from_causal_lm: bool = Field(
        default=False,
        description="Use the causal-LM compatibility backend for legacy diffusion runs.",
    )
    generate_samples: bool = Field(
        default=False,
        description=(
            "Enable sample generation during training; defaults on only with "
            "from_causal_lm."
        ),
    )
    canvas_width: int | None = Field(
        default=None,
        ge=1,
        description="Canvas width override; unset resolves from the model diffusion spec.",
    )
    t_eps: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Native diffusion-time lower bound; unset resolves from the model spec.",
    )
    time_weighting: (
        Literal["none", "inv_t", "linear", "inv_one_minus_t", "cart", "loo"] | None
    ) = Field(
        default=None,
        description="Named native objective weighting override.",
    )
    objective_reduction: (
        Literal["supervised_token_mean", "masked_token_mean", "example_mean"] | None
    ) = Field(
        default=None,
        description="Objective reduction override; unset keeps the model spec's reference reduction.",
    )
    cart_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="CART context-reweighting probability when time_weighting=cart.",
    )
    token_reweighting: bool = Field(
        default=False,
        description="Apply Dream's source focal token reweighting.",
    )
    alpha: float = Field(
        default=0.25,
        ge=0.0,
        description="Dream focal token-reweighting scale.",
    )
    gamma: float = Field(
        default=2.0,
        ge=0.0,
        description="Dream focal token-reweighting exponent.",
    )
    rhine_weight_clip: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional maximum for Rhine 1 / (1 - t) token weights.",
    )
    treat_eos_as_one: bool | None = Field(
        default=None,
        description="Group a trailing EOS run into one absorbing-noise draw.",
    )
    eos_tail: Literal["none", "visible_supervised"] | None = Field(
        default=None,
        description="Logical EOS-tail layout policy, independent of EOS corruption.",
    )

    overflow_policy: Literal["error", "drop"] = Field(
        default="error",
        description="Handling for an example that exceeds its logical layout budget.",
    )
    allow_native_vocab_resize: Literal[False] = Field(
        default=False,
        description="Native diffusion vocab resizing is disabled until explicitly validated.",
    )
    self_conditioning: "SelfConditioningConfig | None" = Field(
        default=None,
        description="Native self-conditioning options; unset resolves from the model spec.",
    )
    unroll: "DiffusionUnrollConfig | None" = Field(
        default=None,
        description="K-step denoising and gradient-through-step policy.",
    )
    encoder_ar_weight: float | None = Field(
        default=None,
        ge=0.0,
        description="Clean encoder autoregressive-loss coefficient.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_generate_samples(cls, data):
        if isinstance(data, dict) and data.get("generate_samples") is None:
            data = {**data, "generate_samples": bool(data.get("from_causal_lm"))}
        return data


class SelfConditioningConfig(BaseModel):
    """Run choices for native two-pass self-conditioning."""

    p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Per-example probability of conditioning on a first-pass prediction.",
    )
    train_module: bool = Field(
        default=False,
        description="Train the self-conditioning module and save it with the adapter.",
    )


class DiffusionUnrollConfig(BaseModel):
    """Denoising unroll and gradient-retention settings."""

    k_max: int = Field(
        default=1,
        ge=1,
        description="Upper bound of the per-step read count, drawn uniformly from 1..k_max; the final read supplies the loss.",
    )
    grad_through_steps: bool = Field(
        default=False,
        description="Backpropagate through earlier reads; requires a self-conditioning model.",
    )
