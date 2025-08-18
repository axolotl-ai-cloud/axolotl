"""Config args for diffusion LM training."""

from typing import Literal

from pydantic import BaseModel, Field


class DiffusionArgs(BaseModel):
    """Arguments for diffusion LM training plugin."""

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

    # Forward process config
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
    mask_token_id: int = Field(
        default=128002,
        description=(
            "Token ID to use for masking. Default is 128002 "
            "(<|reserved_special_token_0|> for Llama 3.2)"
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
