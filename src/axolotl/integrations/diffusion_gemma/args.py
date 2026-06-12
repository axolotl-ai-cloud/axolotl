"""Config args for DiffusionGemma block-diffusion training (nested under `block_diffusion:`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BlockDiffusionConfig(BaseModel):
    """Nested config available under the `block_diffusion` key.

    DiffusionGemma is an encoder-decoder block-diffusion model: an autoregressive
    encoder consumes the prompt prefix into a KV cache, and a bidirectional decoder
    denoises a fixed-length ``canvas`` (block) of target tokens.
    """

    canvas_length: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Block (canvas) length to train on. Defaults to the model config's "
            "`canvas_length` (256 for the released checkpoint)."
        ),
    )
    corruption: Literal["uniform", "mask"] = Field(
        default="uniform",
        description=(
            "Forward noising process. 'uniform' resamples corrupted tokens from the "
            "vocabulary, matching DiffusionGemma's inference sampler. 'mask' is an "
            "absorbing variant that requires `mask_token_id`."
        ),
    )
    mask_token_id: int | None = Field(
        default=None,
        description="Token id used for absorbing ('mask') corruption.",
    )
    timestep_eps: float = Field(
        default=1e-3,
        ge=0.0,
        le=1.0,
        description="Minimum diffusion time, avoids divide-by-zero in ELBO weighting.",
    )
    loss_weighting: Literal["elbo", "uniform"] = Field(
        default="elbo",
        description="Per-example loss weight: 'elbo' uses 1/t, 'uniform' uses 1.0.",
    )
    self_conditioning_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of running the extra no-grad forward pass that produces "
            "self-conditioning logits, matching the inference-time self-conditioning."
        ),
    )
    frozen_fp4_experts: Literal["nvfp4", "mxfp4"] | None = Field(
        default=None,
        description=(
            "Quantize the fused MoE experts to a frozen torchao FP4 format on load "
            "(QLoRA-style 4-bit base). Consumed directly by the ScatterMoE kernel's "
            "selective dequant; requires use_scattermoe. 'nvfp4' mirrors the released "
            "NVFP4 checkpoints."
        ),
    )


class DiffusionGemmaArgs(BaseModel):
    """Plugin entry exposing the nested `block_diffusion` block to the core config."""

    block_diffusion: BlockDiffusionConfig = Field(
        default_factory=BlockDiffusionConfig,
        description="DiffusionGemma block-diffusion training configuration.",
    )
