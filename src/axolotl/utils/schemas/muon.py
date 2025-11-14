"""Pydantic schema for Muon/MuonClip configuration."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MuonClipConfig(BaseModel):
    """Configuration for MuonClip orthogonalization and QK-Clip."""

    enabled: bool = Field(
        default=True,
        description="Enable the MuonClip controller (required for ZeRO-3/FSDP support).",
    )
    momentum: float = Field(
        default=0.95,
        description="Momentum coefficient used for Muon updates.",
    )
    weight_decay: float = Field(
        default=0.0,
        description="Weight decay applied during Muon updates (set 0 to keep optimizer decay).",
    )
    ns_steps: int = Field(
        default=5,
        description="Number of Newton–Schulz iterations for orthogonalization.",
        ge=1,
        le=16,
    )
    rms_scale: float | None = Field(
        default=None,
        description="Override the RMS scaling coefficient (defaults to Muon heuristic).",
    )
    apply_to: list[str] | None = Field(
        default=None,
        description="List of parameter name substrings to force-enable Muon updates.",
    )
    exclude: list[str] | None = Field(
        default_factory=lambda: ["embed", "lm_head"],
        description="Parameter name substrings to exclude from Muon updates.",
    )
    qk_clip: bool = Field(
        default=False,
        description="Whether to apply Moonshot-style QK-Clip to attention projections (currently instrumented for Llama/Qwen3).",
    )
    qk_clip_tau: float = Field(
        default=50.0,
        description="Attention logit threshold τ for QK-Clip.",
        gt=0,
    )
    qk_clip_alpha: float = Field(
        default=0.5,
        description="Blend between scaling query vs key weights (0=keys only,1=queries only).",
        ge=0.0,
        le=1.0,
    )
    qk_clip_max_steps: int | None = Field(
        default=None,
        description="Maximum number of optimizer steps to keep QK-Clip active before automatically disabling it.",
        ge=1,
    )
    gather_strategy: Literal["auto", "deepspeed", "fsdp", "none"] = Field(
        default="auto",
        description="Manual override for parameter gathering backend.",
    )

    @field_validator("apply_to", "exclude", mode="after")
    @classmethod
    def _dedupe(cls, value):
        if not value:
            return value
        return sorted(set(value))
