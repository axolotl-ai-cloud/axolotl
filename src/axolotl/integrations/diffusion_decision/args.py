"""Configuration models for typed diffusion-decision training."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .vendored.djev_template import MAX_QUESTIONS


class DecisionMixtureConfig(BaseModel):
    """Deterministic source-mixture controls."""

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Per-source sampling weights keyed by record source; exclusive with temperature."
        },
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        json_schema_extra={
            "description": "Sample each source in proportion to its row count raised to this power; unset uses 0.5."
        },
    )
    per_batch_stratified: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Draw each microbatch with source-stratified sampling; must be false for packing or premixed data."
        },
    )
    max_examples_per_source: int | dict[str, int] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Cap on training rows kept per source, as one integer or a per-source mapping."
        },
    )
    loss_weight: dict[str, float] = Field(
        default_factory=dict,
        json_schema_extra={"description": "Per-source multiplier on each canvas loss."},
    )
    premixed: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Treat training rows as a prebuilt draw sequence; each row needs source_metadata.premix."
        },
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "DecisionMixtureConfig":
        if self.weights is not None and self.temperature is not None:
            raise ValueError(
                "mixture specifies explicit weights or temperature, not both"
            )
        if self.temperature is not None and not math.isfinite(self.temperature):
            raise ValueError("mixture temperature must be finite")
        for field_name, values in (
            ("weights", self.weights or {}),
            ("loss_weight", self.loss_weight),
        ):
            for source, value in values.items():
                if not source:
                    raise ValueError(
                        f"mixture {field_name} source names must be nonempty"
                    )
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"mixture {field_name} values must be finite and positive"
                    )
        if isinstance(self.max_examples_per_source, int):
            if self.max_examples_per_source < 1:
                raise ValueError("mixture max_examples_per_source must be positive")
        elif isinstance(self.max_examples_per_source, dict):
            for source, value in self.max_examples_per_source.items():
                if (
                    not source
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                ):
                    raise ValueError(
                        "mixture max_examples_per_source requires nonempty names and positive caps"
                    )
        if self.premixed:
            incompatible = []
            if self.weights is not None:
                incompatible.append("weights")
            if self.temperature is not None:
                incompatible.append("temperature")
            if self.max_examples_per_source is not None:
                incompatible.append("max_examples_per_source")
            if self.per_batch_stratified:
                incompatible.append("per_batch_stratified")
            if incompatible:
                raise ValueError(
                    "mixture premixed=true cannot be combined with "
                    + ", ".join(incompatible)
                )
        return self


class DecisionLabelsConfig(BaseModel):
    """Typed-label objective controls."""

    model_config = ConfigDict(extra="forbid")

    label_softmax: Literal["restricted", "full", "both"] = Field(
        default="both",
        json_schema_extra={
            "description": "Label CE over the question candidates, the full vocabulary, or both."
        },
    )
    full_ce_weighting: Literal["ce", "dft"] = Field(
        default="ce",
        json_schema_extra={
            "description": "Full-vocabulary CE for hard labels: plain CE or DFT probability weighting."
        },
    )
    brier_weight: float = Field(
        default=0.1,
        ge=0.0,
        json_schema_extra={
            "description": "Coefficient of the candidate-restricted Brier auxiliary loss."
        },
    )
    hard_label_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        json_schema_extra={
            "description": "Fraction of a one-hot CE target spread uniformly over the valid answer tokens."
        },
    )
    codebook: Literal["vendored26", "expanded52", "expanded128"] = Field(
        default="vendored26",
        json_schema_extra={
            "description": "Answer-label codebook: djev letters A-Z (26 options), A-Z plus a-z (52), or those plus "
            "selected Greek and Cyrillic letters (128); every label is one token in the model's tokenizer."
        },
    )

    @model_validator(mode="after")
    def validate_brier_weight(self) -> "DecisionLabelsConfig":
        if not math.isfinite(self.brier_weight):
            raise ValueError("labels brier_weight must be finite")
        if self.hard_label_smoothing and self.full_ce_weighting == "dft":
            raise ValueError("hard_label_smoothing requires full_ce_weighting=ce")
        return self


class DecisionLatentConfig(BaseModel):
    """Latent-slot initialization controls retained in the run manifest."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "pad", "pinned", "learned", "free", "mask", "prompt"] = Field(
        default="none",
        json_schema_extra={
            "description": "Latent-slot initialization; none adds no slots to the canvas."
        },
    )
    num_slots: int = Field(
        default=0,
        ge=0,
        json_schema_extra={"description": "Number of latent slots per canvas."},
    )
    sample_num_slots: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Sample a slot count up to num_slots per canvas instead of a fixed count."
        },
    )
    token_ids: list[int] = Field(
        default_factory=list,
        json_schema_extra={
            "description": "Slot token IDs for the pinned, learned, and prompt modes."
        },
    )
    free_update_policy: Literal["argmax"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Slot update rule between reads; required for mode free."
        },
    )

    @model_validator(mode="after")
    def validate_slots(self) -> "DecisionLatentConfig":
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("latent token_ids must be nonnegative")
        if self.mode == "none" and (self.num_slots or self.token_ids):
            raise ValueError("latent mode=none requires num_slots=0 and no token_ids")
        if self.sample_num_slots and self.mode == "none":
            raise ValueError("sample_num_slots requires a non-none latent mode")
        if self.sample_num_slots and self.num_slots < 1:
            raise ValueError("sample_num_slots requires num_slots to be positive")
        if self.mode == "free" and self.free_update_policy != "argmax":
            raise ValueError("latent mode=free requires free_update_policy=argmax")
        if self.mode != "free" and self.free_update_policy is not None:
            raise ValueError("free_update_policy is valid only for latent mode=free")
        return self


class DecisionCarryConfig(BaseModel):
    """Cross-read latent carry policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Reserved; carrying latent state across reads is not supported in training."
        },
    )

    @model_validator(mode="after")
    def reject_enabled(self) -> "DecisionCarryConfig":
        if self.enabled:
            raise ValueError("decision carry is not wired for training")
        return self


class DiffusionDecisionConfig(BaseModel):
    """Typed-decision data, latent, and readout settings."""

    model_config = ConfigDict(extra="forbid")

    layout: Literal["thought_block", "prompt_slots"] = Field(
        default="thought_block",
        json_schema_extra={
            "description": "Decision layout recorded in the adapter manifest."
        },
    )
    max_questions_per_canvas: int = Field(
        default=20,
        ge=1,
        le=MAX_QUESTIONS,
        json_schema_extra={
            "description": "Maximum questions grouped into one canvas; capped by the template."
        },
    )
    read_fraction: float = Field(
        default=1.0,
        json_schema_extra={
            "description": "Fraction of canvases trained at the fully masked read time; the rest use sampled diffusion times."
        },
    )
    max_image_size: int = Field(
        default=1400,
        ge=28,
        json_schema_extra={
            "description": "Longest image edge in pixels before token expansion; a multiple of 28, each 28x28 block is one prompt token."
        },
    )
    image_cache_size: int = Field(
        default=64,
        ge=0,
        json_schema_extra={
            "description": "Normalized image tensors each collator (and dataloader worker) keeps in an LRU; 0 disables the cache."
        },
    )
    mixture: DecisionMixtureConfig = Field(
        default_factory=DecisionMixtureConfig,
        json_schema_extra={"description": "Source sampling and loss weighting."},
    )
    labels: DecisionLabelsConfig = Field(
        default_factory=DecisionLabelsConfig,
        json_schema_extra={"description": "Label objective and answer codebook."},
    )
    latent: DecisionLatentConfig = Field(
        default_factory=DecisionLatentConfig,
        json_schema_extra={
            "description": "Optional latent slots placed in the canvas."
        },
    )
    carry: DecisionCarryConfig = Field(
        default_factory=DecisionCarryConfig,
        json_schema_extra={
            "description": "Cross-read latent carry; must stay disabled."
        },
    )

    @model_validator(mode="after")
    def validate_read_fraction(self) -> "DiffusionDecisionConfig":
        if not math.isfinite(self.read_fraction):
            raise ValueError("read_fraction must be finite")
        if not 0.0 <= self.read_fraction <= 1.0:
            raise ValueError("read_fraction must be in [0, 1]")
        if self.max_image_size % 28:
            raise ValueError("max_image_size must be a multiple of 28")
        return self


class DiffusionDecisionArgs(BaseModel):
    """Plugin entry that exposes the optional ``diffusion_decision`` block."""

    diffusion_decision: DiffusionDecisionConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Typed-decision training settings for native diffusion models."
        },
    )
