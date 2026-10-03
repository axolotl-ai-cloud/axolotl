"""Configuration models for typed decision training."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _default_open_jev_target_basis() -> dict[str, Literal["hard", "dist", "set"]]:
    return {
        "rule": "hard",
        "stochastic": "dist",
        "human": "dist",
        "uniform_set": "set",
    }


class DecisionMixtureConfig(BaseModel):
    """Deterministic source-mixture controls."""

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] | None = None
    temperature: float | None = Field(default=None, ge=0.0)
    per_batch_stratified: bool = True
    max_examples_per_source: int | dict[str, int] | None = None
    loss_weight: dict[str, float] = Field(default_factory=dict)
    premixed: bool = False

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

    label_softmax: Literal["restricted", "full", "both"] = "both"
    brier_weight: float = Field(default=0.1, ge=0.0)
    hard_label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    codebook: Literal["vendored26", "expanded52", "spreadsheet151", "reserved151"] = (
        "vendored26"
    )
    open_jev_target_basis: dict[str, Literal["hard", "dist", "set"]] = Field(
        default_factory=_default_open_jev_target_basis
    )

    @model_validator(mode="after")
    def validate_brier_weight(self) -> "DecisionLabelsConfig":
        if not math.isfinite(self.brier_weight):
            raise ValueError("labels brier_weight must be finite")
        return self


class DecisionLatentConfig(BaseModel):
    """Compatibility marker for the retained no-slot decision runtime."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_slots(self) -> "DecisionLatentConfig":
        return self


class DecisionEvalConfig(BaseModel):
    """Decision-read evaluation controls."""

    model_config = ConfigDict(extra="forbid")

    steps: list[int] = Field(default_factory=lambda: [1])
    noise_draws: int = Field(default=1, ge=1)
    log_per_source: bool = True

    @model_validator(mode="after")
    def validate_steps(self) -> "DecisionEvalConfig":
        if not self.steps or any(step < 1 for step in self.steps):
            raise ValueError("decision eval steps must contain positive values")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("decision eval steps must be unique")
        return self


class DecisionConfig(BaseModel):
    """Typed-decision data, latent, and readout settings."""

    model_config = ConfigDict(extra="forbid")

    layout: Literal["thought_block"] = "thought_block"
    reader: Literal["hf", "djev"] = "hf"
    answer_format: Literal["auto"] = "auto"
    max_questions_per_canvas: int = Field(default=20, ge=1)
    read_fraction: float = 1.0
    mixture: DecisionMixtureConfig = Field(default_factory=DecisionMixtureConfig)
    labels: DecisionLabelsConfig = Field(default_factory=DecisionLabelsConfig)
    latent: DecisionLatentConfig = Field(default_factory=DecisionLatentConfig)
    eval: DecisionEvalConfig = Field(default_factory=DecisionEvalConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_modes(cls, value):
        if not isinstance(value, dict):
            return value
        latent = value.get("latent")
        if isinstance(latent, dict) and latent.get("mode", "none") != "none":
            raise ValueError("decision latent slots were removed; use latent.mode: none")
        if "carry" in value:
            raise ValueError("decision latent carry was removed")
        if value.get("layout", "thought_block") != "thought_block":
            raise ValueError("decision prompt-slot layout was removed; use thought_block")
        labels = value.get("labels")
        if isinstance(labels, dict) and labels.get("full_ce_weighting", "ce") != "ce":
            raise ValueError("decision full_ce_weighting=dft was removed; use ce")
        return value

    @model_validator(mode="after")
    def validate_read_fraction(self) -> "DecisionConfig":
        if not math.isfinite(self.read_fraction):
            raise ValueError("read_fraction must be finite")
        if not 0.0 <= self.read_fraction <= 1.0:
            raise ValueError("read_fraction must be in [0, 1]")
        if self.reader == "djev" and self.labels.codebook != "vendored26":
            raise ValueError(
                f"DjevReader does not support the {self.labels.codebook} label codebook"
            )
        return self


class DecisionArgs(BaseModel):
    """Plugin entry that exposes the optional ``decision`` block."""

    decision: DecisionConfig | None = None
