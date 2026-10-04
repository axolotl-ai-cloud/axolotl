"""Serializable decision-training provenance saved with adapters and checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)
from transformers import TrainerCallback

from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
    get_model_support_for_cfg,
    resolve_model_support,
)

from .args import DecisionConfig

MANIFEST_FILENAME = "diffusion_decision_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_id(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return value


def _token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("all_special_ids must be a list of nonnegative integers")
    result: list[int] = []
    for token in value:
        token_id = _token_id(token, "all_special_ids")
        if token_id is None:
            raise ValueError("all_special_ids cannot contain None")
        result.append(token_id)
    return result


class ModelIdentity(BaseModel):
    """Resolved base-model identity retained by a decision adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str = Field(min_length=1)
    requested_revision: str | None = None
    resolved_revision: str | None = None
    model_config_type: str | None = None
    mask_token_id: StrictInt | None = Field(default=None, ge=0)


class TokenizerSpecialIds(BaseModel):
    """Tokenizer IDs required to reconstruct a decision canvas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bos_token_id: StrictInt | None = Field(default=None, ge=0)
    eos_token_id: StrictInt | None = Field(default=None, ge=0)
    pad_token_id: StrictInt | None = Field(default=None, ge=0)
    unk_token_id: StrictInt | None = Field(default=None, ge=0)
    mask_token_id: StrictInt | None = Field(default=None, ge=0)
    all_special_ids: list[StrictInt] = Field(default_factory=list)

    @field_validator("all_special_ids")
    @classmethod
    def validate_special_ids(cls, value: list[int]) -> list[int]:
        if any(token < 0 for token in value):
            raise ValueError("tokenizer special IDs must be nonnegative")
        return list(dict.fromkeys(value))


class DiffusionSpecManifest(BaseModel):
    """Validated primitive representation of the resolved model contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    noise: DiffusionNoise
    layout: DiffusionLayout
    logit_alignment: LogitAlignment
    first_position_alignment: FirstPositionAlignment
    self_conditioning: bool
    max_canvas: StrictInt | None = Field(ge=1, default=None)
    max_context: StrictInt | None = Field(ge=1, default=None)
    eos_handling: EosHandling
    mask_token_policy: MaskTokenPolicy
    default_time_weighting: TimeWeighting
    objective_reduction: ObjectiveReduction
    generation_adapter: GenerationAdapter
    time_floor: float = Field(ge=0.0, le=1.0)
    reduction_scope: ReductionScope

    @model_validator(mode="after")
    def validate_spec_contract(self) -> "DiffusionSpecManifest":
        if not math.isfinite(self.time_floor):
            raise ValueError("diffusion spec time_floor must be finite")
        DiffusionSpec(**self.model_dump())
        return self


class ConfiguredSlots(BaseModel):
    """Read legacy no-slot adapter metadata without supporting slot experiments."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["none"] = "none"
    num_slots: Literal[0] = 0
    token_ids: list[StrictInt] = Field(default_factory=list, max_length=0)
    sample_num_slots: Literal[False] = False
    free_update_policy: None = None
    free_slot_init_policy: Literal["prepared_canvas_v0"] = "prepared_canvas_v0"


class NoiseReadProtocol(BaseModel):
    """Configured decision read and diffusion controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reader: Literal["hf"]
    read_fraction: float
    eval_steps: list[StrictInt]
    noise_draws: StrictInt = Field(ge=1)
    carry_enabled: Literal[False]
    diffusion_steps: StrictInt = Field(ge=1)
    t_eps: float | None = Field(default=None, ge=0.0, le=1.0)
    time_weighting: str | None = None
    objective_reduction: str | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "NoiseReadProtocol":
        if (
            not math.isfinite(self.read_fraction)
            or not 0.0 <= self.read_fraction <= 1.0
        ):
            raise ValueError("read_fraction must be finite and in [0, 1]")
        if self.t_eps is not None and not math.isfinite(self.t_eps):
            raise ValueError("t_eps must be finite")
        if not self.eval_steps or any(step < 1 for step in self.eval_steps):
            raise ValueError("eval_steps must contain positive values")
        if len(set(self.eval_steps)) != len(self.eval_steps):
            raise ValueError("eval_steps must be unique")
        return self


class DecisionManifest(BaseModel):
    """Validated, portable decision-adapter provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    model: ModelIdentity
    diffusion_spec: DiffusionSpecManifest
    decision_layout: Literal["thought_block"]
    canvas_width: StrictInt = Field(ge=1)
    label_codebook: Literal[
        "vendored26", "expanded52", "spreadsheet151", "reserved151"
    ] = "vendored26"
    configured_slots: ConfiguredSlots
    noise_read_protocol: NoiseReadProtocol
    tokenizer_special_ids: TokenizerSpecialIds
    initial_common_adapter_sha256: str | None = None
    initial_common_adapter_fingerprint_reason: str | None = None
    initial_common_adapter_devices: list[str] = Field(default_factory=list)
    initial_common_adapter_dtypes: list[str] = Field(default_factory=list)
    initial_common_adapter_tensor_count: StrictInt = Field(default=0, ge=0)

    @classmethod
    def from_path(cls, path: str | Path) -> "DecisionManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def write_to(self, directory: str | Path) -> Path:
        destination = Path(directory) / MANIFEST_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def _decision_config(cfg: Any) -> DecisionConfig:
    value = _value(cfg, "decision")
    if isinstance(value, DecisionConfig):
        return value
    if isinstance(value, dict):
        return DecisionConfig.model_validate(value)
    raise ValueError("decision manifest requires decision settings")


def _diffusion_spec(cfg: Any) -> DiffusionSpec:
    profile = resolve_model_support(get_model_support_for_cfg(cfg))
    spec = None if profile is None else profile.diffusion
    if not isinstance(spec, DiffusionSpec):
        raise ValueError("decision manifest requires a resolved DiffusionSpec")
    return spec


def build_decision_manifest(
    cfg: Any,
    *,
    tokenizer: Any = None,
    model: Any = None,
) -> DecisionManifest:
    """Build a validated manifest without changing model or tokenizer state."""
    decision = _decision_config(cfg)
    diffusion = _value(cfg, "diffusion")
    if diffusion is None:
        raise ValueError("decision manifest requires diffusion settings")
    spec = _diffusion_spec(cfg)
    model_config = _value(model, "config")
    mask_token_id = _token_id(_value(model_config, "mask_token_id"), "mask_token_id")
    if mask_token_id is None:
        mask_token_id = _token_id(_value(diffusion, "mask_token_id"), "mask_token_id")
    canvas_width = _value(diffusion, "canvas_width")
    if canvas_width is None:
        canvas_width = 128
    return DecisionManifest(
        model=ModelIdentity(
            base_model=str(_value(cfg, "base_model")),
            requested_revision=_value(cfg, "revision_of_model"),
            resolved_revision=_value(model_config, "_commit_hash"),
            model_config_type=_value(cfg, "model_config_type"),
            mask_token_id=mask_token_id,
        ),
        diffusion_spec=DiffusionSpecManifest.model_validate(spec.to_dict()),
        decision_layout=decision.layout,
        canvas_width=canvas_width,
        label_codebook=decision.labels.codebook,
        configured_slots=ConfiguredSlots(mode="none", num_slots=0),
        noise_read_protocol=NoiseReadProtocol(
            reader=decision.reader,
            read_fraction=decision.read_fraction,
            eval_steps=list(decision.eval.steps),
            noise_draws=decision.eval.noise_draws,
            carry_enabled=False,
            diffusion_steps=_value(diffusion, "num_diffusion_steps", 1),
            t_eps=_value(diffusion, "t_eps"),
            time_weighting=_value(diffusion, "time_weighting"),
            objective_reduction=_value(diffusion, "objective_reduction"),
        ),
        tokenizer_special_ids=TokenizerSpecialIds(
            bos_token_id=_token_id(_value(tokenizer, "bos_token_id"), "bos_token_id"),
            eos_token_id=_token_id(_value(tokenizer, "eos_token_id"), "eos_token_id"),
            pad_token_id=_token_id(_value(tokenizer, "pad_token_id"), "pad_token_id"),
            unk_token_id=_token_id(_value(tokenizer, "unk_token_id"), "unk_token_id"),
            mask_token_id=_token_id(
                _value(tokenizer, "mask_token_id"), "mask_token_id"
            ),
            all_special_ids=_token_ids(_value(tokenizer, "all_special_ids")),
        ),
    )


class DecisionManifestCheckpointCallback(TrainerCallback):
    """Write validated decision provenance with Trainer checkpoint directories."""

    def __init__(self, manifest: DecisionManifest) -> None:
        self.manifest = manifest

    def on_save(self, args, state, control, **_kwargs):
        if state.is_world_process_zero:
            self.manifest.write_to(
                Path(args.output_dir) / f"checkpoint-{state.global_step}"
            )
        return control
