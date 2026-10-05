"""Serializable decision-training provenance saved with adapters and checkpoints."""

from __future__ import annotations

__ci_config_keys__ = ("diffusion_decision",)

import json
import math
from pathlib import Path
from typing import Any, Literal, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    create_model,
    field_validator,
    model_validator,
)
from transformers import TrainerCallback

from axolotl.model_support import DiffusionSpec

from ._util import (
    CONTROL_TOKEN_FIELDS,
    _value,
    optional_token_id,
    parse_token_ids,
    require_diffusion_spec,
    resolve_mask_token_id,
)
from .args import DiffusionDecisionConfig

MANIFEST_FILENAME = "diffusion_decision_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


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


_SPEC_FIELD_OVERRIDES: dict[str, Any] = {
    "max_canvas": (StrictInt | None, Field(ge=1, default=None)),
    "max_context": (StrictInt | None, Field(ge=1, default=None)),
    "time_floor": (float, Field(ge=0.0, le=1.0)),
}


class _DiffusionSpecManifestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_spec_contract(self) -> "_DiffusionSpecManifestBase":
        DiffusionSpec(**self.model_dump())
        return self


DiffusionSpecManifest = create_model(
    "DiffusionSpecManifest",
    __base__=_DiffusionSpecManifestBase,
    __doc__="Validated primitive representation of the resolved model contract.",
    **{
        name: _SPEC_FIELD_OVERRIDES.get(name, (hint, ...))
        for name, hint in get_type_hints(DiffusionSpec).items()
    },
)


class _SlotConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["none", "pad", "pinned", "learned", "free", "mask", "prompt"]
    num_slots: StrictInt = Field(ge=0)
    token_ids: list[StrictInt] = Field(default_factory=list)
    free_update_policy: Literal["argmax"] | None = None
    free_slot_init_policy: Literal["prepared_canvas_v0", "fresh_runtime_v1"] = (
        "prepared_canvas_v0"
    )

    @field_validator("token_ids")
    @classmethod
    def validate_token_ids(cls, value: list[int]) -> list[int]:
        if any(token < 0 for token in value):
            raise ValueError("configured slot token IDs must be nonnegative")
        return value


class ConfiguredSlots(_SlotConfiguration):
    """Static slot configuration without runtime count sampling."""

    sample_num_slots: Literal[False] = False


class SampledTrainingSlots(BaseModel):
    """Versioned per-logical-draw slot-count policy for a sampled-slot run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["uniform_inclusive_v1"] = "uniform_inclusive_v1"
    min_slots: StrictInt = Field(default=0, ge=0)
    max_slots: StrictInt = Field(ge=1)
    draw_unit: Literal["logical_draw"] = "logical_draw"
    seed_derivation: Literal["seed_epoch_global_draw_ordinal_v1"] = (
        "seed_epoch_global_draw_ordinal_v1"
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "SampledTrainingSlots":
        if self.min_slots > self.max_slots:
            raise ValueError("sampled slot minimum cannot exceed its maximum")
        return self


class FixedMaxEvaluationSlots(BaseModel):
    """Fixed evaluation materialization used for a sampled-slot training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["fixed_configured_max_v1"] = "fixed_configured_max_v1"
    count: StrictInt = Field(ge=1)


class SampledConfiguredSlots(_SlotConfiguration):
    """Configured slots with distinct training-sampling and evaluation policies."""

    sample_num_slots: Literal[True] = True
    sampled_training: SampledTrainingSlots
    fixed_max_evaluation: FixedMaxEvaluationSlots

    @model_validator(mode="after")
    def validate_sampled_contract(self) -> "SampledConfiguredSlots":
        if self.mode == "none" or self.num_slots < 1:
            raise ValueError("sampled slots require a non-none mode and positive count")
        if self.sampled_training.max_slots != self.num_slots:
            raise ValueError("sampled slot maximum must equal configured num_slots")
        if self.fixed_max_evaluation.count != self.num_slots:
            raise ValueError(
                "fixed evaluation slot count must equal configured num_slots"
            )
        return self


_RETIRED_READ_FIELDS = frozenset({"eval_steps", "noise_draws", "reader"})


class NoiseReadProtocol(BaseModel):
    """Configured decision read and diffusion controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_fraction: float
    carry_enabled: bool
    diffusion_steps: StrictInt = Field(ge=1)
    t_eps: float | None = Field(default=None, ge=0.0, le=1.0)
    time_weighting: str | None = None
    objective_reduction: str | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_retired_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in _RETIRED_READ_FIELDS}
        return data

    @model_validator(mode="after")
    def validate_protocol(self) -> "NoiseReadProtocol":
        if (
            not math.isfinite(self.read_fraction)
            or not 0.0 <= self.read_fraction <= 1.0
        ):
            raise ValueError("read_fraction must be finite and in [0, 1]")
        if self.t_eps is not None and not math.isfinite(self.t_eps):
            raise ValueError("t_eps must be finite")
        return self


class DecisionManifest(BaseModel):
    """Validated, portable decision-adapter provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    model: ModelIdentity
    diffusion_spec: DiffusionSpecManifest
    decision_layout: Literal["thought_block", "prompt_slots"]
    canvas_width: StrictInt = Field(ge=1)
    label_codebook: Literal["vendored26", "expanded52"] = "vendored26"
    configured_slots: ConfiguredSlots | SampledConfiguredSlots
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


def _decision_config(cfg: Any) -> DiffusionDecisionConfig:
    value = _value(cfg, "diffusion_decision")
    if isinstance(value, DiffusionDecisionConfig):
        return value
    if isinstance(value, dict):
        return DiffusionDecisionConfig.model_validate(value)
    raise ValueError("diffusion_decision manifest requires diffusion_decision settings")


def build_decision_manifest(
    cfg: Any,
    *,
    tokenizer: Any = None,
    model: Any = None,
) -> DecisionManifest:
    """Build a validated manifest without changing model or tokenizer state."""
    decision = _decision_config(cfg)
    diffusion = _value(cfg, "diffusion_lm")
    if diffusion is None:
        raise ValueError("diffusion_decision manifest requires diffusion_lm settings")
    spec = require_diffusion_spec(cfg)
    model_config = _value(model, "config")
    mask_token_id = resolve_mask_token_id(diffusion, model_config)
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
        configured_slots=(
            SampledConfiguredSlots(
                mode=decision.latent.mode,
                num_slots=decision.latent.num_slots,
                token_ids=list(decision.latent.token_ids),
                free_update_policy=decision.latent.free_update_policy,
                free_slot_init_policy=(
                    "fresh_runtime_v1"
                    if decision.latent.mode == "free"
                    else "prepared_canvas_v0"
                ),
                sampled_training=SampledTrainingSlots(
                    max_slots=decision.latent.num_slots
                ),
                fixed_max_evaluation=FixedMaxEvaluationSlots(
                    count=decision.latent.num_slots
                ),
            )
            if decision.latent.sample_num_slots
            else ConfiguredSlots(
                mode=decision.latent.mode,
                num_slots=decision.latent.num_slots,
                token_ids=list(decision.latent.token_ids),
                sample_num_slots=False,
                free_update_policy=decision.latent.free_update_policy,
                free_slot_init_policy=(
                    "fresh_runtime_v1"
                    if decision.latent.mode == "free"
                    else "prepared_canvas_v0"
                ),
            )
        ),
        noise_read_protocol=NoiseReadProtocol(
            read_fraction=decision.read_fraction,
            carry_enabled=decision.carry.enabled,
            diffusion_steps=_value(diffusion, "num_diffusion_steps", 1),
            t_eps=_value(diffusion, "t_eps"),
            time_weighting=_value(diffusion, "time_weighting"),
            objective_reduction=_value(diffusion, "objective_reduction"),
        ),
        tokenizer_special_ids=TokenizerSpecialIds(
            **{
                name: optional_token_id(_value(tokenizer, name), name)
                for name in CONTROL_TOKEN_FIELDS
            },
            all_special_ids=list(
                parse_token_ids(_value(tokenizer, "all_special_ids"), "all_special_ids")
            ),
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
