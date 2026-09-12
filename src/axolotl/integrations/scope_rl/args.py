"""Input and training arguments for the SCOPE-RL plugin."""

from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator

TRAINER_CLS = "axolotl.integrations.scope_rl.trainer.ScopeRLAsyncGRPOTrainer"


class ScopeRLArgs(BaseModel):
    """Input args for SCOPE-RL entropy control (arXiv:2510.08141)."""

    scope_rl: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable SCOPE-RL entropy control. Requires async GRPO with vLLM."
        },
    )
    scope_target_entropy: float | None = Field(
        default=None,
        json_schema_extra={"description": "Target policy entropy H0 for SCOPE-RL."},
    )
    scope_alpha: float | None = Field(
        default=None,
        gt=0,
        le=1,
        json_schema_extra={
            "description": "Fraction of rollout groups resampled for the SCOPE-RL auxiliary branch, "
            "and the weight of its loss term."
        },
    )
    scope_temperature_min: float | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Lower clip for the SCOPE-RL auxiliary sampling temperature."
        },
    )
    scope_temperature_max: float | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Upper clip for the SCOPE-RL auxiliary sampling temperature."
        },
    )
    scope_positive_threshold: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Total reward at or above which a SCOPE-RL auxiliary sample counts as positive."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def check_scope_rl(cls, data):
        if not data.get("scope_rl"):
            return data
        trl = data.get("trl") or {}
        if data.get("rl") != "grpo":
            raise ValueError("scope_rl requires `rl: grpo`")
        # the auxiliary rollout is only issued on the prefetching producer's path
        if not trl.get("async_prefetch"):
            raise ValueError("scope_rl requires `trl.async_prefetch: true`")
        # token-normalised loss types weight rows by length, losing the alpha weighting
        if trl.get("loss_type") not in ("grpo", "sapo", "dr_grpo"):
            raise ValueError(
                "scope_rl requires `trl.loss_type` to be grpo, sapo or dr_grpo "
                "(TRL defaults to dapo)"
            )
        t_min, t_max = (
            data.get("scope_temperature_min"),
            data.get("scope_temperature_max"),
        )
        if t_min is not None and t_max is not None and t_min > t_max:
            raise ValueError(
                "`scope_temperature_min` must not exceed `scope_temperature_max`"
            )
        trainer_cls = data.get("trainer_cls")
        if trainer_cls and trainer_cls != TRAINER_CLS:
            raise ValueError(
                f"scope_rl runs on {TRAINER_CLS}; remove `trainer_cls: {trainer_cls}`"
            )
        return data


@dataclass
class ScopeRLTrainingArgsMixin:
    """Training args for SCOPE-RL (arXiv:2510.08141)."""

    scope_rl: bool = False
    scope_target_entropy: float = 0.5
    scope_alpha: float = 1 / 64
    scope_temperature_min: float = 0.8
    scope_temperature_max: float = 1.2
    scope_positive_threshold: float = 1.0


SCOPE_KEYS = tuple(ScopeRLTrainingArgsMixin.__dataclass_fields__)
