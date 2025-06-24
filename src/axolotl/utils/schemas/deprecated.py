"""Pydantic models for deprecated and remapped configuration parameters"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class DeprecatedParameters(BaseModel):
    """configurations that are deprecated"""

    max_packed_sequence_len: int | None = None
    rope_scaling: Any | None = None
    noisy_embedding_alpha: float | None = None
    dpo_beta: float | None = None
    evaluation_strategy: str | None = None

    @field_validator("max_packed_sequence_len")
    @classmethod
    def validate_max_packed_sequence_len(cls, max_packed_sequence_len):
        if max_packed_sequence_len:
            raise DeprecationWarning("`max_packed_sequence_len` is no longer supported")
        return max_packed_sequence_len

    @field_validator("rope_scaling")
    @classmethod
    def validate_rope_scaling(cls, rope_scaling):
        if rope_scaling:
            raise DeprecationWarning(
                "`rope_scaling` is no longer supported, it should now be be a key under `model_config`"
            )
        return rope_scaling

    @field_validator("noisy_embedding_alpha")
    @classmethod
    def validate_noisy_embedding_alpha(cls, noisy_embedding_alpha):
        if noisy_embedding_alpha:
            LOG.warning("noisy_embedding_alpha is deprecated, use neftune_noise_alpha")
        return noisy_embedding_alpha

    @field_validator("dpo_beta")
    @classmethod
    def validate_dpo_beta(cls, dpo_beta):
        if dpo_beta is not None:
            LOG.warning("dpo_beta is deprecated, use rl_beta instead")
        return dpo_beta

    @field_validator("evaluation_strategy")
    @classmethod
    def validate_evaluation_strategy(cls, evaluation_strategy):
        if evaluation_strategy is not None:
            LOG.warning("evaluation_strategy is deprecated, use eval_strategy instead")
        return evaluation_strategy


class RemappedParameters(BaseModel):
    """Parameters that have been remapped to other names"""

    overrides_of_model_config: dict[str, Any] | None = Field(
        default=None,
        alias="model_config",
        json_schema_extra={
            "description": "optional overrides to the base model configuration"
        },
    )
    overrides_of_model_kwargs: dict[str, Any] | None = Field(
        default=None,
        alias="model_kwargs",
        json_schema_extra={
            "description": "optional overrides the base model loading from_pretrained"
        },
    )
    type_of_model: str | None = Field(
        default=None,
        alias="model_type",
        json_schema_extra={
            "description": "If you want to specify the type of model to load, AutoModelForCausalLM is a good choice too"
        },
    )
    revision_of_model: str | None = Field(
        default=None,
        alias="model_revision",
        json_schema_extra={
            "description": "You can specify to choose a specific model revision from huggingface hub"
        },
    )
