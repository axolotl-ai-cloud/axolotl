"""Pydantic models for training hyperparameters"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from transformers import SchedulerType
from transformers.training_args import OptimizerNames

from axolotl.utils.schemas.enums import CustomSupportedOptimizers

LOG = logging.getLogger(__name__)


class LrGroup(BaseModel):
    """Custom learning rate group configuration"""

    name: str
    modules: list[str]
    lr: float


class HyperparametersConfig(BaseModel):
    """Training hyperparams configuration subset"""

    gradient_accumulation_steps: int | None = Field(default=1)
    micro_batch_size: int | None = Field(
        default=1,
        json_schema_extra={"description": "per gpu micro batch size for training"},
    )
    batch_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Total batch size, we do not recommended setting this manually"
        },
    )
    eval_batch_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "per gpu micro batch size for evals, defaults to value of micro_batch_size"
        },
    )

    auto_find_batch_size: bool | None = None

    train_on_inputs: bool | None = False
    group_by_length: bool | None = None

    learning_rate: str | float
    embedding_lr: float | None = None
    embedding_lr_scale: float | None = None
    weight_decay: float | None = 0.0
    optimizer: (OptimizerNames | CustomSupportedOptimizers) | None = (
        OptimizerNames.ADAMW_TORCH_FUSED
    )
    optim_args: (str | dict[str, Any]) | None = Field(
        default=None,
        json_schema_extra={"description": "Optional arguments to supply to optimizer."},
    )
    optim_target_modules: (list[str] | Literal["all_linear"]) | None = Field(
        default=None,
        json_schema_extra={
            "description": "The target modules to optimize, i.e. the module names that you would like to train."
        },
    )
    torchdistx_path: str | None = None
    lr_scheduler: (SchedulerType | Literal["one_cycle"] | Literal["rex"]) | None = (
        SchedulerType.COSINE
    )
    lr_scheduler_kwargs: dict[str, Any] | None = None
    lr_quadratic_warmup: bool | None = None
    cosine_min_lr_ratio: float | None = None
    cosine_constant_lr_ratio: float | None = None
    lr_div_factor: float | None = None
    lr_groups: list[LrGroup] | None = None

    adam_epsilon: float | None = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    max_grad_norm: float | None = None
    num_epochs: float = Field(default=1.0)

    @field_validator("batch_size")
    @classmethod
    def hint_batch_size_set(cls, batch_size):
        if batch_size:
            LOG.warning(
                "%s\n%s",
                "batch_size is not recommended. Please use gradient_accumulation_steps instead.",
                "To calculate the equivalent gradient_accumulation_steps, divide batch_size / micro_batch_size / number of gpus.",
            )
        return batch_size

    @field_validator("learning_rate")
    @classmethod
    def convert_learning_rate(cls, learning_rate):
        if learning_rate and isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        return learning_rate
