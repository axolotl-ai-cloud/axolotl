"""Pydantic models for training hyperparameters"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from transformers import SchedulerType
from transformers.training_args import OptimizerNames

from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import CustomSupportedOptimizers

LOG = get_logger(__name__)


class LrGroup(BaseModel):
    """Custom learning rate group configuration"""

    name: str
    modules: list[str]
    lr: float


class HyperparametersConfig(BaseModel):
    """Training hyperparams configuration subset"""

    gradient_accumulation_steps: int | None = Field(
        default=1,
        json_schema_extra={
            "description": "If greater than 1, backpropagation will be skipped and the gradients will be accumulated for the given number of steps."
        },
    )
    micro_batch_size: int | None = Field(
        default=1,
        json_schema_extra={
            "description": "The number of samples to include in each batch. This is the number of samples sent to each GPU. Batch size per gpu = micro_batch_size * gradient_accumulation_steps"
        },
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

    auto_find_batch_size: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "whether to find batch size that fits in memory. Passed to underlying transformers Trainer"
        },
    )

    train_on_inputs: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Whether to mask out or include the human's prompt from the training labels"
        },
    )
    group_by_length: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Group similarly sized data to minimize padding. May be slower to start, as it must download and sort the entire dataset. Note that training loss may have an oscillating pattern with this enabled."
        },
    )

    learning_rate: str | float
    embedding_lr: float | None = None
    embedding_lr_scale: float | None = None
    weight_decay: float | None = Field(
        default=0.0, json_schema_extra={"description": "Specify weight decay"}
    )
    optimizer: (OptimizerNames | CustomSupportedOptimizers) | None = Field(
        default=OptimizerNames.ADAMW_TORCH_FUSED,
        json_schema_extra={"description": "Specify optimizer"},
    )
    optim_args: (str | dict[str, Any]) | None = Field(
        default=None,
        json_schema_extra={
            "description": "Dictionary of arguments to pass to the optimizer"
        },
    )
    optim_target_modules: (list[str] | Literal["all_linear"]) | None = Field(
        default=None,
        json_schema_extra={
            "description": "The target modules to optimize, i.e. the module names that you would like to train, right now this is used only for GaLore algorithm"
        },
    )
    torchdistx_path: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Path to torch distx for optim 'adamw_anyprecision'"
        },
    )
    lr_scheduler: (
        SchedulerType | Literal["one_cycle"] | Literal["rex"]
    ) | None = SchedulerType.COSINE
    lr_scheduler_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Specify a scheduler and kwargs to use with the optimizer"
        },
    )
    lr_quadratic_warmup: bool | None = None
    cosine_min_lr_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "decay lr to some percentage of the peak lr, e.g. cosine_min_lr_ratio=0.1 for 10% of peak lr"
        },
    )
    cosine_constant_lr_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "freeze lr at some percentage of the step, e.g. cosine_constant_lr_ratio=0.8 means start cosine_min_lr at 80% of training step"
        },
    )
    lr_div_factor: float | None = Field(
        default=None, json_schema_extra={"description": "Learning rate div factor"}
    )
    lr_groups: list[LrGroup] | None = None

    adam_epsilon: float | None = Field(
        default=None, json_schema_extra={"description": "adamw hyperparams"}
    )
    adam_epsilon2: float | None = Field(
        default=None, json_schema_extra={"description": "only used for CAME Optimizer"}
    )
    adam_beta1: float | None = Field(
        default=None, json_schema_extra={"description": "adamw hyperparams"}
    )
    adam_beta2: float | None = Field(
        default=None, json_schema_extra={"description": "adamw hyperparams"}
    )
    adam_beta3: float | None = Field(
        default=None, json_schema_extra={"description": "only used for CAME Optimizer"}
    )

    dion_lr: float | None = Field(
        default=None, json_schema_extra={"description": "Dion Optimizer learning rate"}
    )
    dion_momentum: float | None = Field(
        default=None, json_schema_extra={"description": "Dion Optimizer momentum"}
    )
    dion_rank_fraction: float | None = Field(
        default=1.0,
        json_schema_extra={
            "description": "Dion Optimizer: r/d fraction for low-rank approximation. Used to compute the low-rank dimension."
        },
    )
    dion_rank_multiple_of: int | None = Field(
        default=1,
        json_schema_extra={
            "description": "Dion Optimizer: Round up the low-rank dimension to a multiple of this number. This may be useful to ensure even sharding."
        },
    )

    qgalore_rank: int | None = Field(
        default=256,
        json_schema_extra={
            "description": "Q-GaLore: rank r of the low-rank gradient projection. Smaller r reduces optimizer state but loses gradient information."
        },
    )
    qgalore_update_proj_gap: int | None = Field(
        default=200,
        json_schema_extra={
            "description": "Q-GaLore: maximum number of steps between SVD recomputations of the projection matrix. The adaptive scheduler may skip updates earlier based on cos_threshold."
        },
    )
    qgalore_scale: float | None = Field(
        default=0.25,
        json_schema_extra={
            "description": "Q-GaLore: scaling factor applied to the projected gradient after project_back. Equivalent to GaLore's `galore_scale`."
        },
    )
    qgalore_proj_type: str | None = Field(
        default="std",
        json_schema_extra={
            "description": "Q-GaLore: projection type for the GaLoreProjector. One of 'std', 'reverse_std', 'right', 'left', 'full'."
        },
    )
    qgalore_proj_quant: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Q-GaLore: enable INT-quantization of the projection matrix P (the resilient-to-quantization observation from the paper)."
        },
    )
    qgalore_proj_bits: int | None = Field(
        default=4,
        json_schema_extra={
            "description": "Q-GaLore: bitwidth for the quantized projection matrix when qgalore_proj_quant is True (paper default: 4)."
        },
    )
    qgalore_proj_group_size: int | None = Field(
        default=256,
        json_schema_extra={
            "description": "Q-GaLore: group size for projection-matrix quantization. Must evenly divide the projection's last dimension."
        },
    )
    qgalore_cos_threshold: float | None = Field(
        default=0.4,
        json_schema_extra={
            "description": "Q-GaLore: cosine-similarity threshold for the lazy subspace update. If the new P is within this similarity of the previous one, the SVD is skipped."
        },
    )
    qgalore_gamma_proj: int | None = Field(
        default=2,
        json_schema_extra={
            "description": "Q-GaLore: multiplicative factor by which update_proj_gap grows once a layer's subspace is judged stable."
        },
    )
    qgalore_queue_size: int | None = Field(
        default=5,
        json_schema_extra={
            "description": "Q-GaLore: length of the moving-average queue used by the adaptive-frequency scheduler."
        },
    )
    max_grad_norm: float | None = Field(
        default=None, json_schema_extra={"description": "Gradient clipping max norm"}
    )
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


class JaggedLRConfig(BaseModel):
    """JaggedLR configuration subset, can be used w/ ReLoRA training"""

    jagged_restart_steps: int | None = Field(
        default=None,
        json_schema_extra={"description": "how often to reset for jagged restarts"},
    )
    jagged_restart_warmup_steps: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "how many warmup steps to take after reset for jagged restarts"
        },
    )
    jagged_restart_anneal_steps: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "how many anneal steps to take before reset for jagged restarts"
        },
    )
