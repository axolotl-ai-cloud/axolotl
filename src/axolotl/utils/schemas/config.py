"""Module with Pydantic models for configuration."""

import re
from typing import Annotated, Any, Literal

from accelerate.utils import is_fp8_available
from annotated_types import MinLen
from packaging import version
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from axolotl.utils.datasets import get_default_process_count
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.datasets import (
    DatasetConfig,
    DPODataset,
    KTODataset,
    PretrainingDataset,
    SFTDataset,
    StepwiseSupervisedDataset,
    SyntheticDataset,
)
from axolotl.utils.schemas.deprecated import DeprecatedParameters, RemappedParameters
from axolotl.utils.schemas.dynamic_checkpoint import DynamicCheckpointConfig
from axolotl.utils.schemas.enums import (
    ATTN_IMPLS_SUPPORTING_PACKING,
    ATTN_IMPLS_USING_FLASH_LIB,
    ATTN_IMPLS_WITHOUT_DTYPE_CAST,
    CANONICAL_ATTN_IMPLS,
    LEGACY_ATTN_FLAG_TO_IMPL,
    SHORT_FORM_ALIAS_TO_CANONICAL,
    ChatTemplate,
    RingAttnFunc,
    RLType,
)
from axolotl.utils.schemas.fsdp import FSDPConfig
from axolotl.utils.schemas.integrations import (
    CometConfig,
    GradioConfig,
    LISAConfig,
    MLFlowConfig,
    OpenTelemetryConfig,
    RayConfig,
    TrackioConfig,
    WandbConfig,
)
from axolotl.utils.schemas.internal import EnvCapabilities, GPUCapabilities
from axolotl.utils.schemas.model import (
    ModelInputConfig,
    ModelOutputConfig,
    SpecialTokensConfig,
)
from axolotl.utils.schemas.multimodal import MultiModalConfig
from axolotl.utils.schemas.peft import LoraConfig, ReLoRAConfig
from axolotl.utils.schemas.quantization import PTQConfig, QATConfig
from axolotl.utils.schemas.training import HyperparametersConfig, JaggedLRConfig
from axolotl.utils.schemas.trl import TRLConfig
from axolotl.utils.schemas.validation import ValidationMixin
from axolotl.utils.schemas.vllm import VllmConfig

LOG = get_logger(__name__)


class EBFTConfig(BaseModel):
    """Configuration for Energy-Based Fine-Tuning (EBFT)"""

    feature_layers: list[float] = Field(
        default=[0.25, 0.5, 0.75],
        json_schema_extra={
            "description": "Fractional layer depths for feature extraction (e.g., [0.25, 0.5, 0.75])"
        },
    )
    embed_method: Literal["last_token", "mean_pooling", "completion_mean", "concat"] = (
        Field(
            default="last_token",
            json_schema_extra={
                "description": "Embedding method: 'last_token', 'mean_pooling', 'completion_mean', or 'concat'"
            },
        )
    )
    use_whitening: bool = Field(
        default=False,
        json_schema_extra={"description": "Apply SVD whitening to feature embeddings"},
    )
    alignment_coef: float = Field(
        default=1.0,
        json_schema_extra={
            "description": "Coefficient for alignment reward (cosine similarity with ground truth)"
        },
    )
    diversity_coef: float = Field(
        default=1.0,
        json_schema_extra={
            "description": "Coefficient for diversity penalty (pairwise similarity between samples)"
        },
    )
    ce_coef: float = Field(
        default=0.0,
        json_schema_extra={
            "description": "Cross-entropy loss coefficient on ground-truth tokens"
        },
    )
    adaptive_max_tokens: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Set per-batch max_tokens based on ground-truth length"
        },
    )
    gt_length_multiplier: float = Field(
        default=1.5,
        ge=0.1,
        json_schema_extra={
            "description": "Multiplier for ground-truth token count when computing adaptive max_tokens"
        },
    )

    # Strided mode fields (for unstructured text)
    mode: Literal["structured", "strided"] = Field(
        default="structured",
        json_schema_extra={
            "description": "EBFT mode: 'structured' (QA with vLLM) or 'strided' (unstructured text)"
        },
    )
    stride: int = Field(
        default=8,
        ge=1,
        json_schema_extra={"description": "Stride between anchor points (tokens)"},
    )
    context_length: int = Field(
        default=8,
        ge=1,
        json_schema_extra={"description": "Context window size per block"},
    )
    generate_max_len: int = Field(
        default=8,
        ge=1,
        json_schema_extra={"description": "Tokens to generate per block"},
    )
    n_samples_per_prompt: int = Field(
        default=4,
        ge=1,
        json_schema_extra={"description": "Independent rollouts per document"},
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        json_schema_extra={
            "description": "Sampling temperature for strided generation"
        },
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        json_schema_extra={"description": "Top-p nucleus sampling threshold"},
    )
    rl_coef: float = Field(
        default=1.0,
        json_schema_extra={"description": "RL policy gradient loss coefficient"},
    )
    advantage_estimator: Literal["rloo", "group_norm", "reinforce"] = Field(
        default="rloo",
        json_schema_extra={
            "description": "Advantage estimator: 'rloo', 'group_norm', 'reinforce'"
        },
    )
    min_completion_prefix: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Minimum tokens into completion before placing anchors. "
            "Skips anchors too close to the prompt boundary where features are dominated by prompt context."
        },
    )


class AxolotlInputConfig(
    ModelInputConfig,
    ModelOutputConfig,
    LoraConfig,
    ReLoRAConfig,
    JaggedLRConfig,
    HyperparametersConfig,
    WandbConfig,
    MLFlowConfig,
    CometConfig,
    TrackioConfig,
    OpenTelemetryConfig,
    LISAConfig,
    GradioConfig,
    RayConfig,
    MultiModalConfig,
    RemappedParameters,
    DeprecatedParameters,
    ValidationMixin,
    BaseModel,
):
    """Wrapper of all config options."""

    model_config = {"populate_by_name": True}

    strict: bool | None = Field(
        default=False,
        json_schema_extra={"description": "Allow overwrite yml config using from cli"},
    )
    resume_from_checkpoint: str | None = Field(
        default=None,
        json_schema_extra={"description": "Resume from a specific checkpoint dir"},
    )
    auto_resume_from_checkpoints: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "If resume_from_checkpoint isn't set and you simply want it to start where it left off. Be careful with this being turned on between different models."
        },
    )
    resize_token_embeddings_to_32x: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Resize the model embeddings when new tokens are added to multiples of 32. This is reported to improve training speed on some models"
        },
    )
    mean_resizing_embeddings: bool | None = False
    # optionally shrink the embeddings when the tokenizer vocab size is smaller
    shrink_embeddings: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to shrink the embeddings to len(tokenizer). By default, we won't shrink."
        },
    )
    embeddings_skip_upcast: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Don't upcast the embeddings to float32 when using PEFT. Useful for low-VRAM GPUs"
        },
    )
    reinit_weights: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Reinitialize model weights randomly instead of loading pretrained weights"
        },
    )

    trainer_cls: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "module to custom trainer class to use for training"
        },
    )

    rl: RLType | None = Field(
        default=None,
        json_schema_extra={
            "description": "Use RL training: 'dpo', 'ipo', 'kto', 'simpo', 'orpo', 'grpo', 'ebft'"
        },
    )
    trl: TRLConfig | None = Field(
        default_factory=lambda: TRLConfig(),
    )
    vllm: VllmConfig | None = Field(
        default_factory=lambda: VllmConfig(),
    )
    ebft: EBFTConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Configuration for Energy-Based Fine-Tuning (EBFT)"
        },
    )
    qat: QATConfig | None = None
    quantization: PTQConfig | None = None
    reward_model: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Reward modelling: `True` or `False`"},
    )
    dynamic_checkpoint: DynamicCheckpointConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Configuration for dynamic checkpointing (trigger by file or signal). "
            "Set 'enabled: true' to activate this feature."
        },
    )
    process_reward_model: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Process reward modelling: `True` or `False`"
        },
    )
    center_rewards_coefficient: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Coefficient to incentivize the reward model to output mean-zero rewards (proposed by https://huggingface.co/papers/2312.09244, Eq. 2). Recommended value: `0.01`."
        },
    )
    num_labels: int | None = None
    # Whether to use weighting in DPO trainer.
    # If `None`, default is `False` in the trainer.
    dpo_use_weighting: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to perform weighting in DPO trainer"
        },
    )
    dpo_label_smoothing: float | None = None
    precompute_ref_log_probs: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Precompute reference model log probabilities for DPO"
        },
    )

    dpo_use_liger_kernel: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Whether to use Liger kernel for DPO loss."},
    )

    dpo_padding_free: bool | None = None

    dpo_loss_type: Annotated[list[str], MinLen(1)] | None = Field(
        default=None,
        json_schema_extra={"description": "List of DPO losses to use."},
    )

    dpo_loss_weights: Annotated[list[float], MinLen(1)] | None = Field(
        default=None,
        json_schema_extra={"description": "Weights for each DPO loss."},
    )

    datasets: (
        Annotated[
            list[
                SFTDataset
                | DPODataset
                | KTODataset
                | StepwiseSupervisedDataset
                | SyntheticDataset
            ],
            MinLen(1),
        ]
        | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "A list of one or more datasets to finetune the model with"
        },
    )

    test_datasets: (
        Annotated[
            list[
                SFTDataset
                | DPODataset
                | KTODataset
                | StepwiseSupervisedDataset
                | SyntheticDataset
            ],
            MinLen(1),
        ]
        | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "A list of one or more datasets to eval the model with. You can use either test_datasets, or val_set_size, but not both."
        },
    )
    shuffle_merged_datasets: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "If false, the datasets will not be shuffled and will keep their original order in `datasets`. The same applies to the `test_datasets` option and the `pretraining_dataset` option. Default is true."
        },
    )
    shuffle_before_merging_datasets: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "If true, each dataset in `datasets` will be shuffled before merging. This allows curriculum learning strategies to be applied at the dataset level. Default is false."
        },
    )
    dataset_prepared_path: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Axolotl attempts to save the dataset as an arrow after packing the data together so subsequent training attempts load faster, relative path"
        },
    )
    dataset_shard_num: int | None = Field(
        default=None, json_schema_extra={"description": "Num shards for whole dataset"}
    )
    dataset_shard_idx: int | None = Field(
        default=None,
        json_schema_extra={"description": "Index of shard to use for whole dataset"},
    )
    skip_prepare_dataset: bool | None = False
    num_dataset_shards_to_save: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of shards to save the prepared dataset"
        },
    )

    pretraining_dataset: (
        Annotated[list[PretrainingDataset | SFTDataset], MinLen(1)] | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "Set to HF dataset for type: 'completion' for streaming instead of pre-tokenize"
        },
    )
    dataset_processes: int | None = Field(
        default=None,
        deprecated="Use `dataset_num_proc` instead. This parameter will be removed in a future version.",
        json_schema_extra={
            "description": (
                "The maximum number of processes to use while preprocessing your input dataset. This defaults to `os.cpu_count()` if not set.\n"
                "For Runpod VMs, it will default to number of vCPUs via RUNPOD_CPU_COUNT."
            )
        },
    )
    dataset_num_proc: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "The maximum number of processes to use while preprocessing your input dataset. This defaults to `os.cpu_count()` if not set.\n"
                "For Runpod VMs, it will default to number of vCPUs via RUNPOD_CPU_COUNT."
            )
        },
    )

    dataset_exact_deduplication: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Deduplicates datasets and test_datasets with identical entries"
        },
    )
    dataset_keep_in_memory: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Keep dataset in memory while preprocessing. Only needed if cached dataset is taking too much storage"
        },
    )
    dataloader_pin_memory: bool | None = None
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_drop_last: bool | None = None

    accelerator_config: dict[str, Any] | None = None

    remove_unused_columns: bool | None = None

    push_dataset_to_hub: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Push prepared dataset to hub - repo_org/repo_name"
        },
    )
    hf_use_auth_token: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use hf `use_auth_token` for loading datasets. Useful for fetching private datasets. Required to be true when used in combination with `push_dataset_to_hub`"
        },
    )

    device: Any | None = None
    device_map: Any | None = Field(
        default=None,
        json_schema_extra={
            "description": "Passed through to transformers when loading the model when launched without accelerate. Use `sequential` when training w/ model parallelism to limit memory"
        },
    )
    world_size: int | None = None
    local_rank: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Don't mess with this, it's here for accelerate and torchrun"
        },
    )
    ddp: bool | None = None

    seed: int | None = Field(
        default=None, json_schema_extra={"description": "Seed for reproducibility"}
    )
    ddp_timeout: int | None = Field(
        default=None,
        json_schema_extra={"description": "Advanced DDP Arguments - timeout"},
    )
    ddp_bucket_cap_mb: int | None = Field(
        default=None,
        json_schema_extra={"description": "Advanced DDP Arguments - bucket cap in MB"},
    )
    ddp_broadcast_buffers: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Advanced DDP Arguments - broadcast buffers"},
    )
    ddp_find_unused_parameters: bool | None = None

    do_causal_lm_eval: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to run causal language model evaluation for metrics in `eval_causal_lm_metrics`"
        },
    )
    eval_causal_lm_metrics: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "HF evaluate metrics used during evaluation. Default is ['sacrebleu', 'comet', 'ter', 'chrf', 'perplexity']"
        },
    )
    do_bench_eval: bool | None = None
    bench_dataset: str | None = None
    bench_split: str | None = None
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None

    loss_watchdog_threshold: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "High loss value, indicating the learning has broken down (a good estimate is ~2 times the loss at the start of training)"
        },
    )
    loss_watchdog_patience: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of high-loss steps in a row before the trainer aborts (default: 3)"
        },
    )

    gc_steps: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Run garbage collection every `gc_steps` steps. -1 will run on epoch end and before evaluations. Default is 0 (disabled)."
        },
    )

    bf16: Literal["auto"] | bool | None = Field(
        default="auto",
        json_schema_extra={
            "description": "Use CUDA bf16. bool or 'full' for `bf16_full_eval`, or 'auto' for automatic detection. require >=ampere"
        },
    )
    fp16: bool | None = Field(
        default=None, json_schema_extra={"description": "Use CUDA fp16"}
    )
    fp8: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable FP8 mixed precision training using TorchAO. Best "
            "used in combination with torch.compile."
        },
    )
    fp8_enable_fsdp_float8_all_gather: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable FSDP float8 all-gather optimization for FP8 training. Can "
            "improve training speed by 10-15% when FSDP is enabled."
        },
    )
    bfloat16: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "No AMP (automatic mixed precision) - require >=ampere"
        },
    )  # for non-AMP cases
    float16: bool | None = Field(
        default=None,
        json_schema_extra={"description": "No AMP (automatic mixed precision)"},
    )  # for non-AMP cases
    tf32: Literal["auto"] | bool | None = Field(
        default="auto",
        json_schema_extra={
            "description": "bool to use CUDA tf32 or 'auto' for automatic detection - require >=ampere"
        },
    )
    float32: bool | None = None

    gradient_checkpointing: Literal["offload", "offload_disk"] | bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Whether to use gradient checkpointing. Available options are: true, false, 'offload', 'offload_disk'. https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing"
        },
    )
    gradient_checkpointing_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Additional kwargs to pass to the trainer for gradient checkpointing"
        },
    )
    activation_offloading: Literal["legacy", "disk"] | bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Whether to offload activations. Available options are: true, false, 'legacy', 'disk'."
        },
    )
    layer_offloading: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Offload model layer parameters to CPU during forward, prefetch back during backward."
        },
    )

    freeze_mm_modules: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Freeze multimodal encoder parameters (vision, audio, etc.) for "
            "text-only training of multimodal models. When True, parameters belonging to "
            "vision towers, audio towers, multimodal projectors, and similar non-language "
            "modules are frozen (requires_grad=False). This allows DDP training without "
            "ddp_find_unused_parameters=True."
        },
    )

    unfrozen_parameters: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "List of regex patterns for parameter names to keep unfrozen. "
            "All other parameters will be frozen via requires_grad=False. "
            "Note: range-based patterns (e.g. embed_tokens.weight$[:32000]) use gradient "
            "zeroing rather than a true freeze, so weight decay will still apply to the "
            "frozen portion and optimizer states are allocated for the full parameter."
        },
    )

    sequence_len: int = Field(
        default=512,
        json_schema_extra={
            "description": "The maximum length of an input to train with, this should typically be less than 2048 as most models have a token/context limit of 2048"
        },
    )
    excess_length_strategy: Literal["drop", "truncate", "raise"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "What to do when a tokenized row exceeds sequence_len. 'drop' removes the row; 'truncate' slices tensors to sequence_len; 'raise' raises a ValueError. Defaults to 'drop' for backward compatibility."
        },
    )
    eval_sequence_len: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "The maximum length of an input for evaluation. If not specified, defaults to sequence_len"
        },
    )
    min_sample_len: int | None = None
    max_prompt_len: int | None = Field(
        default=None,
        json_schema_extra={"description": "maximum prompt length for RL training"},
    )
    sample_packing: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Use efficient multi-packing with block diagonal attention and per sequence position_ids. Recommend set to 'true'"
        },
    )
    sample_packing_group_size: int | None = Field(
        default=100_000,
        json_schema_extra={
            "description": "The number of samples packed at a time. Increasing the following values helps with packing, but usually only slightly (<%1.)"
        },
    )
    sample_packing_bin_size: int | None = Field(
        default=200,
        json_schema_extra={
            "description": "The number of samples which can be packed into one sequence. Increase if using a large sequence_len with many short samples."
        },
    )
    sample_packing_sequentially: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Whether to pack samples sequentially"},
    )
    sample_packing_mp_start_method: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The multiprocessing start method to use for packing. Should be 'fork', 'spawn' or 'forkserver'"
        },
    )
    eval_sample_packing: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Set to 'false' if getting errors during eval with sample_packing on"
        },
    )
    pad_to_sequence_len: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Pad inputs so each step uses constant sized buffers. This will reduce memory fragmentation and may prevent OOMs, by re-using memory more efficiently. Defaults to True if `sample_packing` enabled"
        },
    )
    pad_to_multiple_of: int | None = Field(
        default=None,
        json_schema_extra={
            "description": ("Pad each batch to a multiple of this value.")
        },
    )
    curriculum_sampling: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use sequential sampling for curriculum learning"
        },
    )
    multipack_real_batches: bool | None = None

    batch_flattening: Literal["auto"] | bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Use batch flattening for speedups when not using sample_packing"
        },
    )

    # for PoSE context length extension
    use_pose: bool | None = None
    pose_split_on_token_ids: list[int] | None = None
    pose_max_context_len: int | None = None
    pose_num_chunks: int | None = None

    # Deprecated: Use streaming_multipack_buffer_size instead
    pretrain_multipack_buffer_size: int | None = Field(
        default=None,
        deprecated="Deprecated in v0.13.0, will be removed in v0.14.0. Use streaming_multipack_buffer_size instead",
    )
    pretrain_multipack_attn: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "whether to prevent cross attention for packed sequences during pretraining",
        },
    )
    pretraining_sample_concatenation: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "whether to concatenate samples during pretraining",
        },
    )

    streaming: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Use streaming mode for loading datasets"},
    )
    streaming_multipack_buffer_size: int | None = Field(
        default=10_000,
        json_schema_extra={
            "description": "Buffer size for multipack streaming datasets"
        },
    )

    xformers_attention: bool | None = Field(
        default=None,
        deprecated="Use `attn_implementation: xformers` instead.",
        json_schema_extra={
            "description": "[DEPRECATED] Use `attn_implementation: xformers`. https://github.com/facebookresearch/xformers"
        },
    )
    sdp_attention: bool | None = Field(
        default=None,
        deprecated="Use `attn_implementation: sdpa` instead.",
        json_schema_extra={
            "description": "[DEPRECATED] Use `attn_implementation: sdpa`."
        },
    )
    s2_attention: bool | None = Field(
        default=None,
        deprecated="Use `attn_implementation: s2` instead.",
        json_schema_extra={
            "description": "[DEPRECATED] Use `attn_implementation: s2`. Shifted-sparse attention (only llama) - https://arxiv.org/pdf/2309.12307.pdf"
        },
    )
    flex_attention: bool | None = Field(
        default=None,
        deprecated="Use `attn_implementation: flex_attention` instead.",
    )
    flex_attn_compile_kwargs: dict[str, Any] | None = None
    flash_attention: bool | None = Field(
        default=None,
        deprecated="Use `attn_implementation: flash_attention_2` instead.",
        json_schema_extra={
            "description": "[DEPRECATED] Use `attn_implementation: flash_attention_2`. https://github.com/Dao-AILab/flash-attention"
        },
    )
    flash_attn_cross_entropy: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use flash-attention cross entropy implementation - advanced use only"
        },
    )
    flash_attn_rms_norm: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use flash-attention rms norm implementation - advanced use only"
        },
    )
    flash_attn_fuse_mlp: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to fuse part of the MLP into a single operation"
        },
    )
    flash_optimum: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Whether to use bettertransformers"},
    )
    sage_attention: bool | None = Field(
        default=None,
        deprecated="Use `attn_implementation: sage` instead.",
        json_schema_extra={
            "description": "[DEPRECATED] Use `attn_implementation: sage`. https://github.com/thu-ml/SageAttention"
        },
    )

    eager_attention: bool | None = Field(
        default=None,
        deprecated="Use `attn_implementation: eager` instead.",
    )

    attn_implementation: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Attention backend. Canonical values: eager, sdpa, flash_attention_2, "
                "flash_attention_3, flex_attention, xformers, sage, s2, fp8. Hub-kernel "
                "paths (e.g. kernels-community/flash-attn3) are also accepted and passed "
                "through to transformers."
            )
        },
    )

    gemma4_hybrid_attn_impl: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Use hybrid attention for Gemma 4: flash_attention_2 for sliding window layers "
            "and sdpa for global (full_attention) layers. Global layers have head_dim=512 which "
            "exceeds flash attention's supported size."
        },
    )

    experts_implementation: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Which experts implementation to use for MoE models,"
        },
    )

    quantize_moe_experts: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Quantize MoE expert weights on load to reduce VRAM. "
            "Requires adapter (lora/qlora) with load_in_4bit or load_in_8bit. "
            "Requires CUDA (not compatible with ROCm or other backends). "
            "Note: total parameter count may be reported incorrectly when enabled "
            "(trainable param count is correct)."
        },
    )

    scaling_softmax: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use Scaled Softmax (SSMax) attention. Ref: https://arxiv.org/abs/2501.19399"
        },
    )
    scaling_softmax_factor: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Scaling factor for SSMax attention. Default is 0.43"
        },
    )
    scaling_softmax_bias: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Bias for SSMax attention. Default is 0.0. Note: The paper recommends bias=0 for better length generalization."
        },
    )

    lora_mlp_kernel: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Apply custom LoRA autograd functions and activation function Triton kernels for speed and memory savings. See: https://docs.axolotl.ai/docs/lora_optims.html"
        },
    )
    lora_qkv_kernel: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Apply custom LoRA autograd functions and activation function Triton kernels for speed and memory savings. See: https://docs.axolotl.ai/docs/lora_optims.html"
        },
    )
    lora_o_kernel: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Apply custom LoRA autograd functions and activation function Triton kernels for speed and memory savings. See: https://docs.axolotl.ai/docs/lora_optims.html"
        },
    )
    lora_embedding_kernel: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Apply custom LoRA autograd function for embedding layers. See: https://docs.axolotl.ai/docs/lora_optims.html"
        },
    )

    chunked_cross_entropy: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use chunked cross entropy loss for memory efficiency"
        },
    )
    chunked_cross_entropy_num_chunks: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of chunks to use for chunked cross entropy loss"
        },
    )
    use_eaft: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable Entropy-Aware Focal Training loss (EAFT)"
        },
    )
    eaft_alpha: float | None = Field(
        default=1.0,
        json_schema_extra={
            "description": "Exponent for entropy weighting in EAFT (default: 1.0)"
        },
    )
    eaft_k: int | None = Field(
        default=20,
        json_schema_extra={
            "description": "Number of top logits for entropy approximation (default: 20)"
        },
    )

    tiled_mlp: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use ALST tiled mlp for memory efficient long context"
        },
    )

    tiled_mlp_num_shards: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of shards to use for ALST tiled mlp. If unset, it will be set based on seqlen/hidden_size"
        },
    )

    tiled_mlp_use_original_mlp: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to use original mlp for ALST tiled mlp. Otherwise uses a generic MLP based on llama."
        },
    )

    llama4_linearized_experts: bool | None = None

    deepspeed: str | dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Deepspeed config path. e.g., deepspeed_configs/zero3.json"
        },
    )
    deepcompile: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use deepcompile for faster training with deepspeed"
        },
    )
    fsdp: list[str] | None = Field(
        default=None,
        json_schema_extra={"description": "FSDP configuration"},
        deprecated="Configuring FSDP using `fsdp` is deprecated. Please use `fsdp_config` instead. ",
    )
    fsdp_config: FSDPConfig | None = Field(
        default=None, json_schema_extra={"description": "FSDP configuration options"}
    )
    fsdp_version: int | None = Field(
        default=None,
        json_schema_extra={"description": "FSDP version"},
    )
    fsdp_final_state_dict_type: (
        Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"] | None
    ) = Field(
        default=None,
        deprecated="Configuring FSDP final state dict type using `fsdp_final_state_dict_type` is deprecated. Please use `fsdp_config.final_state_dict_type` instead.",
    )

    val_set_size: float | None = Field(
        default=0.0,
        json_schema_extra={
            "description": "How much of the dataset to set aside as evaluation. 1 = 100%, 0.50 = 50%, etc. 0 for no eval."
        },
    )

    dp_shard_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of devices to shard across. If not set, will use all available devices."
        },
    )
    dp_replicate_size: int | None = Field(
        default=None,
        json_schema_extra={"description": "Number of devices to replicate across."},
    )
    sequence_parallel_degree: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Deprecated: use `context_parallel_size` instead"
        },
    )
    context_parallel_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Set to a divisor of the number of GPUs available to split sequences into chunks of equal size. Use in long context training to prevent OOM when sequences cannot fit into a single GPU's VRAM. E.g., if 4 GPUs are available, set this value to 2 to split each sequence into two equal-sized subsequences, or set to 4 to split into four equal-sized subsequences. See https://docs.axolotl.ai/docs/sequence_parallelism.html for more details."
        },
    )
    heads_k_stride: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Optional; strides across the key dimension. Larger values use more memory but should make training faster. Must evenly divide the number of KV heads in your model."
        },
    )
    ring_attn_func: RingAttnFunc | None = Field(
        default=None,
        json_schema_extra={
            "description": "One of 'varlen_llama3', 'batch_ring', 'batch_zigzag', 'batch_stripe'. Defaults to 'varlen_llama3' in the sample packing case, and 'batch_ring' in the non-sample packing case."
        },
    )
    tensor_parallel_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of tensor parallel processes in TP group. Only supported with DeepSpeed AutoTP."
        },
    )
    special_tokens: SpecialTokensConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Add or change special tokens. If you add tokens here, you don't need to add them to the `tokens` list."
        },
    )
    tokens: list[str] | None = Field(
        default=None,
        json_schema_extra={"description": "Add extra tokens to the tokenizer"},
    )
    added_tokens_overrides: dict[int, str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Mapping token_id to new_token_string to override reserved added_tokens in the tokenizer. Only works for tokens that are not part of the base vocab (aka are added_tokens). Can be checked if they exist in tokenizer.json added_tokens."
        },
    )

    torch_compile: Literal["auto"] | bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use torch.compile and which backend to use. setting to `auto` will enable torch compile when torch>=2.6.0"
        },
    )
    torch_compile_backend: str | None = Field(
        default=None,
        json_schema_extra={"description": "Backend to use for torch.compile"},
    )
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] | None = (
        None
    )

    max_steps: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Maximum number of iterations to train for. It precedes num_epochs which means that if both are set, num_epochs will not be guaranteed. e.g., when 1 epoch is 1000 steps => `num_epochs: 2` and `max_steps: 100` will train for 100 steps"
        },
    )
    warmup_steps: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of warmup steps. Cannot use with warmup_ratio"
        },
    )
    warmup_ratio: float | None = Field(
        default=None,
        json_schema_extra={"description": "Warmup ratio. Cannot use with warmup_steps"},
    )
    eval_steps: int | float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Leave empty to eval at each epoch, integer for every N steps. float for fraction of total steps"
        },
    )
    evals_per_epoch: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of times per epoch to run evals, mutually exclusive with eval_steps"
        },
    )
    eval_strategy: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Set to `no` to skip evaluation, `epoch` at end of each epoch, leave empty to infer from `eval_steps`"
        },
    )

    save_steps: int | float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Leave empty to save at each epoch, integer for every N steps. float for fraction of total steps"
        },
    )
    saves_per_epoch: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of times per epoch to save a checkpoint, mutually exclusive with save_steps"
        },
    )
    save_strategy: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Set to `no` to skip checkpoint saves, `epoch` at end of each epoch, `best` when better result is achieved, leave empty to infer from `save_steps`"
        },
    )
    save_total_limit: int | None = Field(
        default=None, json_schema_extra={"description": "Checkpoints saved at a time"}
    )
    save_first_step: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to checkpoint a model after the first step of training. Defaults to False."
        },
    )

    logging_steps: int | None = Field(
        default=None, json_schema_extra={"description": "Logging frequency"}
    )
    early_stopping_patience: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Stop training after this many evaluation losses have increased in a row. https://huggingface.co/transformers/v4.2.2/_modules/transformers/trainer_callback.html#EarlyStoppingCallback"
        },
    )
    load_best_model_at_end: bool | None = False
    save_only_model: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Save only the model weights, skipping the optimizer. Using this means you can't resume from checkpoints."
        },
    )
    use_tensorboard: bool | None = Field(
        default=None, json_schema_extra={"description": "Use tensorboard for logging"}
    )
    profiler_steps: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable the pytorch profiler to capture the first N steps of training to the output_dir. see https://pytorch.org/blog/understanding-gpu-memory-1/ for more information. Snapshots can be visualized @ https://pytorch.org/memory_viz"
        },
    )
    profiler_steps_start: int | None = Field(
        default=0,
        json_schema_extra={
            "description": "Which step to start the profiler at. Useful for only capturing a few steps mid-run."
        },
    )
    include_tokens_per_second: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "bool of whether to report tokens per second at the end of training. This is not supported with pre-training datasets."
        },
    )
    include_tkps: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "bool of whether to report tokens per second per-gpu during training by measuring throughput of non-padding tokens."
        },
    )
    neftune_noise_alpha: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "NEFT https://arxiv.org/abs/2310.05914, set this to a number (paper default is 5) to add noise to embeddings. Currently only supported on Llama and Mistral"
        },
    )

    orpo_alpha: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Parameter controlling the relative ratio loss weight in the ORPO loss. Passed to `beta` in `ORPOConfig` due to trl mapping."
        },
    )
    simpo_gamma: float | None = Field(
        default=None,
        json_schema_extra={"description": "Target reward margin for the SimPO loss"},
    )
    cpo_alpha: float | None = Field(
        default=None, json_schema_extra={"description": "Weight of the BC regularizer"}
    )

    kto_desirable_weight: float | None = Field(
        default=None,
        json_schema_extra={"description": "Factor for desirable loss term in KTO loss"},
    )
    kto_undesirable_weight: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "Factor for undesirable loss term in KTO loss"
        },
    )
    rl_beta: float | None = Field(
        default=None,
        json_schema_extra={"description": "The beta parameter for the RL training"},
    )

    max_memory: dict[int | Literal["cpu", "disk"], int | str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Defines the max memory usage per gpu on the system. Passed through to transformers when loading the model."
        },
    )
    gpu_memory_limit: int | str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Limit the memory for all available GPUs to this amount (if an integer, expressed in gigabytes); default: unset"
        },
    )
    low_cpu_mem_usage: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Whether to use low_cpu_mem_usage"},
    )

    chat_template: (
        ChatTemplate
        | Annotated[str, StringConstraints(pattern="^tokenizer_default_fallback_")]
    ) | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the chat template to use for training, following values are supported: tokenizer_default: Uses the chat template that is available in the tokenizer_config.json. If the chat template is not available in the tokenizer, it will raise an error. This is the default value. alpaca/inst/chatml/gemma/cohere/llama3/phi_3/deepseek_v2/jamba: These chat templates are available in the axolotl codebase at src/axolotl/utils/chat_templates.py. tokenizer_default_fallback_*: where * is the name of the chat template to fallback to. E.g. tokenizer_default_fallback_chatml. This is useful when the chat template is not available in the tokenizer. jinja: Uses a custom jinja template for the chat template. The custom jinja template should be provided in the chat_template_jinja field. The selected chat template will be saved to the tokenizer_config.json for easier inferencing"
        },
    )
    chat_template_jinja: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Custom jinja template or path to jinja file for chat template. This will be only used if chat_template is set to `jinja` or `null` (in which case chat_template is automatically set to `jinja`). Default is null."
        },
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Additional kwargs to pass to the chat template. This is useful for customizing the chat template. For example, you can pass `thinking=False` to add a generation prompt to the chat template."
        },
    )
    eot_tokens: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Custom EOT (End-of-Turn) tokens to mask/unmask during training. These tokens mark the boundaries between conversation turns. For example: ['/INST', '</s>', '[/SYSTEM_PROMPT]']. If not specified, defaults to just the model's eos_token. This is useful for templates that use multiple delimiter tokens."
        },
    )
    default_system_message: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Changes the default system message. Currently only supports chatml."
        },
    )

    fix_untrained_tokens: int | list[int] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Token index or indices to adjust embedding weights to the mean of the other tokens. "
                "This is useful when the model has untrained embeddings."
            )
        },
    )

    # INTERNALS - document for now, generally not set externally
    is_preprocess: bool | None = None
    preprocess_iterable: bool | None = None

    total_num_tokens: int | None = Field(
        default=None,
        json_schema_extra={"description": "Total number of tokens - internal use"},
    )
    total_supervised_tokens: int | None = None
    sample_packing_eff_est: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "You can set these packing optimizations AFTER starting a training at least once. The trainer will provide recommended values for these values."
        },
    )
    axolotl_config_path: str | None = None

    is_falcon_derived_model: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Internal use only - Used to identify which the model is based on"
        },
    )
    is_llama_derived_model: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Internal use only - Used to identify which the model is based on"
        },
    )
    is_mistral_derived_model: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Internal use only - Used to identify which the model is based on. Please note that if you set this to true, `padding_side` will be set to 'left' by default"
        },
    )
    is_qwen_derived_model: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Internal use only - Used to identify which the model is based on"
        },
    )

    plugins: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Add plugins to extend the pipeline. See `src/axolotl/integrations` for the available plugins or doc below for more details. https://docs.axolotl.ai/docs/custom_integrations.html"
        },
    )
    generate_samples: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Enable sample generation during training for monitoring"
        },
    )
    num_generation_samples: int | None = Field(
        default=3,
        json_schema_extra={
            "description": "Number of samples to generate at each interval"
        },
    )
    generation_max_new_tokens: int | None = Field(
        default=50,
        json_schema_extra={"description": "Maximum new tokens to generate per sample"},
    )
    generation_temperature: float | None = Field(
        default=0.7,
        json_schema_extra={
            "description": "Temperature for sample generation (0.0 = greedy)"
        },
    )
    generation_top_p: float | None = Field(
        default=None,
        json_schema_extra={"description": "Nucleus sampling parameter for generation"},
    )
    generation_top_k: int | None = Field(
        default=None,
        json_schema_extra={"description": "Top-k sampling parameter for generation"},
    )
    generation_prompt_ratio: float | None = Field(
        default=0.5,
        json_schema_extra={"description": "Ratio of input to use as prompt (0.0-1.0)"},
    )
    generation_do_sample: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to use sampling (vs greedy decoding)"
        },
    )

    @field_serializer("datasets")
    def datasets_serializer(
        self, ds_configs: list[DatasetConfig] | None
    ) -> list[dict[str, Any]] | None:
        if ds_configs:
            return [ds_config.model_dump(exclude_none=True) for ds_config in ds_configs]
        return None

    # --- Attention capability flags (derived from attn_implementation) ---

    @computed_field  # type: ignore[misc]
    @property
    def attn_supports_packing(self) -> bool:
        return self.attn_implementation in ATTN_IMPLS_SUPPORTING_PACKING

    @computed_field  # type: ignore[misc]
    @property
    def attn_uses_flash_lib(self) -> bool:
        return self.attn_implementation in ATTN_IMPLS_USING_FLASH_LIB

    @computed_field  # type: ignore[misc]
    @property
    def attn_needs_dtype_cast(self) -> bool:
        if self.attn_implementation is None:
            return False
        return self.attn_implementation not in ATTN_IMPLS_WITHOUT_DTYPE_CAST

    @model_validator(mode="before")
    @classmethod
    def warn_peft_trainable_token_to_fix_untrained(cls, data):
        if (
            peft_trainable_token_indices := data.get("peft_trainable_token_indices")
        ) and (fix_untrained_tokens := data.get("fix_untrained_tokens")):
            if isinstance(fix_untrained_tokens, int):
                fix_untrained_tokens = (fix_untrained_tokens,)

            if isinstance(peft_trainable_token_indices, int):
                peft_trainable_token_indices = (peft_trainable_token_indices,)

            for untrained_token_id in fix_untrained_tokens:
                if untrained_token_id not in peft_trainable_token_indices:
                    LOG.warning_once(
                        f"Token {untrained_token_id} is fixed via `fix_untrained_tokens`, yet not in `peft_trainable_token_indices: ` list. "
                        "Please add it, otherwise the token won't be trained on."
                    )
        return data

    @model_validator(mode="before")
    @classmethod
    def normalize_attn_implementation(cls, data):
        """Map legacy boolean attention flags to the canonical `attn_implementation`.

        `attn_implementation` is the single source of truth on the validated
        config. Legacy booleans (`flash_attention: true`, …) are input-only
        aliases; this validator warns, maps them to their canonical value, and
        strips them from `data` so they cannot be read downstream.

        Raises if a canonical `attn_implementation` is set alongside any legacy
        boolean — users must pick one.
        """
        if not isinstance(data, dict):
            return data

        attn_impl = data.get("attn_implementation")
        set_flags = [f for f in LEGACY_ATTN_FLAG_TO_IMPL if data.get(f)]

        # gemma4_hybrid requires flash_attention_2 for the sliding-window layers;
        # post-load patching swaps global layers to sdpa (see
        # `_apply_gemma_hybrid_attention`). Default it in when the user didn't
        # pick a backend; reject any incompatible explicit choice.
        if data.get("gemma4_hybrid_attn_impl"):
            if not attn_impl and not set_flags:
                data["attn_implementation"] = "flash_attention_2"
                attn_impl = "flash_attention_2"
            elif attn_impl and attn_impl != "flash_attention_2":
                raise ValueError(
                    f"gemma4_hybrid_attn_impl requires attn_implementation="
                    f"flash_attention_2 (sliding-window layers run under FA2); "
                    f"got {attn_impl!r}."
                )

        if attn_impl and set_flags:
            raise ValueError(
                f"attn_implementation={attn_impl!r} cannot be combined with legacy "
                f"attention flags ({', '.join(sorted(set_flags))}). The legacy "
                f"flags are deprecated — set only `attn_implementation`."
            )

        if not attn_impl and set_flags:
            # Priority: specific backends beat generic flash/sdp/eager fallbacks.
            for flag in LEGACY_ATTN_FLAG_TO_IMPL:
                if flag in set_flags:
                    canonical = LEGACY_ATTN_FLAG_TO_IMPL[flag]
                    data["attn_implementation"] = canonical
                    LOG.warning(
                        "`%s: true` is deprecated and will be removed in a future "
                        "release. Use `attn_implementation: %s` instead.",
                        flag,
                        canonical,
                    )
                    break

        # Strip legacy flags from validated data — canonical field is authoritative.
        for flag in LEGACY_ATTN_FLAG_TO_IMPL:
            data.pop(flag, None)

        return data

    @field_validator("attn_implementation", mode="before")
    @classmethod
    def validate_attn_implementation(cls, value):
        """Accept canonical names and hub-kernel paths; reject short-form aliases."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"attn_implementation must be a string, got {type(value).__name__}"
            )
        if value in CANONICAL_ATTN_IMPLS:
            return value
        if "/" in value:
            # Hub-kernel path, e.g. "kernels-community/flash-attn3". Pass through.
            return value
        if value in SHORT_FORM_ALIAS_TO_CANONICAL:
            canonical = SHORT_FORM_ALIAS_TO_CANONICAL[value]
            raise ValueError(
                f"attn_implementation={value!r} is not accepted. "
                f"Use the canonical name {canonical!r} instead."
            )
        raise ValueError(
            f"attn_implementation={value!r} is not a recognized backend. "
            f"Expected one of: {sorted(CANONICAL_ATTN_IMPLS)}, or a hub-kernel "
            f"path containing '/'."
        )

    @model_validator(mode="after")
    def check_sageattn_wo_sample_packing(self):
        if (
            self.attn_implementation == "sage"
            and not self.sample_packing
            and not self.pad_to_sequence_len
        ):
            LOG.warning(
                "We recommend turning on `pad_to_sequence_len` for SageAttention "
                "without packing. The loss has been observed to explode otherwise."
            )
        return self

    @model_validator(mode="after")
    def check_sageattn_fft(self):
        if self.attn_implementation == "sage" and not self.adapter:
            LOG.warning(
                "SageAttention full finetuning has been observed to drop loss to 0. "
                "Monitor the loss, or switch to LoRA/QLoRA or another attention method."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_save_strategy_best_requires_metric(cls, data):
        if data.get("save_strategy") == "best" and not data.get(
            "metric_for_best_model"
        ):
            raise ValueError(
                "save_strategy: 'best' requires metric_for_best_model to be set. "
                "Please specify the metric to use, e.g. metric_for_best_model: eval_loss"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_lora_target_modules_regex(cls, data):
        lora_target_modules = data.get("lora_target_modules")
        if not isinstance(lora_target_modules, list):
            return data
        invalid = []
        for pattern in lora_target_modules:
            if not isinstance(pattern, str):
                continue
            try:
                re.compile(pattern)
            except re.error:
                invalid.append(pattern)
        if invalid:
            raise ValueError(
                f"lora_target_modules contains invalid regex pattern(s): {invalid}. "
                "Please provide valid Python regex patterns or plain module name strings."
            )
        return data


class AxolotlConfigWCapabilities(AxolotlInputConfig):
    """Wrapper to valdiate GPU capabilities with the configured options"""

    capabilities: GPUCapabilities
    env_capabilities: EnvCapabilities

    @model_validator(mode="after")
    def check_bf16(self):
        if self.capabilities.bf16:
            if not self.bf16 and not self.bfloat16:
                LOG.info(
                    "bf16 support detected, but not enabled for this configuration."
                )
        else:
            if (
                not self.merge_lora
                and not self.is_preprocess
                and (self.bf16 is True or self.bfloat16 is True)
            ):
                LOG.warning(
                    "bf16 requested, but AMP is not supported on this GPU. Requires Ampere series or above. Training will fail, but other operations (such as merging) are still functional."
                )
        return self

    @model_validator(mode="after")
    def check_tf32(self):
        if self.tf32 == "auto":
            self.tf32 = self.capabilities.tf32
        return self

    @model_validator(mode="after")
    def check_fp8(self):
        if self.fp8 and not self.capabilities.fp8:
            raise ValueError("fp8 requested, but fp8 is not supported on this GPU")
        elif self.fp8 and self.capabilities.fp8 and not is_fp8_available():
            raise ValueError(
                "fp8 requested, but missing one of ms-amp, transformers-engine or torchao."
            )
        return self

    @model_validator(mode="after")
    def check_sample_packing_w_sdpa_bf16(self):
        is_sm_90 = self.capabilities and self.capabilities.compute_capability == "sm_90"
        if (
            self.sample_packing
            and self.attn_implementation == "sdpa"
            and (self.bfloat16 or self.bf16)
            and not is_sm_90
        ):
            # https://github.com/pytorch/pytorch/blob/1b03423526536b5f3d35bdfa95ccc6197556cf9b/test/test_transformers.py#L2440-L2450
            LOG.warning(
                "sample_packing & torch sdpa with bf16 is unsupported may results in 0.0 loss. "
                "This may work on H100s."
            )
        return self

    @model_validator(mode="after")
    def check_compute_capability_w_sageattn(self):
        if (
            self.attn_implementation == "sage"
            and self.capabilities
            and self.capabilities.compute_capability
            not in ["sm_80", "sm_86", "sm_89", "sm_90", "sm_120"]
        ):
            raise ValueError(
                "SageAttention supports compute capability between sm_80 and sm_120. "
                "Please use a different attention implementation."
            )
        return self

    @model_validator(mode="after")
    def check_fp8_attention_preflight(self):
        """fp8 attention requires SM90+ and torch >= 2.11 (torchao >= 0.17 is pinned)."""
        if self.attn_implementation != "fp8":
            return self

        if self.capabilities and self.capabilities.compute_capability:
            cc = self.capabilities.compute_capability
            # Accept sm_90 (H100/H200), sm_100 (B100/B200), sm_120 (B300-class).
            if not cc.startswith("sm_") or int(cc.split("_", 1)[1]) < 90:
                raise ValueError(
                    f"attn_implementation=fp8 requires compute capability sm_90 or "
                    f"higher (Hopper+). Detected {cc!r}."
                )

        torch_version = (
            self.env_capabilities.torch_version if self.env_capabilities else None
        )
        if torch_version is None:
            import torch

            torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
        if version.parse(torch_version) < version.parse("2.11.0"):
            raise ValueError(
                f"attn_implementation=fp8 requires PyTorch >= 2.11.0. "
                f"Detected {torch_version}."
            )

        return self

    @model_validator(mode="before")
    @classmethod
    def check_multigpu_lora_kernels(cls, data):
        if (
            data.get("lora_mlp_kernel")
            or data.get("lora_qkv_kernel")
            or data.get("lora_o_kernel")
            or data.get("lora_embedding_kernel")
        ):
            capabilities = data.get("capabilities")
            is_fsdp = data.get("fsdp_config") is not None
            is_fsdp2 = is_fsdp and str(data.get("fsdp_version")) == "2"

            if capabilities and capabilities.get("n_gpu", 0) > 1 and not is_fsdp2:
                if is_fsdp:
                    raise ValueError(
                        "lora_mlp_kernel, lora_qkv_kernel, and lora_o_kernel are not compatible with FSDP1."
                    )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_quantize_moe_experts(cls, data):
        if data.get("quantize_moe_experts"):
            if data.get("lora_target_linear"):
                raise ValueError(
                    "lora_target_linear is not compatible with quantize_moe_experts. "
                    "Use lora_target_parameters to target expert weights instead."
                )
            if data.get("adapter") not in ("lora", "qlora"):
                raise ValueError("quantize_moe_experts requires adapter: lora or qlora")
            if not (data.get("load_in_4bit") or data.get("load_in_8bit")):
                raise ValueError(
                    "quantize_moe_experts requires load_in_4bit or load_in_8bit"
                )
            if (
                data.get("capabilities")
                and data["capabilities"].get("compute_capability")
                and not data["capabilities"]["compute_capability"].startswith("sm_")
            ):
                raise ValueError(
                    "quantize_moe_experts requires CUDA (not compatible with ROCm or other backends)"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_auto_enable_lora_kernels(cls, data):
        # Only proceed if using LoRA or QLoRA adapter
        if data.get("rl"):
            # RL trainers not tested so don't enable kernels by default
            return data
        if data.get("adapter") in ["lora", "qlora"]:
            # Skip if already set or using 8-bit
            kernel_fields = [
                "lora_mlp_kernel",
                "lora_qkv_kernel",
                "lora_o_kernel",
                "lora_embedding_kernel",
            ]
            if (
                any(data.get(k) is not None for k in kernel_fields)
                or data.get("adapter") == "lora"
                and data.get("load_in_8bit")
            ):
                return data

            # Skip if trust_remote_code is enabled, as lora kernels are not compatible
            if data.get("trust_remote_code"):
                return data

            # Skip auto-enable for MoE models when native grouped_mm is unavailable
            # (torch < 2.9). The grouped_mm fallback in transformers uses torch.mm
            # with out= which bypasses autocast and fails on mixed dtypes during eval.
            env_capabilities = data.get("env_capabilities", {})
            torch_version = env_capabilities.get("torch_version")
            if torch_version is None:
                import torch

                torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
            has_grouped_mm = version.parse(torch_version) >= version.parse("2.9.0")
            if not has_grouped_mm:
                is_moe = False
                model_type = data.get("model_config_type", "")
                if model_type and "moe" in model_type.lower():
                    is_moe = True
                if not is_moe:
                    try:
                        from transformers import AutoConfig

                        base_model = data.get("base_model")
                        if base_model:
                            auto_cfg = AutoConfig.from_pretrained(
                                base_model, trust_remote_code=False
                            )
                            if getattr(auto_cfg, "num_local_experts", None) or getattr(
                                auto_cfg, "num_experts", None
                            ):
                                is_moe = True
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                if is_moe:
                    return data

            # Check multi-GPU compatibility
            capabilities = data.get("capabilities")
            is_multi_gpu = capabilities and capabilities.get("n_gpu", 0) > 1
            is_fsdp = data.get("fsdp_config") is not None
            is_fsdp2 = is_fsdp and str(data.get("fsdp_version")) == "2"

            if (
                not is_multi_gpu
                or (is_multi_gpu and not is_fsdp)
                or (is_multi_gpu and is_fsdp2)
            ):
                # Auto-enable kernels if not explicitly set by user
                if data.get("lora_mlp_kernel") is None:
                    data["lora_mlp_kernel"] = True

                if data.get("lora_qkv_kernel") is None:
                    data["lora_qkv_kernel"] = True

                if data.get("lora_o_kernel") is None:
                    data["lora_o_kernel"] = True

                if data.get("lora_embedding_kernel") is None:
                    data["lora_embedding_kernel"] = True

                LOG.warning(
                    "Auto-enabling LoRA kernel optimizations for faster training. "
                    + "Please explicitly set `lora_*_kernel` config values to `false` to disable. "
                    + "See https://docs.axolotl.ai/docs/lora_optims.html for more info."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_adopt_torch_version(cls, data):
        if (data.get("optimizer") is not None) and ("adopt" in data.get("optimizer")):
            env_capabilities = data.get("env_capabilities", {})
            torch_version = env_capabilities.get("torch_version")

            if torch_version is None:
                import torch

                torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

            if version.parse(torch_version) < version.parse("2.5.1"):
                raise ValueError(
                    "ADOPT optimizer is incompatible with torch version < 2.5.1"
                )
        return data

    @model_validator(mode="after")
    def check_flex_torch_version(self):
        if self.attn_implementation == "flex_attention":
            torch_version = (
                self.env_capabilities.torch_version if self.env_capabilities else None
            )
            if torch_version is None:
                import torch

                torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

            if version.parse(torch_version) < version.parse("2.6.0"):
                raise ValueError(
                    "Flex attention is not supported on torch version < 2.6.0"
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_torch_compile_auto(cls, data):
        if data.get("torch_compile") == "auto":
            env_capabilities = data.get("env_capabilities", {})
            if env_capabilities.get("torch_version"):
                if version.parse(
                    env_capabilities.get("torch_version")
                ) >= version.parse("2.5.1"):
                    LOG.info(
                        "torch.compile is available, setting torch_compile to True"
                    )
                    data["torch_compile"] = True
                else:
                    data["torch_compile"] = False
            else:
                data["torch_compile"] = False
        return data

    @model_validator(mode="before")
    @classmethod
    def check_beta_and_trl_beta_match(cls, data):
        if data.get("beta") and data.get("trl", {}).get("beta"):
            if data["beta"] != data["trl"]["beta"]:
                raise ValueError("beta and trl.beta must match or one must be removed")
        return data

    @model_validator(mode="after")
    def check_min_torch_version(self):
        if self.env_capabilities and self.env_capabilities.torch_version:
            torch_version = self.env_capabilities.torch_version
            if version.parse(torch_version) < version.parse("2.6.0"):
                LOG.warning(
                    f"torch=={torch_version} not be supported. Please upgrade to torch>=2.6.0."
                )

        return self

    @model_validator(mode="before")
    @classmethod
    def check_qat_config(cls, data):
        qat_cfg = data.get("qat", {})
        if not qat_cfg:
            return data

        if data.get("peft"):
            raise ValueError("QAT and PEFT cannot be used together.")

        if data.get("load_in_8bit"):
            raise ValueError("QAT and load_in_8bit cannot be used together.")

        if data.get("load_in_4bit"):
            raise ValueError("QAT and load_in_4bit cannot be used together.")

        env_capabilities = data.get("env_capabilities", {})
        torch_version = env_capabilities.get("torch_version")

        if torch_version is None:
            import torch

            torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

        if version.parse(torch_version) < version.parse("2.6.0"):
            raise ValueError("QAT is not supported on torch version < 2.6.0")

        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_torch_version(cls, data):
        env_capabilities = data.get("env_capabilities", {})
        torch_version = env_capabilities.get("torch_version")

        if torch_version is None:
            import torch

            torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

        if data.get("fsdp_config") and str(data.get("fsdp_version")) == "2":
            if version.parse(torch_version) < version.parse("2.7.0"):
                raise ValueError("FSDP2 is not supported on torch version < 2.7.0")

        return data

    @model_validator(mode="before")
    @classmethod
    def default_dataloader_opts(cls, data):
        if (
            data.get("dataloader_num_workers") is None
            and data.get("dataloader_pin_memory") is None
            and data.get("dataloader_prefetch_factor") is None
        ):
            data["dataloader_num_workers"] = data.get("capabilities").get("n_gpu", 1)
            data["dataloader_pin_memory"] = True
            data["dataloader_prefetch_factor"] = 256

        return data

    @model_validator(mode="before")
    @classmethod
    def default_dataset_num_proc(cls, data):
        if data.get("dataset_processes") is not None:
            if data.get("dataset_num_proc") is None:
                data["dataset_num_proc"] = data["dataset_processes"]
                LOG.warning(
                    "dataset_processes is deprecated and will be removed in a future version. "
                    "Please use dataset_num_proc instead."
                )
            else:
                LOG.warning(
                    "Both dataset_processes and dataset_num_proc are set. "
                    "Using dataset_num_proc and ignoring dataset_processes."
                )
            del data["dataset_processes"]
        elif data.get("dataset_num_proc") is None:
            data["dataset_num_proc"] = get_default_process_count()
        return data

    @model_validator(mode="before")
    @classmethod
    def check_deduplication_with_streaming(cls, data):
        if data.get("dataset_exact_deduplication") and (
            data.get("streaming") or data.get("pretraining_dataset")
        ):
            raise NotImplementedError(
                "dataset_exact_deduplication is not available for streaming datasets. "
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_deduplication_with_skip_prepare(cls, data):
        if data.get("dataset_exact_deduplication") and data.get("skip_prepare_dataset"):
            raise ValueError(
                "dataset_exact_deduplication=True has no effect when "
                "skip_prepare_dataset=True. Deduplication runs as part of the "
                "prepare pipeline, which is skipped. Either set "
                "skip_prepare_dataset: false or disable "
                "dataset_exact_deduplication."
            )
        return data
