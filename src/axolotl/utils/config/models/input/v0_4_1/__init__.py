"""
Module for pydantic models for configuration
"""

# pylint: disable=too-many-lines

import logging
import os
from enum import Enum
from importlib.metadata import version
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, conlist, field_validator, model_validator
from transformers import SchedulerType
from transformers.training_args import OptimizerNames

from axolotl.utils.config.models.internals import GPUCapabilities

LOG = logging.getLogger("axolotl.utils.config.models.input")

SUPPORTED_METRICS = {"sacrebleu", "comet", "ter", "chrf", "perplexity"}


class DeprecatedParameters(BaseModel):
    """configurations that are deprecated"""

    max_packed_sequence_len: Optional[int] = None
    rope_scaling: Optional[Any] = None
    noisy_embedding_alpha: Optional[float] = None
    dpo_beta: Optional[float] = None

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


class RemappedParameters(BaseModel):
    """parameters that have been remapped to other names"""

    overrides_of_model_config: Optional[Dict[str, Any]] = Field(
        default=None, alias="model_config"
    )
    type_of_model: Optional[str] = Field(default=None, alias="model_type")
    revision_of_model: Optional[str] = Field(default=None, alias="model_revision")


class PretrainingDataset(BaseModel):
    """pretraining dataset configuration subset"""

    name: Optional[str] = None
    path: Optional[str] = None
    split: Optional[str] = "train"
    text_column: Optional[str] = "text"
    type: Optional[str] = "pretrain"
    trust_remote_code: Optional[bool] = False


class UserDefinedPrompterType(BaseModel):
    """structure for user defined prompt types"""

    system_prompt: Optional[str] = None
    system_format: Optional[str] = None
    field_system: Optional[str] = None
    field_instruction: Optional[str] = None
    field_input: Optional[str] = None
    field_output: Optional[str] = None

    format: Optional[str] = None
    no_input_format: Optional[str] = None
    field: Optional[str] = None


class SFTDataset(BaseModel):
    """SFT configuration subset"""

    path: Optional[str] = None
    split: Optional[str] = None
    type: Optional[Union[str, UserDefinedPrompterType]] = None
    shards: Optional[int] = None
    conversation: Optional[str] = None
    chat_template: Optional[str] = None
    data_files: Optional[Union[str, List[str]]] = None
    name: Optional[str] = None
    ds_type: Optional[str] = None
    train_on_split: Optional[str] = None

    field: Optional[str] = None
    field_human: Optional[str] = None
    field_model: Optional[str] = None
    field_messages: Optional[str] = None
    message_field_role: Optional[str] = None
    message_field_content: Optional[str] = None
    message_field_training: Optional[str] = None
    message_field_training_detail: Optional[str] = None
    roles_to_train: Optional[List[str]] = None
    train_on_eos: Optional[str] = None

    roles: Optional[Dict[str, List[str]]] = None
    drop_system_message: Optional[bool] = None

    trust_remote_code: Optional[bool] = False


class UserDefinedDPOType(BaseModel):
    """User defined typing for DPO"""

    field_system: Optional[str] = None
    field_prompt: Optional[str] = None
    field_chosen: Optional[str] = None
    field_rejected: Optional[str] = None
    prompt_format: Optional[str] = None
    chosen_format: Optional[str] = None
    rejected_format: Optional[str] = None


class DPODataset(BaseModel):
    """DPO configuration subset"""

    path: Optional[str] = None
    split: Optional[str] = None
    type: Optional[Union[UserDefinedDPOType, str]] = None
    data_files: Optional[List[str]] = None


class UserDefinedKTOType(BaseModel):
    """User defined typing for KTO"""

    field_system: Optional[str] = None
    field_prompt: Optional[str] = None
    field_completion: Optional[str] = None
    field_label: Optional[bool] = None
    prompt_format: Optional[str] = None
    completion_format: Optional[str] = None


class KTODataset(BaseModel):
    """KTO configuration subset"""

    path: Optional[str] = None
    split: Optional[str] = None
    type: Optional[Union[UserDefinedKTOType, str]] = None
    data_files: Optional[List[str]] = None
    trust_remote_code: Optional[bool] = False


class RLType(str, Enum):
    """RL trainer type configuration subset"""

    dpo = "dpo"  # pylint: disable=invalid-name
    ipo = "ipo"  # pylint: disable=invalid-name
    orpo = "orpo"  # pylint: disable=invalid-name
    kto = "kto"  # pylint: disable=invalid-name
    simpo = "simpo"  # pylint: disable=invalid-name


class ChatTemplate(str, Enum):
    """Chat templates configuration subset"""

    alpaca = "alpaca"  # pylint: disable=invalid-name
    chatml = "chatml"  # pylint: disable=invalid-name
    inst = "inst"  # pylint: disable=invalid-name
    gemma = "gemma"  # pylint: disable=invalid-name
    cohere = "cohere"  # pylint: disable=invalid-name
    llama3 = "llama3"  # pylint: disable=invalid-name
    phi_3 = "phi_3"  # pylint: disable=invalid-name
    phi_35 = "phi_35"  # pylint: disable=invalid-name
    deepseek_v2 = "deepseek_v2"  # pylint: disable=invalid-name
    jamba = "jamba"  # pylint: disable=invalid-name


class LoftQConfig(BaseModel):
    """LoftQ configuration subset"""

    loftq_bits: int = Field(default=4, metadata={"help": "Quantization bits for LoftQ"})
    # loftq_iter: int = Field(default=1, metadata={"help": "Alternating iterations for LoftQ"})


class PeftConfig(BaseModel):
    """peftq configuration subset"""

    loftq_config: Optional[LoftQConfig] = None


class SpecialTokensConfig(BaseModel):
    """Special tokens configuration subset"""

    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    unk_token: Optional[str] = None
    additional_special_tokens: Optional[List[str]] = None


class LoraConfig(BaseModel):
    """Peft / LoRA configuration subset"""

    load_in_8bit: Optional[bool] = Field(default=False)
    load_in_4bit: Optional[bool] = Field(default=False)

    adapter: Optional[str] = None
    lora_model_dir: Optional[str] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_fan_in_fan_out: Optional[bool] = None
    lora_target_modules: Optional[List[str]] = None
    lora_target_linear: Optional[bool] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_dropout: Optional[float] = 0.0
    peft_layers_to_transform: Optional[List[int]] = None
    peft: Optional[PeftConfig] = None
    peft_use_dora: Optional[bool] = None
    peft_use_rslora: Optional[bool] = None
    peft_layer_replication: Optional[List[Tuple[int, int]]] = None

    qlora_sharded_model_loading: Optional[bool] = Field(
        default=False,
        metadata={
            "help": "load qlora model in sharded format for FSDP using answer.ai technique."
        },
    )
    lora_on_cpu: Optional[bool] = None
    gptq: Optional[bool] = None
    bnb_config_kwargs: Optional[Dict[str, Any]] = None

    loraplus_lr_ratio: Optional[float] = Field(
        default=None,
        metadata={
            "help": "loraplus learning rate ratio lr_B / lr_A. Recommended value is 2^4."
        },
    )
    loraplus_lr_embedding: Optional[float] = Field(
        default=1e-6,
        metadata={"help": "loraplus learning rate for lora embedding layers."},
    )

    merge_lora: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def validate_adapter(cls, data):
        if not data.get("adapter") and (
            data.get("load_in_8bit") or data.get("load_in_4bit")
        ):
            raise ValueError(
                "load_in_8bit and load_in_4bit are not supported without setting an adapter."
                "If you want to full finetune, please turn off load_in_8bit and load_in_4bit."
            )
        return data

    @model_validator(mode="after")
    def validate_qlora(self):
        if self.adapter == "qlora":
            if self.merge_lora:
                # can't merge qlora if loaded in 8bit or 4bit
                if self.load_in_8bit:
                    raise ValueError("Can't merge qlora if loaded in 8bit")

                if self.gptq:
                    raise ValueError("Can't merge qlora if gptq")

                if self.load_in_4bit:
                    raise ValueError("Can't merge qlora if loaded in 4bit")

            else:
                if self.load_in_8bit:
                    raise ValueError("Can't load qlora in 8bit")

                if self.gptq:
                    raise ValueError("Can't load qlora if gptq")

                if not self.load_in_4bit:
                    raise ValueError("Require cfg.load_in_4bit to be True for qlora")
        return self

    @field_validator("loraplus_lr_embedding")
    @classmethod
    def convert_loraplus_lr_embedding(cls, loraplus_lr_embedding):
        if loraplus_lr_embedding and isinstance(loraplus_lr_embedding, str):
            loraplus_lr_embedding = float(loraplus_lr_embedding)
        return loraplus_lr_embedding


class ReLoRAConfig(BaseModel):
    """ReLoRA configuration subset"""

    relora_steps: Optional[int] = None
    relora_warmup_steps: Optional[int] = None
    relora_anneal_steps: Optional[int] = None
    relora_prune_ratio: Optional[float] = None
    relora_cpu_offload: Optional[bool] = None


class ModelInputConfig(BaseModel):
    """model to train on configuration subset"""

    base_model: str
    base_model_config: Optional[str] = None
    cls_model_config: Optional[str] = None
    tokenizer_config: Optional[str] = None
    tokenizer_use_fast: Optional[bool] = None
    tokenizer_legacy: Optional[bool] = None
    tokenizer_type: Optional[str] = Field(
        default=None, metadata={"help": "transformers tokenizer class"}
    )
    trust_remote_code: Optional[bool] = None

    model_kwargs: Optional[Dict[str, Any]] = None

    @field_validator("trust_remote_code")
    @classmethod
    def hint_trust_remote_code(cls, trust_remote_code):
        if trust_remote_code:
            LOG.warning(
                "`trust_remote_code` is set to true. Please make sure that you reviewed the remote code/model."
            )
        return trust_remote_code


class HyperparametersConfig(BaseModel):
    """training hyperparams configuration subset"""

    gradient_accumulation_steps: Optional[int] = Field(default=1)
    micro_batch_size: Optional[int] = Field(
        default=1,
        metadata={"help": "per gpu micro batch size for training"},
    )
    batch_size: Optional[int] = Field(
        default=None,
        metadata={
            "help": "Total batch size, we do not recommended setting this manually"
        },
    )
    eval_batch_size: Optional[int] = Field(
        default=None,
        metadata={
            "help": "per gpu micro batch size for evals, defaults to value of micro_batch_size"
        },
    )

    auto_find_batch_size: Optional[bool] = None

    train_on_inputs: Optional[bool] = False
    group_by_length: Optional[bool] = None

    learning_rate: Union[str, float]
    weight_decay: Optional[float] = 0.0
    optimizer: Optional[
        Union[
            OptimizerNames,
            Literal[
                "lion_pytorch",
                "optimi_adamw",
                "ao_adamw_4bit",
                "ao_adamw_8bit",
                "ao_adamw_fp8",
            ],
        ]
    ] = OptimizerNames.ADAMW_HF.value
    optim_args: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, metadata={"help": "Optional arguments to supply to optimizer."}
    )
    optim_target_modules: Optional[Union[List[str], Literal["all_linear"]]] = Field(
        default=None,
        metadata={
            "help": "The target modules to optimize, i.e. the module names that you would like to train."
        },
    )
    torchdistx_path: Optional[str] = None
    lr_scheduler: Optional[Union[SchedulerType, Literal["one_cycle"]]] = "cosine"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None
    lr_quadratic_warmup: Optional[bool] = None
    cosine_min_lr_ratio: Optional[float] = None
    cosine_constant_lr_ratio: Optional[float] = None
    lr_div_factor: Optional[float] = None

    adam_epsilon: Optional[float] = None
    adam_beta1: Optional[float] = None
    adam_beta2: Optional[float] = None
    max_grad_norm: Optional[float] = None
    num_epochs: int = Field(default=1)

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


class ModelOutputConfig(BaseModel):
    """model save configuration subset"""

    output_dir: str = Field(default="./model-out")
    hub_model_id: Optional[str] = None
    hub_strategy: Optional[str] = None
    save_safetensors: Optional[bool] = None


class MLFlowConfig(BaseModel):
    """mlflow configuration subset"""

    use_mlflow: Optional[bool] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    hf_mlflow_log_artifacts: Optional[bool] = None


class LISAConfig(BaseModel):
    """LISA options"""

    lisa_n_layers: Optional[int] = Field(
        default=None,
        metadata={"help": "the number of activate layers in LISA"},
    )
    lisa_step_interval: Optional[int] = Field(
        default=None,
        metadata={"help": "how often to switch layers in LISA"},
    )
    lisa_layers_attribute: Optional[str] = Field(
        default="model.layers",
        metadata={"help": "path under the model to access the layers"},
    )


class WandbConfig(BaseModel):
    """wandb configuration subset"""

    use_wandb: Optional[bool] = None
    wandb_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_watch: Optional[str] = None
    wandb_log_model: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def check_wandb_run(cls, data):
        if data.get("wandb_run_id") and not data.get("wandb_name"):
            data["wandb_name"] = data.get("wandb_run_id")

            LOG.warning(
                "wandb_run_id sets the ID of the run. If you would like to set the name, please use wandb_name instead."
            )

        return data


class CometConfig(BaseModel):
    """Comet configuration subset"""

    use_comet: Optional[bool] = None
    comet_api_key: Optional[str] = None
    comet_workspace: Optional[str] = None
    comet_project_name: Optional[str] = None
    comet_experiment_key: Optional[str] = None
    comet_mode: Optional[str] = None
    comet_online: Optional[bool] = None
    comet_experiment_config: Optional[Dict[str, Any]] = None


class GradioConfig(BaseModel):
    """Gradio configuration subset"""

    gradio_title: Optional[str] = None
    gradio_share: Optional[bool] = None
    gradio_server_name: Optional[str] = None
    gradio_server_port: Optional[int] = None
    gradio_max_new_tokens: Optional[int] = None
    gradio_temperature: Optional[float] = None


# pylint: disable=too-many-public-methods,too-many-ancestors
class AxolotlInputConfig(
    ModelInputConfig,
    ModelOutputConfig,
    LoraConfig,
    ReLoRAConfig,
    HyperparametersConfig,
    WandbConfig,
    MLFlowConfig,
    CometConfig,
    LISAConfig,
    GradioConfig,
    RemappedParameters,
    DeprecatedParameters,
    BaseModel,
):
    """wrapper of all config options"""

    class Config:
        """Config for alias"""

        populate_by_name = True

    strict: Optional[bool] = Field(default=False)
    resume_from_checkpoint: Optional[str] = None
    auto_resume_from_checkpoints: Optional[bool] = None
    resize_token_embeddings_to_32x: Optional[bool] = None

    rl: Optional[RLType] = None

    datasets: Optional[conlist(Union[SFTDataset, DPODataset, KTODataset], min_length=1)] = None  # type: ignore
    test_datasets: Optional[conlist(Union[SFTDataset, DPODataset, KTODataset], min_length=1)] = None  # type: ignore
    shuffle_merged_datasets: Optional[bool] = True
    dataset_prepared_path: Optional[str] = None
    dataset_shard_num: Optional[int] = None
    dataset_shard_idx: Optional[int] = None

    pretraining_dataset: Optional[  # type: ignore
        conlist(Union[PretrainingDataset, SFTDataset], min_length=1)
    ] = Field(
        default=None, metadata={"help": {"streaming dataset to use for pretraining"}}
    )
    dataset_processes: Optional[int] = Field(default=os.cpu_count())
    dataset_keep_in_memory: Optional[bool] = None
    dataloader_pin_memory: Optional[bool] = None
    dataloader_num_workers: Optional[int] = None
    dataloader_prefetch_factor: Optional[int] = None
    dataloader_drop_last: Optional[bool] = None

    accelerator_config: Optional[Dict[str, Any]] = None

    remove_unused_columns: Optional[bool] = None

    push_dataset_to_hub: Optional[str] = None
    hf_use_auth_token: Optional[bool] = None

    device: Optional[Any] = None
    device_map: Optional[Any] = None
    world_size: Optional[int] = None
    local_rank: Optional[int] = None
    ddp: Optional[bool] = None

    seed: Optional[int] = None
    ddp_timeout: Optional[int] = None
    ddp_bucket_cap_mb: Optional[int] = None
    ddp_broadcast_buffers: Optional[bool] = None
    ddp_find_unused_parameters: Optional[bool] = None

    eval_table_size: Optional[int] = None
    eval_max_new_tokens: Optional[int] = None
    do_causal_lm_eval: Optional[bool] = None
    eval_causal_lm_metrics: Optional[List[str]] = None
    do_bench_eval: Optional[bool] = None
    bench_dataset: Optional[str] = None
    bench_split: Optional[str] = None
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None

    loss_watchdog_threshold: Optional[float] = None
    loss_watchdog_patience: Optional[int] = None

    bf16: Optional[Union[Literal["auto"], bool]] = "auto"
    fp16: Optional[bool] = None
    bfloat16: Optional[bool] = None  # for non-AMP cases
    float16: Optional[bool] = None  # for non-AMP cases
    tf32: Optional[bool] = None
    float32: Optional[bool] = None

    # torch_dtype: Optional[torch.dtype]

    gradient_checkpointing: Optional[Union[Literal["unsloth"], bool]] = Field(
        default=False
    )
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None

    unfrozen_parameters: Optional[List[str]] = None

    sequence_len: int = Field(default=512)
    min_sample_len: Optional[int] = None
    max_prompt_len: int = Field(
        default=512, metadata={"help": "maximum prompt length for RL training"}
    )
    sample_packing: Optional[bool] = None
    sample_packing_group_size: Optional[int] = 100_000
    sample_packing_bin_size: Optional[int] = 200
    eval_sample_packing: Optional[bool] = None
    pad_to_sequence_len: Optional[bool] = None
    curriculum_sampling: Optional[bool] = None
    multipack_real_batches: Optional[bool] = None

    # for PoSE context length extension
    use_pose: Optional[bool] = None
    pose_split_on_token_ids: Optional[List[int]] = None
    pose_max_context_len: Optional[int] = None
    pose_num_chunks: Optional[int] = None

    pretrain_multipack_buffer_size: Optional[int] = 10_000
    pretrain_multipack_attn: Optional[bool] = Field(
        default=True,
        metadata={
            "help": "whether to prevent cross attention for packed sequences during pretraining",
        },
    )

    xformers_attention: Optional[bool] = None
    sdp_attention: Optional[bool] = None
    s2_attention: Optional[bool] = None
    flash_attention: Optional[bool] = None
    flash_attn_cross_entropy: Optional[bool] = None
    flash_attn_rms_norm: Optional[bool] = None
    flash_attn_fuse_qkv: Optional[bool] = None
    flash_attn_fuse_mlp: Optional[bool] = None
    flash_optimum: Optional[bool] = None

    eager_attention: Optional[bool] = None

    unsloth_cross_entropy_loss: Optional[bool] = None
    unsloth_lora_mlp: Optional[bool] = None
    unsloth_lora_qkv: Optional[bool] = None
    unsloth_lora_o: Optional[bool] = None
    unsloth_rms_norm: Optional[bool] = None
    unsloth_rope: Optional[bool] = None

    deepspeed: Optional[Union[str, Dict[str, Any]]] = None
    fsdp: Optional[List[str]] = None
    fsdp_config: Optional[Dict[str, Any]] = None
    fsdp_final_state_dict_type: Optional[
        Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"]
    ] = None

    val_set_size: Optional[float] = Field(default=0.0)

    special_tokens: Optional[SpecialTokensConfig] = None
    tokens: Optional[List[str]] = None

    torch_compile: Optional[bool] = None
    torch_compile_backend: Optional[str] = None
    torch_compile_mode: Optional[
        Literal["default", "reduce-overhead", "max-autotune"]
    ] = None

    max_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    eval_steps: Optional[Union[int, float]] = None
    evals_per_epoch: Optional[Union[int]] = None
    evaluation_strategy: Optional[str] = None
    save_steps: Optional[Union[int, float]] = None
    saves_per_epoch: Optional[int] = None
    save_strategy: Optional[str] = None
    save_total_limit: Optional[int] = None
    logging_steps: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    load_best_model_at_end: Optional[bool] = False
    save_only_model: Optional[bool] = False
    use_tensorboard: Optional[bool] = None

    neftune_noise_alpha: Optional[float] = None

    orpo_alpha: Optional[float] = None
    rpo_alpha: Optional[float] = None
    simpo_gamma: Optional[float] = None
    cpo_alpha: Optional[float] = None

    kto_desirable_weight: Optional[float] = None
    kto_undesirable_weight: Optional[float] = None
    rl_beta: Optional[float] = None

    max_memory: Optional[
        Dict[Union[int, Literal["cpu", "disk"]], Union[int, str]]
    ] = None
    gpu_memory_limit: Optional[Union[int, str]] = None
    low_cpu_mem_usage: Optional[bool] = None

    chat_template: Optional[ChatTemplate] = None
    default_system_message: Optional[str] = None

    fix_untrained_tokens: Optional[bool] = None

    # INTERNALS - document for now, generally not set externally
    is_preprocess: Optional[bool] = None

    total_num_tokens: Optional[int] = None
    total_supervised_tokens: Optional[int] = None
    sample_packing_eff_est: Optional[float] = None
    axolotl_config_path: Optional[str] = None

    is_falcon_derived_model: Optional[bool] = Field(default=None)
    is_llama_derived_model: Optional[bool] = Field(default=None)
    is_mistral_derived_model: Optional[bool] = Field(default=None)
    is_qwen_derived_model: Optional[bool] = Field(default=None)

    @field_validator("datasets", mode="before")
    @classmethod
    def fix_sharegpt_datasets(cls, datasets):
        for idx, ds_cfg in enumerate(datasets):
            if not ds_cfg["type"]:
                continue
            if ds_cfg["type"] == "sharegpt:chat":
                LOG.warning(
                    PendingDeprecationWarning(
                        "`type: sharegpt:chat` will soon be deprecated. simply use `type: sharegpt` instead."
                    )
                )
                datasets[idx]["type"] = "sharegpt"
            if "sharegpt_simple" in ds_cfg["type"]:
                LOG.warning(
                    PendingDeprecationWarning(
                        "`type: sharegpt_simple` will soon be deprecated. simply use `type: sharegpt` instead."
                    )
                )
                datasets[idx]["type"] = datasets[idx]["type"].replace(
                    "sharegpt_simple", "sharegpt"
                )
        return datasets

    @model_validator(mode="before")
    @classmethod
    def check_batch_size_fields(cls, data):
        fields = ("micro_batch_size", "gradient_accumulation_steps", "batch_size")
        non_empty_count = sum(1 for field in fields if data.get(field))

        if non_empty_count < 2:
            raise ValueError(f"At least two of {', '.join(fields)} must be set")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_w_max_steps(cls, data):
        if data.get("pretraining_dataset") and not data.get("max_steps"):
            raise ValueError(
                "max_steps must be set when using iterable pretraining_dataset, Trainer can't infer length and schedule optimizer/learning rate without it!"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_w_group_by_length(cls, data):
        if data.get("pretraining_dataset") and data.get("group_by_length"):
            LOG.warning(
                "You probably want to disable group_by_length as it will force a streamed dataset to download completely."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_split_batches_accelerate(cls, data):
        # alternatively set ACCELERATE_SPLIT_BATCHES=False
        if data.get("pretraining_dataset"):
            accelerator_config = data.get("accelerator_config", {})
            if not accelerator_config:
                data["accelerator_config"] = {
                    "split_batches": False,
                    "dispatch_batches": False,
                }
            else:
                if accelerator_config.get("split_batches") is None:
                    data["accelerator_config"]["split_batches"] = False
                if accelerator_config.get("dispatch_batches") is None:
                    data["accelerator_config"]["dispatch_batches"] = False
        return data

    @model_validator(mode="before")
    @classmethod
    def check_gptq_w_revision(cls, data):
        if data.get("gptq") and data.get("revision_of_model"):
            raise ValueError(
                "revision_of_model is not supported for GPTQ models. "
                + "Please download the model from HuggingFace Hub manually for correct branch, "
                + "point to its path, and remove revision_of_model from the config."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_w_xformers(cls, data):
        if data.get("sample_packing") and data.get("xformers_attention"):
            raise ValueError(
                "sample_packing not compatible with xformers_attention. Use flash_attention"
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_wo_flash(cls, data):
        if (
            data.get("sample_packing")
            and not data.get("flash_attention")
            and not data.get("sdp_attention")
        ):
            LOG.warning(
                "sample_packing without flash_attention or sdp_attention does not handle cross-attention."
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_w_rl(cls, data):
        if data.get("sample_packing") and data.get("rl"):
            raise ValueError("`sample_packing: true` does not work with RLHF training")
        return data

    @model_validator(mode="before")
    @classmethod
    def hint_sample_packing_padding(cls, data):
        if data.get("sample_packing") and not data.get("pad_to_sequence_len"):
            LOG.warning(
                "`pad_to_sequence_len: true` is recommended when using sample_packing"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_gas_bsz(cls, data):
        if data.get("gradient_accumulation_steps") and data.get("batch_size"):
            raise ValueError(
                "please set only one of gradient_accumulation_steps or batch_size"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def hint_eval_train_mbsz(cls, data):
        if (
            data.get("eval_batch_size")
            and data.get("micro_batch_size")
            and data.get("eval_batch_size") != data.get("micro_batch_size")
        ):
            LOG.warning(
                "eval_batch_size != micro_batch_size. This can lead to VRAM instability."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_push_ds_auth(cls, data):
        if (
            data.get("push_dataset_to_hub")
            and data.get("hf_use_auth_token") is not True
        ):
            raise ValueError(
                "Require cfg.hf_use_auth_token to be True for push_dataset_to_hub"
            )
        return data

    @model_validator(mode="after")
    def check_falcon_fsdp(self):
        if (self.base_model and "falcon" in self.base_model.lower()) and self.fsdp:
            raise ValueError("FSDP is not supported for falcon models")
        return self

    @model_validator(mode="after")
    def check_mpt_checkpointing(self):
        if (
            self.base_model and "mpt" in self.base_model.lower()
        ) and self.gradient_checkpointing:
            raise ValueError("gradient_checkpointing is not supported for MPT models")
        return self

    @model_validator(mode="after")
    def check_better_transformers(self):
        if self.flash_optimum is True:
            if self.adapter:
                LOG.warning(
                    "BetterTransformers probably doesn't work with PEFT adapters"
                )
            if self.fp16 or self.bf16:
                raise ValueError("AMP is not supported with BetterTransformer")
            if self.float16 is not True and self.bfloat16 is not True:
                LOG.warning(
                    "You should probably set bfloat16 or float16 to true to "
                    "load the model in float16 for BetterTransformers"
                )
        return self

    @model_validator(mode="after")
    def check_adamw_optimizer_params(self):
        if any([self.adam_beta1, self.adam_beta2, self.adam_epsilon]) and (
            not self.optimizer or "adamw" not in str(self.optimizer).lower()
        ):
            LOG.warning("adamw hyperparameters found, but no adamw optimizer set")
        return self

    @model_validator(mode="before")
    @classmethod
    def check_saves(cls, data):
        if (
            data.get("save_strategy")
            and data.get("save_steps")
            and data.get("save_strategy") != "steps"
        ):
            raise ValueError(
                "save_strategy and save_steps mismatch. Please set save_strategy to 'steps' or remove save_steps."
            )
        if data.get("saves_per_epoch") and data.get("save_steps"):
            raise ValueError(
                "save_steps and saves_per_epoch are mutually exclusive and cannot be used together."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_push_save(cls, data):
        if data.get("hub_model_id") and (
            data.get("save_strategy") not in ["steps", "epoch", None]
        ):
            LOG.warning(
                "hub_model_id is set without any models being saved. To save a model, set save_strategy."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_evals(cls, data):
        if (
            data.get("evaluation_strategy")
            and data.get("eval_steps")
            and data.get("evaluation_strategy") != "steps"
        ):
            raise ValueError(
                "evaluation_strategy and eval_steps mismatch. Please set evaluation_strategy to 'steps' or remove eval_steps."
            )

        if (
            data.get("val_set_size") == 0
            and (data.get("eval_steps") or data.get("evaluation_strategy"))
            and not data.get("test_datasets")
        ):
            raise ValueError(
                "eval_steps and evaluation_strategy are not supported with val_set_size == 0"
            )
        if data.get("evals_per_epoch") and data.get("eval_steps"):
            raise ValueError(
                "eval_steps and evals_per_epoch are mutually exclusive and cannot be used together."
            )
        if (
            data.get("evals_per_epoch")
            and data.get("evaluation_strategy")
            and data.get("evaluation_strategy") != "steps"
        ):
            raise ValueError(
                "evaluation_strategy must be empty or set to `steps` when used with evals_per_epoch."
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_eval_packing(cls, data):
        # TODO also should check test_datasets and val_set_size as we can skip
        # if there are no eval datasets/splits
        if (
            data.get("sample_packing")
            and data.get("eval_table_size")
            and data.get("eval_sample_packing") is not False
        ):
            raise ValueError(
                "eval_table_size and eval_sample_packing are not supported together with sample_packing. Please set 'eval_sample_packing' to false."
            )
        if (
            data.get("sample_packing")
            and data.get("eval_sample_packing") is None
            and not data.get("eval_table_size")
        ):
            LOG.info(
                "explicitly setting `eval_sample_packing` to match `sample_packing`"
            )
            data["eval_sample_packing"] = True

        if (
            data.get("sample_packing")
            and data.get("eval_sample_packing") is False
            and data.get("remove_unused_columns") is None
        ):
            LOG.info(
                "setting `remove_unused_columns: false` for when sample_packing and eval_sample_packing don't match"
            )
            data["remove_unused_columns"] = False

        return data

    @model_validator(mode="before")
    @classmethod
    def check_warmup(cls, data):
        if data.get("warmup_steps") and data.get("warmup_ratio"):
            raise ValueError("warmup_steps and warmup_ratio are mutually exclusive")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_neftune(cls, data):
        if data.get("noisy_embedding_alpha") and not data.get("neftune_noise_alpha"):
            data["neftune_noise_alpha"] = data["noisy_embedding_alpha"]
            del data["noisy_embedding_alpha"]
        elif data.get("noisy_embedding_alpha") and not data.get("neftune_noise_alpha"):
            raise ValueError(
                "noisy_embedding_alpha is deprecated, use neftune_noise_alpha; both are set, please remove the deprecated noisy_embedding_alpha setting"
            )
        return data

    @field_validator("neftune_noise_alpha")
    @classmethod
    def validate_neftune_noise_alpha(cls, neftune_noise_alpha):
        if neftune_noise_alpha is not None and neftune_noise_alpha <= 0.0:
            raise ValueError("neftune_noise_alpha must be > 0.0")
        return neftune_noise_alpha

    @model_validator(mode="after")
    def check_rl_beta(self):
        if self.dpo_beta and not self.rl_beta:
            self.rl_beta = self.dpo_beta
            del self.dpo_beta
        return self

    @model_validator(mode="after")
    def check_simpo_warmup(self):
        if self.rl == "simpo" and self.warmup_ratio:
            raise ValueError(
                "warmup_ratio is not supported with the simpo trainer. Please use `warmup_steps` instead"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_frozen(cls, data):
        if (
            data.get("adapter")
            and data.get("peft_layers_to_transform")
            and data.get("unfrozen_parameters")
        ):
            raise ValueError(
                "`unfrozen_parameters` used with `peft_layers_to_transform` can have unexpected behavior."
            )

        return data

    @model_validator(mode="after")
    def check_fft_possible_bad_config(self):
        if (
            # pylint: disable=too-many-boolean-expressions
            not (self.bf16 or self.bfloat16)
            and (self.fp16 or self.float16)
            and not self.adapter
            and not self.flash_attention
            and self.sample_packing
        ):
            LOG.warning(
                "Full fine tune w/o FA2 w/ sample packing and fp16/float16 is likely to raise errors. Try LoRA."
            )
            # ValueError: Attempting to unscale FP16 gradients.
            # OR
            # RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
        return self

    @model_validator(mode="after")
    def check_fused_lora(self):
        if self.adapter in ["lora", "qlora"] and (
            self.flash_attn_fuse_qkv or self.flash_attn_fuse_mlp
        ):
            raise ValueError("Fused modules are not supported with LoRA/QLoRA")
        return self

    @model_validator(mode="after")
    def hint_lora_8bit(self):
        loftq = (
            self.peft and self.peft.loftq_config and self.peft.loftq_config.loftq_bits
        )
        if not self.load_in_8bit and self.adapter == "lora" and not loftq:
            LOG.warning("We recommend setting `load_in_8bit: true` for LORA finetuning")
        return self

    @model_validator(mode="after")
    def check_early_stopping(self):
        if self.early_stopping_patience:
            if not self.save_steps or not self.eval_steps:
                raise ValueError(
                    "`early_stopping_patience` requires save_steps and eval_steps to be set. eval_steps should evenly divide save_steps."
                )
            if self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "`early_stopping_patience` requires that eval_steps should evenly divide save_steps."
                )
        return self

    @model_validator(mode="after")
    def check_relora(self):
        if self.relora_steps:
            if self.adapter not in ("lora", "qlora"):
                raise ValueError("cfg.adapter must be lora or qlora to use ReLoRA")

            if self.fsdp:
                raise ValueError("fsdp not supported with ReLoRA")

            if self.deepspeed:
                raise ValueError("deepspeed not supported with ReLoRA")

            if self.lr_scheduler == "one_cycle":
                raise ValueError(
                    "ReLoRA is not compatible with the one_cycle scheduler"
                )

            if self.flash_attn_fuse_qkv or self.flash_attn_fuse_mlp:
                raise ValueError("Fused modules are not supported with ReLoRA")
        return self

    @model_validator(mode="before")
    @classmethod
    def check_mem_mismatch(cls, data):
        if (
            data.get("max_memory") is not None
            and data.get("gpu_memory_limit") is not None
        ):
            raise ValueError(
                "max_memory and gpu_memory_limit are mutually exclusive and cannot be used together."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_use_reentrant_mismatch(cls, data):
        if (
            data.get("unfrozen_parameters")
            and data.get("gradient_checkpointing_kwargs")
            and data.get("gradient_checkpointing_kwargs", {}).get("use_reentrant")
            is True
        ):
            # https://github.com/huggingface/transformers/issues/21381
            raise ValueError(
                "`use_reentrant` must be false when used with partially frozen model."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_val_w_test_datasets(cls, data):
        if data.get("test_datasets") and data.get("val_set_size"):
            raise ValueError(
                "non-zero val_set_size should not be used with test_datasets configuration"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_offload_w_8bit_optimizer(cls, data):
        if (
            data.get("fsdp")
            and "8bit" in data.get("optimizer", "")
            and data.get("fsdp_config")
            and data["fsdp_config"].get("fsdp_offload_params")
        ):
            raise ValueError(
                f"FSDP Offload not compatible with {data.get('optimizer')}"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_sharded_state_dict_w_safetensors(cls, data):
        if (
            data.get("fsdp")
            and data.get("save_safetensors")
            and data.get("fsdp_config")
            and data["fsdp_config"].get("fsdp_state_dict_type") == "SHARDED_STATE_DICT"
        ):
            raise ValueError(
                "FSDP SHARDED_STATE_DICT not compatible with save_safetensors"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_causal_lm_evals(cls, data):
        if data.get("do_causal_lm_eval") and data.get("eval_sample_packing"):
            raise ValueError(
                "do_causal_lm_eval is enabled, eval_sample_packing must be set to False"
            )

        if data.get("eval_causal_lm_metrics"):
            if not isinstance(data.get("eval_causal_lm_metrics"), list):
                raise ValueError("eval_causal_lm_metrics must be a list")
            # only ["sacrebleu", "comet", "ter", "chrf"] supported
            if set(data.get("eval_causal_lm_metrics")) - SUPPORTED_METRICS:
                raise ValueError(
                    f"eval_causal_lm_metrics must be one of {SUPPORTED_METRICS}"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_dataset_or_pretraining_dataset(cls, data):
        if data.get("datasets") is None and data.get("pretraining_dataset") is None:
            raise ValueError("either datasets or pretraining_dataset is required")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_xentropy_patch_conflicts(cls, data):
        if data.get("flash_attn_cross_entropy") and data.get(
            "unsloth_cross_entropy_loss"
        ):
            raise ValueError(
                "flash_attn_cross_entropy and unsloth_cross_entropy_loss cannot be both enabled"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_qlora_unsloth(cls, data):
        if (
            data.get("unsloth_lora_mlp")
            or data.get("unsloth_lora_qkv")
            or data.get("unsloth_lora_o")
        ):
            if data.get("adapter") == "lora" or data.get("load_in_8bit"):
                raise ValueError(
                    "unsloth_lora_mlp, unsloth_lora_qkv, and unsloth_lora_o are not compatible with 8-bit LoRA"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_unsloth_xformers_version(cls, data):
        if (
            data.get("unsloth_lora_mlp")
            or data.get("unsloth_lora_qkv")
            or data.get("unsloth_lora_o")
        ):
            xformers_version = version("xformers")
            if xformers_version == "0.0.27":
                raise ValueError(
                    "xformers version 0.0.27 is not supported with unsloth. Please downgrade to 0.0.26.post1"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_torch_compile_deepspeed(cls, data):
        if data.get("deepspeed") and data.get("torch_compile"):
            raise ValueError(
                "torch_compile should be set within your deepspeed config file"
            )
        return data


class AxolotlConfigWCapabilities(AxolotlInputConfig):
    """wrapper to valdiate gpu capabilities with the configured options"""

    capabilities: GPUCapabilities

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
                raise ValueError(
                    "bf16 requested, but AMP is not supported on this GPU. Requires Ampere series or above."
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_w_sdpa_bf16(cls, data):
        is_sm_90: bool = (
            data["capabilities"]
            and data["capabilities"].get("compute_capability") == "sm_90"
        )
        if (
            data.get("sample_packing")
            and data.get("sdp_attention")
            and (data.get("bfloat16") or data.get("bf16"))
            and not is_sm_90
        ):
            # https://github.com/pytorch/pytorch/blob/1b03423526536b5f3d35bdfa95ccc6197556cf9b/test/test_transformers.py#L2440-L2450
            LOG.warning(
                "sample_packing & torch sdpa with bf16 is unsupported may results in 0.0 loss. "
                "This may work on H100s."
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_hopper_8bit_lora(cls, data):
        is_sm_90: bool = (
            data["capabilities"]
            and data["capabilities"].get("compute_capability") == "sm_90"
        )
        if data.get("adapter") and data.get("load_in_8bit") and is_sm_90:
            # see https://github.com/bitsandbytes-foundation/bitsandbytes/issues/538#issuecomment-2262945464
            raise ValueError("8-bit LoRA is not supported on Hopper GPUs")

        return data

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_deepspeed(cls, data):
        if data.get("deepspeed") and data.get("fsdp"):
            raise ValueError("deepspeed and fsdp cannot be used together.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_multigpu_unsloth(cls, data):
        if (
            data.get("unsloth_lora_mlp")
            or data.get("unsloth_lora_qkv")
            or data.get("unsloth_lora_o")
        ):
            capabilities = data.get("capabilities")
            if capabilities and capabilities.get("n_gpu", 0) > 1:
                raise ValueError(
                    "unsloth_lora_mlp, unsloth_lora_qkv, and unsloth_lora_o are not compatible with multi-GPU training."
                )
        return data
