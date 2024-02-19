"""
Module for pydantic models for configuration
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from transformers import SchedulerType
from transformers.training_args import OptimizerNames

from axolotl.utils.config.models.internals import GPUCapabilities

LOG = logging.getLogger("axolotl.utils.config.models.input")


class DeprecatedParameters(BaseModel):
    """configurations that are deprecated"""

    max_packed_sequence_len: Optional[int] = None

    @field_validator("max_packed_sequence_len")
    @classmethod
    def validate_max_packed_sequence_len(cls, max_packed_sequence_len):
        if max_packed_sequence_len:
            raise DeprecationWarning("`max_packed_sequence_len` is no longer supported")


class PretrainingDataset(BaseModel):
    """pretraining dataset configuration subset"""

    path: Optional[str] = None


class SFTDataset(BaseModel):
    """SFT configuration subset"""

    path: Optional[str] = None
    split: Optional[str] = None
    type: Optional[str] = None
    shards: Optional[int] = None
    conversation: Optional[str] = None
    data_files: Optional[List[str]] = None
    name: Optional[str] = None
    ds_type: Optional[str] = None


class DPODataset(BaseModel):
    """DPO configuration subset"""

    path: Optional[str] = None
    split: Optional[str] = None
    type: Optional[str] = None
    data_files: Optional[List[str]] = None


class RLType(str, Enum):
    """RL trainer type configuration subset"""

    dpo = "dpo"  # pylint: disable=invalid-name
    ipo = "ipo"  # pylint: disable=invalid-name
    kto_pair = "kto_pair"  # pylint: disable=invalid-name


class ChatTemplate(str, Enum):
    """Chat templates configuration subset"""

    chatml = "chatml"  # pylint: disable=invalid-name
    inst = "inst"  # pylint: disable=invalid-name


class LoftQConfig(BaseModel):
    """LoftQ configuration subset"""

    loftq_bits: int = Field(default=4, metadata={"help": "Quantization bits for LoftQ"})
    # loftq_iter: int = Field(default=1, metadata={"help": "Alternating iterations for LoftQ"})


class PeftConfig(BaseModel):
    """peftq configuration subset"""

    loftq_config: Optional[LoftQConfig] = None


class AutoType(str, Enum):
    """auto type string configuration subset - used for bf16"""

    AUTO = "auto"


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
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_fan_in_fan_out: Optional[bool] = None
    lora_target_modules: Optional[List[str]] = None
    lora_target_linear: Optional[bool] = None
    lora_dropout: Optional[float] = None
    peft_layers_to_transform: Optional[List[int]] = None
    peft: Optional[PeftConfig] = None

    gptq: Optional[bool] = None

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


class ModelInputConfig(BaseModel):
    """model to train on configuration subset"""

    base_model: str
    base_model_config: Optional[str] = None
    tokenizer_config: Optional[str] = None
    tokenizer_use_fast: Optional[bool] = None
    tokenizer_legacy: Optional[bool] = None
    tokenizer_type: Optional[str] = Field(
        default=None, metadata={"help": "transformers tokenizer class"}
    )
    model_type: Optional[str] = Field(default=None)
    model_revision: Optional[str] = None
    trust_remote_code: Optional[bool] = None

    model_config: Optional[Dict[str, Any]] = None


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

    learning_rate: Union[str, float]
    weight_decay: Optional[float] = None
    optimizer: Optional[OptimizerNames] = None
    lr_scheduler: Optional[SchedulerType] = None
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


class ModelOutputConfig(BaseModel):
    """model save configuration subset"""

    output_dir: str = Field(default="./model-out")
    hub_model_id: Optional[str] = None
    hub_strategy: Optional[str] = None
    save_safetensors: Optional[bool] = None


class WandbConfig(BaseModel):
    """wandb configuration subset"""

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


class AxolotlInputConfig(
    ModelInputConfig,
    LoraConfig,
    HyperparametersConfig,
    WandbConfig,
    DeprecatedParameters,
    BaseModel,
):
    """wrapper of all config options"""

    strict: Optional[bool] = Field(default=False)
    # resume_from_checkpoint

    rl: Optional[RLType] = None

    datasets: List[Union[SFTDataset, DPODataset]]
    pretraining_dataset: Optional[List[PretrainingDataset]] = Field(
        default=None, metadata={"help": {"streaming dataset to use for pretraining"}}
    )
    dataset_processes: Optional[int] = Field(default=os.cpu_count())
    dataloader_pin_memory: Optional[bool] = None
    dataloader_num_workers: Optional[int] = None
    dataloader_prefetch_factor: Optional[int] = None
    dataloader_drop_last: Optional[bool] = None

    push_dataset_to_hub: Optional[str] = None
    hf_use_auth_token: Optional[bool] = None

    device: Optional[Any] = None
    device_map: Optional[Any] = None
    world_size: Optional[int] = None
    local_rank: Optional[int] = None
    ddp: Optional[bool] = None

    eval_table_size: Optional[int] = None
    eval_table_max_new_tokens: Optional[int] = None

    bf16: Optional[Union[AutoType, bool]] = AutoType.AUTO
    fp16: Optional[bool] = None
    bfloat16: Optional[bool] = None
    float16: Optional[bool] = None
    tf32: Optional[bool] = None
    float32: Optional[bool] = None

    # torch_dtype: Optional[torch.dtype]

    is_falcon_derived_model: Optional[bool] = Field(default=False)
    is_llama_derived_model: Optional[bool] = Field(default=False)
    is_mistral_derived_model: Optional[bool] = Field(default=False)
    is_qwen_derived_model: Optional[bool] = Field(default=False)

    gradient_checkpointing: Optional[bool] = Field(default=False)
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None

    unfrozen_parameters: Optional[List[str]] = None

    is_preprocess: Optional[bool] = None

    sequence_len: int = Field(default=1024)
    sample_packing: Optional[bool] = None
    eval_sample_packing: Optional[bool] = None
    pad_to_sequence_len: Optional[bool] = None

    xformers_attention: Optional[bool] = None
    sdp_attention: Optional[bool] = None
    flash_attention: Optional[bool] = None
    flash_attn_fuse_qkv: Optional[bool] = None
    flash_attn_fuse_mlp: Optional[bool] = None
    flash_optimum: Optional[bool] = None

    deepspeed: Optional[Union[str, Dict[str, Any]]] = None
    fsdp: Optional[List[str]] = None
    fsdp_config: Optional[Dict[str, Any]] = None

    val_set_size: Optional[float] = Field(default=0.0)

    special_tokens: Optional[SpecialTokensConfig] = None
    tokens: Optional[List[str]] = None

    max_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_strategy: Optional[str] = None
    logging_steps: Optional[int] = None
    early_stopping_patience: Optional[int] = None

    # INTERNALS - document for now
    # - total_supervised_tokens

    @field_validator("datasets")
    @classmethod
    def check_non_empty_datasets(cls, datasets):
        if len(datasets) == 0:
            raise ValueError("datasets list cannot be empty")
        return datasets

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
    def check_batch_size_fields(cls, root):
        non_empty_count = sum(
            1
            for field in (
                "micro_batch_size",
                "gradient_accumulation_steps",
                "batch_size",
            )
            if root.get(field)
        )
        if non_empty_count < 2:
            raise ValueError(
                "At least two of [micro_batch_size, gradient_accumulation_steps, batch_size] must be set"
            )
        return root

    @model_validator(mode="before")
    @classmethod
    def check_pretraining_w_max_steps(cls, root):
        if root.get("pretraining_dataset") and not root.get("max_steps"):
            raise ValueError(
                "max_steps must be set when using iterable pretraining_dataset, Trainer can't infer length and schedule optimizer/learning rate without it!"
            )
        return root

    @model_validator(mode="before")
    @classmethod
    def check_sample_packing_w_rl(cls, root):
        if root.get("sample_packing") and root.get("rl"):
            raise ValueError("`sample_packing: true` does not work with RLHF training")
        return root

    @model_validator(mode="before")
    @classmethod
    def hint_sample_packing_padding(cls, root):
        if root.get("sample_packing") and not root.get("pad_to_sequence_len"):
            LOG.warning(
                "`pad_to_sequence_len: true` is recommended when using sample_packing"
            )
        return root

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
            not self.optimizer or "adamw" not in self.optimizer.value
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
        return data

    @model_validator(mode="before")
    @classmethod
    def check_push_save(cls, data):
        if data.get("hub_model_id") and not (
            data.get("save_steps") or data.get("saves_per_epoch")
        ):
            LOG.warning(
                "hub_model_id is set without any models being saved. To save a model, set either save_steps or saves_per_epoch."
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

        return data

    @model_validator(mode="before")
    @classmethod
    def check_eval_packing(cls, data):
        if (
            data.get("sample_packing")
            and data.get("eval_table_size")
            and data.get("eval_sample_packing") is not False
        ):
            raise ValueError(
                "eval_table_size and eval_sample_packing are not supported together with sample_packing. Please set 'eval_sample_packing' to false."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_warmup(cls, data):
        if data.get("warmup_steps") and data.get("warmup_ratio"):
            raise ValueError("warmup_steps and warmup_ratio are mutually exclusive")
        return data

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
