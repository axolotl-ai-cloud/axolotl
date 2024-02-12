import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator
from transformers import SchedulerType
from transformers.training_args import OptimizerNames

LOG = logging.getLogger("axolotl.utils.config.models.input")


class PretrainingDataset(BaseModel):
    path: Optional[str]


class SFTDataset(BaseModel):
    path: Optional[str]
    split: Optional[str]
    type: Optional[str]
    shards: Optional[int]
    conversation: Optional[str]
    data_files: Optional[List[str]]
    name: Optional[str]
    ds_type: Optional[str]


class DPODataset(BaseModel):
    path: Optional[str]
    split: Optional[str]
    type: Optional[str]
    data_files: List[str]


class RLType(str, Enum):
    dpo = "dpo"
    ipo = "ipo"
    kto_pair = "kto_pair"


class ChatTemplate(str, Enum):
    chatml = "chatml"
    inst = "inst"


class LoftQConfig(BaseModel):
    loftq_bits: int = Field(default=4, metadata={"help": "Quantization bits for LoftQ"})
    # loftq_iter: int = Field(default=1, metadata={"help": "Alternating iterations for LoftQ"})


class PeftConfig(BaseModel):
    loftq_config: Optional[LoftQConfig]


class AutoType(str, Enum):
    AUTO = "auto"


class SpecialTokensConfig(BaseModel):
    bos_token: Optional[str]
    eos_token: Optional[str]
    pad_token: Optional[str]
    unk_token: Optional[str]
    additional_special_tokens: Optional[List[str]]


class LoraConfig(BaseModel):
    load_in_8bit: Optional[bool] = Field(default=False)
    load_in_4bit: Optional[bool] = Field(default=False)

    adapter: Optional[str]
    lora_model_dir: Optional[str]
    lora_rank: Optional[int]
    lora_alpha: Optional[int]
    lora_fan_in_fan_out: Optional[bool]
    lora_target_modules: Optional[List[str]]
    lora_target_linear: Optional[bool]
    lora_dropout: Optional[float]
    peft_layers_to_transform: Optional[List[int]]
    peft: Optional[PeftConfig]


class ModelInputConfig(BaseModel):
    base_model: str
    base_model_config: Optional[str]
    tokenizer_config: Optional[str]
    tokenizer_use_fast: Optional[bool]
    tokenizer_legacy: Optional[bool]
    tokenizer_type: Optional[str] = Field(
        default=None, metadata={"help": "transformers tokenizer class"}
    )
    model_type: Optional[str] = Field(default=None)
    model_revision: Optional[str]
    trust_remote_code: Optional[bool]
    gptq: Optional[bool]

    model_config: Optional[Dict[str, Any]]


class HyperparametersConfig(BaseModel):
    gradient_accumulation_steps: Optional[int] = Field(default=1)
    micro_batch_size: Optional[int] = Field(
        default=1, metadata={"help": "per gpu micro batch size for training"}
    )
    batch_size: Optional[int] = Field(
        metadata={
            "help": "Total batch size, we do not recommended setting this manually"
        }
    )
    eval_batch_size: Optional[int] = Field(
        metadata={
            "help": "per gpu micro batch size for evals, defaults to value of micro_batch_size"
        }
    )

    learning_rate: Union[str, float]
    weight_decay: Optional[float]
    optimizer: Optional[OptimizerNames]
    lr_scheduler: Optional[SchedulerType]
    adam_epsilon: Optional[float]
    adam_beta1: Optional[float]
    adam_beta2: Optional[float]
    max_grad_norm: Optional[float]
    num_epochs: int = Field(default=1)

    @validator(batch_size)
    def hint_batch_size_set(cls, batch_size):
        if batch_size:
            LOG.warning(
                "%s\n%s",
                "batch_size is not recommended. Please use gradient_accumulation_steps instead.",
                "To calculate the equivalent gradient_accumulation_steps, divide batch_size / micro_batch_size / number of gpus.",
            )
        return batch_size


class ModelOutputConfig(BaseModel):
    output_dir: str = Field(default="./model-out")
    hub_model_id: Optional[str]
    hub_strategy: Optional[str]
    save_safetensors: Optional[bool]


class AxolotlInputConfig(ModelInputConfig, LoraConfig, HyperparametersConfig, BaseModel):
    strict: Optional[bool] = Field(default=False)
    # resume_from_checkpoint

    rl: Optional[RLType]

    datasets: List[Union[SFTDataset, DPODataset]]
    pretraining_dataset: Optional[List[PretrainingDataset]] = Field(
        metadata={"help": {"streaming dataset to use for pretraining"}}
    )
    dataset_processes: Optional[int] = Field(default=os.cpu_count())
    dataloader_pin_memory: Optional[bool]
    dataloader_num_workers: Optional[int]
    dataloader_prefetch_factor: Optional[int]
    dataloader_drop_last: Optional[bool]


    device: Optional[Any]
    device_map: Optional[Any]
    world_size: Optional[int]
    local_rank: Optional[int]
    ddp: Optional[bool]

    eval_table_size: Optional[int]
    eval_table_max_new_tokens: Optional[int]

    bf16: Optional[Union[AutoType, bool]]
    fp16: Optional[bool]
    bfloat16: Optional[bool]
    float16: Optional[bool]
    tf32: Optional[bool]
    float32: Optional[bool]

    # torch_dtype: Optional[torch.dtype]

    is_falcon_derived_model: Optional[bool] = Field(default=False)
    is_llama_derived_model: Optional[bool] = Field(default=False)
    is_mistral_derived_model: Optional[bool] = Field(default=False)
    is_qwen_derived_model: Optional[bool] = Field(default=False)

    gradient_checkpointing: Optional[bool] = Field(default=False)
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]]

    unfrozen_parameters: Optional[List[str]]

    merge_lora: Optional[bool]
    is_preprocess: Optional[bool]

    sequence_len: int = Field(default=1024)
    sample_packing: Optional[bool]
    eval_sample_packing: Optional[bool]
    pad_to_sequence_len: Optional[bool]

    xformers_attention: Optional[bool]
    sdp_attention: Optional[bool]
    flash_attention: Optional[bool]
    flash_attn_fuse_qkv: Optional[bool]
    flash_attn_fuse_mlp: Optional[bool]

    deepspeed: Optional[Union[str, Dict[str, Any]]]
    fsdp: Optional[List[str]]
    fsdp_config: Optional[Dict[str, Any]]

    val_set_size: Optional[float] = Field(default=0.0)

    special_tokens: Optional[SpecialTokensConfig]
    tokens: Optional[List[str]]

    max_steps: Optional[int]
    warmup_steps: Optional[int]
    eval_steps: Optional[int]
    save_steps: Optional[int]
    logging_steps: Optional[int]
    early_stopping_patience: Optional[int]

    # INTERNALS - document for now
    # - total_supervised_tokens

    @validator("datasets")
    def check_non_empty_datasets(cls, d):
        if len(d) == 0:
            raise ValueError("datasets list cannot be empty")
        return d

    @root_validator
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

    @root_validator
    def check_pretraining_w_max_steps(cls, root):
        if root.get("pretraining_dataset") and not root.get("max_steps"):
            raise ValueError(
                "max_steps must be set when using iterable pretraining_dataset, Trainer can't infer length and schedule optimizer/learning rate without it!"
            )
        return root

    @root_validator
    def check_sample_packing_w_rl(cls, root):
        if root.get("sample_packing") and root.get("rl"):
            raise ValueError("`sample_packing: true` does not work with RLHF training")
        return root

    @root_validator
    def hint_sample_packing_padding(cls, root):
        if root.get("sample_packing") and not root.get("pad_to_sequence_len"):
            LOG.warning(
                "`pad_to_sequence_len: true` is recommended when using sample_packing"
            )
        return root

    @root_validator
    def check_gas_bsz(cls, root):
        if root.get("gradient_accumulation_steps") and root.get("batch_size"):
            raise ValueError(
                "please set only one of gradient_accumulation_steps or batch_size"
            )
        return root

    @root_validator
    def hint_eval_train_mbsz(cls, root):
        if (
            root.get("eval_batch_size")
            and root.get("micro_batch_size")
            and root.get("eval_batch_size") != root.get("micro_batch_size")
        ):
            LOG.warning(
                "eval_batch_size != micro_batch_size. This can lead to VRAM instability."
            )
