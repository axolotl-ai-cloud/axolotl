import os
from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class PretrainingDataset(BaseModel):
    path: Optional[str]


class SFTDataset(BaseModel):
    path: Optional[str]
    split: Optional[str]
    type: Optional[str]
    shards: Optional[int]
    conversation: Optional[str]
    data_files: List[str]
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
    loftq_bits: int = field(default=4, metadata={"help": "Quantization bits for LoftQ"})
    # loftq_iter: int = field(default=1, metadata={"help": "Alternating iterations for LoftQ"})


class PeftConfig(BaseModel):
    loftq_config: Optional[LoftQConfig]


class AxolotlInputConfig(BaseModel):
    base_model: str
    base_model_config: Optional[str]
    tokenizer_config: Optional[str]
    model_revision: Optional[str]
    trust_remote_code: Optional[bool]
    gptq: Optional[bool]

    rl: Optional[RLType]
    datasets: List[Union[SFTDataset, DPODataset]]
    pretraining_dataset: List[Union[PretrainingDataset]]

    device: Optional[Any]
    device_map: Optional[Any]
    world_size: Optional[int]
    local_rank: Optional[int]
    ddp: Optional[bool]

    gradient_accumulation_steps: Optional[int]
    micro_batch_size: Optional[int]
    batch_size: Optional[int]
    eval_batch_size: Optional[int]

    eval_table_size: Optional[int]
    eval_table_max_new_tokens: Optional[int]

    bf16: Optional[Union[str, bool]]
    fp16: Optional[bool]
    bfloat16: Optional[bool]
    float16: Optional[bool]
    tf32: Optional[bool]
    float32: Optional[bool]

    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)

    # torch_dtype: Optional[torch.dtype]

    dataset_processes: Optional[int] = field(default=os.cpu_count())

    is_falcon_derived_model: Optional[bool] = field(default=False)
    is_llama_derived_model: Optional[bool] = field(default=False)
    is_mistral_derived_model: Optional[bool] = field(default=False)
    is_qwen_derived_model: Optional[bool] = field(default=False)

    model_type: Optional[str] = field(default=None)
    learning_rate: Union[str, float]

    gradient_checkpointing: Optional[bool] = field(default=False)
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]]

    unfrozen_parameters: Optional[List[str]]

    merge_lora: Optional[bool]
    is_preprocess: Optional[bool]

    sample_packing: Optional[bool]
    pad_to_sequence_len: Optional[bool]

    xformers_attention: Optional[bool]
    sdp_attention: Optional[bool]
    flash_attention: Optional[bool]
    flash_attn_fuse_qkv: Optional[bool]
    flash_attn_fuse_mlp: Optional[bool]

    deepspeed: Optional[Union[str, Dict[str, Any]]]
    fsdp: Optional[List[str]]
    fsdp_config: Optional[Dict[str, Any]]

    optimizer: Optional[str]

    val_set_size: Optional[float]
