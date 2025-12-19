"""Pydantic models for PEFT-related configuration"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class LoftQConfig(BaseModel):
    """LoftQ configuration subset"""

    loftq_bits: int = Field(
        default=4, json_schema_extra={"description": "typically 4 bits"}
    )
    # loftq_iter: int = Field(default=1, json_schema_extra={"description": "Alternating iterations for LoftQ"})


class PeftConfig(BaseModel):
    """peftq configuration subset"""

    loftq_config: LoftQConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Configuration options for loftq initialization for LoRA"
        },
    )


class LoraConfig(BaseModel):
    """Peft / LoRA configuration subset"""

    load_in_8bit: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "This will attempt to quantize the model down to 8 bits and use adam 8 bit optimizer"
        },
    )
    load_in_4bit: bool | None = Field(
        default=False, json_schema_extra={"description": "Use bitsandbytes 4 bit"}
    )

    adapter: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "If you want to use 'lora' or 'qlora' or leave blank to train all parameters in original model"
        },
    )
    lora_model_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "If you already have a lora model trained that you want to load, put that here. This means after training, if you want to test the model, you should set this to the value of `output_dir`. Note that if you merge an adapter to the base model, a new subdirectory `merged` will be created under the `output_dir`."
        },
    )
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_fan_in_fan_out: bool | None = None
    lora_target_modules: str | list[str] | None = None
    lora_target_parameters: str | list[str] | None = None
    lora_target_linear: bool | None = Field(
        default=None,
        json_schema_extra={"description": "If true, will target all linear modules"},
    )
    lora_modules_to_save: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "If you added new tokens to the tokenizer, you may need to save some LoRA modules because they need to know the new tokens. For LLaMA and Mistral, you need to save `embed_tokens` and `lm_head`. It may vary for other models. `embed_tokens` converts tokens to embeddings, and `lm_head` converts embeddings to token probabilities."
        },
    )
    lora_dropout: float | None = 0.0
    peft_layers_to_transform: list[int] | None = Field(
        default=None,
        json_schema_extra={
            "description": "The layer indices to transform, otherwise, apply to all layers"
        },
    )
    peft_layers_pattern: list[str] | None = None
    peft: PeftConfig | None = None
    peft_use_dora: bool | None = Field(
        default=None, json_schema_extra={"description": "Whether to use DoRA."}
    )
    peft_use_rslora: bool | None = Field(
        default=None, json_schema_extra={"description": "Whether to use RSLoRA."}
    )
    peft_layer_replication: list[tuple[int, int]] | None = Field(
        default=None,
        json_schema_extra={"description": "List of layer indices to replicate."},
    )
    peft_init_lora_weights: bool | str | None = Field(
        default=None,
        json_schema_extra={
            "description": "How to initialize LoRA weights. Default to True which is MS original implementation."
        },
    )
    peft_trainable_token_indices: list[int] | dict[str, list[int]] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "A list of token indices to fine-tune on the `embed_tokens` layer.\n"
                "Otherwise, a dict mapping an embedding layer name to its trainable token indices.\n"
                "See https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/lora#efficiently-train-tokens-alongside-lora"
            )
        },
    )
    peft_ensure_weight_tying: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Whether to tie adapter weights for tied model weights. "
                "See https://github.com/huggingface/peft/issues/2864"
            )
        },
    )
    peft_autocast_adapter_dtype: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to upcast the LoRA adapter to fp32. This is enabled by default in PEFT."
        },
    )

    qlora_sharded_model_loading: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "load qlora model in sharded format for FSDP using answer.ai technique."
        },
    )
    lora_on_cpu: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Do the LoRA/PEFT loading on CPU -- this is required if the base model is so large it takes up most or all of the available GPU VRAM, e.g. during a model and LoRA merge"
        },
    )
    gptq: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether you are training a 4-bit GPTQ quantized model"
        },
    )
    bnb_config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "optional overrides to the bnb 4bit quantization configuration"
        },
    )

    loraplus_lr_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "loraplus learning rate ratio lr_B / lr_A. Recommended value is 2^4."
        },
    )
    loraplus_lr_embedding: float | None = Field(
        default=1e-6,
        json_schema_extra={
            "description": "loraplus learning rate for lora embedding layers. Default value is 1e-6."
        },
    )

    merge_lora: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_adapter(cls, data):
        if (
            not data.get("adapter")
            and not data.get("inference")
            and (data.get("load_in_8bit") or data.get("load_in_4bit"))
        ):
            raise ValueError(
                "load_in_8bit and load_in_4bit are not supported without setting an adapter for training."
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

    @model_validator(mode="before")
    @classmethod
    def validate_lora_dropout(cls, data):
        if data.get("adapter") is not None and data.get("lora_dropout") is None:
            data["lora_dropout"] = 0.0
        return data


class ReLoRAConfig(BaseModel):
    """ReLoRA configuration subset"""

    relora: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use ReLoRA. Use with jagged_restart_*steps options."
        },
    )
    relora_prune_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "threshold for optimizer magnitude when pruning"
        },
    )
    relora_cpu_offload: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "True to perform lora weight merges on cpu during restarts, for modest gpu memory savings"
        },
    )
