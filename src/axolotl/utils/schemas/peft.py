"""Pydantic models for PEFT-related configuration"""

from pydantic import BaseModel, Field, field_validator, model_validator

from axolotl.utils.schemas.quant import QuantizationConfig


class LoftQConfig(BaseModel):
    """LoftQ configuration subset"""

    loftq_bits: int = Field(
        default=4, json_schema_extra={"description": "Quantization bits for LoftQ"}
    )
    # loftq_iter: int = Field(default=1, json_schema_extra={"description": "Alternating iterations for LoftQ"})


class PeftConfig(BaseModel):
    """peftq configuration subset"""

    loftq_config: LoftQConfig | None = None


class LoraConfig(BaseModel):
    """Peft / LoRA configuration subset"""

    quantization: QuantizationConfig | None = None
    load_in_4bit: bool | None = None  # for internal use
    load_in_8bit: bool | None = None  # for internal use
    hqq: bool | None = None  # for internal use
    gptq: bool | None = None  # for internal use

    adapter: str | None = None
    lora_model_dir: str | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_fan_in_fan_out: bool | None = None
    lora_target_modules: str | list[str] | None = None
    lora_target_linear: bool | None = None
    lora_modules_to_save: list[str] | None = None
    lora_dropout: float | None = 0.0
    peft_layers_to_transform: list[int] | None = None
    peft_layers_pattern: list[str] | None = None
    peft: PeftConfig | None = None
    peft_use_dora: bool | None = None
    peft_use_rslora: bool | None = None
    peft_layer_replication: list[tuple[int, int]] | None = None
    peft_init_lora_weights: bool | str | None = None

    qlora_sharded_model_loading: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "load qlora model in sharded format for FSDP using answer.ai technique."
        },
    )
    lora_on_cpu: bool | None = None

    loraplus_lr_ratio: float | None = Field(
        default=None,
        json_schema_extra={
            "description": "loraplus learning rate ratio lr_B / lr_A. Recommended value is 2^4."
        },
    )
    loraplus_lr_embedding: float | None = Field(
        default=1e-6,
        json_schema_extra={
            "description": "loraplus learning rate for lora embedding layers."
        },
    )

    merge_lora: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_adapter(cls, data):
        if (
            not data.get("adapter")
            and not data.get("inference")
            and (data.get("quantization"))
        ):
            raise ValueError(
                "Quantization is not supported without setting an adapter for training."
                "If you want to full finetune, please turn off Quantization."
            )
        return data

    @model_validator(mode="after")
    def validate_qlora(self):
        if self.adapter == "qlora":
            if self.merge_lora:
                # can't merge qlora if loaded in 8bit or 4bit
                if self.quantization:
                    raise ValueError("Can't merge qlora if loaded in quantized model")

                if self.quantization.backend == "gptq":
                    raise ValueError("Can't merge qlora if using gptq")

            else:
                if self.quantization:
                    if self.quantization.bits >= 4:
                        raise ValueError("Can't load qlora in >4 bit")

                    if self.quantization.backend == "gptq":
                        raise ValueError("Can't load qlora if using gptq")

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

    relora_steps: int | None = None
    relora_warmup_steps: int | None = None
    relora_anneal_steps: int | None = None
    relora_prune_ratio: float | None = None
    relora_cpu_offload: bool | None = None
