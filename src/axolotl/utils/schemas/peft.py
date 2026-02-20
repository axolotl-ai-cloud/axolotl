"""Pydantic models for PEFT-related configuration"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from axolotl.utils.schemas.enums import TorchAOQuantDType
from axolotl.utils.schemas.quantization import validate_ao_dtype


class LoftQConfig(BaseModel):
    """LoftQ configuration subset"""

    loftq_bits: int = Field(
        default=4, json_schema_extra={"description": "typically 4 bits"}
    )
    # loftq_iter: int = Field(default=1, json_schema_extra={"description": "Alternating iterations for LoftQ"})


class PeftConfig(BaseModel):
    """PEFT configuration subset"""

    loftq_config: LoftQConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Configuration options for loftq initialization for LoRA"
        },
    )
    backend: Literal["bnb", "torchao"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Quantization backend for QLoRA. 'bnb' for bitsandbytes (default), 'torchao' for torchao."
        },
    )
    weight_dtype: TorchAOQuantDType | None = Field(
        default=None,
        json_schema_extra={
            "description": "Weight quantization dtype (int4, int8, or nf4). Also used with bnb backend to auto-configure quantization."
        },
    )
    group_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Group size for quantization. Defaults to 128 for int4, 64 for nf4."
        },
    )

    @field_validator("weight_dtype", mode="before")
    @classmethod
    def validate_weight_dtype(cls, v):
        return validate_ao_dtype(v)


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
    def auto_detect_qlora(cls, data):
        """Auto-set adapter type and quantization flags from peft config.

        When peft.backend and peft.weight_dtype are set, this infers the correct
        adapter type and internal flags (load_in_4bit, load_in_8bit) so users
        don't need to set them manually.
        """
        peft = data.get("peft")
        if not isinstance(peft, dict):
            return data

        backend = peft.get("backend")
        weight_dtype = peft.get("weight_dtype")

        # Validate: weight_dtype requires backend
        if weight_dtype and not backend:
            raise ValueError(
                "peft.backend is required when peft.weight_dtype is set. "
                "Use 'torchao' or 'bnb'."
            )

        if not weight_dtype:
            return data

        adapter = data.get("adapter")

        if backend == "torchao":
            # torchao: any quantized weight_dtype means qlora
            if adapter == "lora":
                data["adapter"] = "qlora"

        elif backend == "bnb":
            if weight_dtype == "nf4":
                # bnb nf4 = qlora with load_in_4bit
                if adapter == "lora":
                    data["adapter"] = "qlora"
                data.setdefault("load_in_4bit", True)
            elif weight_dtype == "int8":
                # bnb int8 = lora with load_in_8bit
                data.setdefault("load_in_8bit", True)
            else:
                raise ValueError(
                    f"peft.weight_dtype '{weight_dtype}' is not supported with bnb backend. "
                    "Supported: nf4, int8."
                )

        return data

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
            is_torchao = self.peft and self.peft.backend == "torchao"

            if self.merge_lora:
                # can't merge qlora if loaded in 8bit or 4bit
                if self.load_in_8bit:
                    raise ValueError("Can't merge qlora if loaded in 8bit")

                if self.gptq:
                    raise ValueError("Can't merge qlora if gptq")

                if self.load_in_4bit:
                    raise ValueError("Can't merge qlora if loaded in 4bit")

            elif is_torchao:
                # torchao backend: validate torchao-specific requirements
                if self.load_in_4bit or self.load_in_8bit:
                    raise ValueError(
                        "load_in_4bit/load_in_8bit are for bitsandbytes. "
                        "With peft.backend: torchao, quantization is handled by torchao."
                    )
                if not self.peft.weight_dtype:
                    raise ValueError(
                        "peft.weight_dtype is required when peft.backend is 'torchao'"
                    )

            else:
                # Default bnb path
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
