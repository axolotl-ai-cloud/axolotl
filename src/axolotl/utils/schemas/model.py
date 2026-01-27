"""Pydantic models for model input / output, etc. configuration"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ModelInputConfig(BaseModel):
    """Model configuration subset"""

    model_config = {"protected_namespaces": ()}

    base_model: str = Field(
        json_schema_extra={
            "description": "This is the huggingface model that contains *.pt, *.safetensors, or *.bin files. This can also be a relative path to a model on disk"
        }
    )
    base_model_config: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "If the base_model repo on hf hub doesn't include configuration .json files, You can set that here, or leave this empty to default to base_model"
        },
    )
    cls_model_config: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "transformers config class (e.g., 'LlamaConfig', 'MistralConfig'). Defaults to AutoConfig."
        },
    )
    tokenizer_config: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Optional tokenizer configuration path in case you want to use a different tokenizer than the one defined in the base model"
        },
    )
    tokenizer_use_fast: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "use_fast option for tokenizer loading from_pretrained, default to True"
        },
    )
    tokenizer_legacy: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use the legacy tokenizer setting, defaults to True"
        },
    )
    tokenizer_use_mistral_common: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use mistral-common tokenizer. If set to True, it will use the mistral-common tokenizer."
        },
    )
    tokenizer_type: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Corresponding tokenizer for the model AutoTokenizer is a good choice"
        },
    )
    processor_type: str | None = Field(
        default=None, json_schema_extra={"description": "transformers processor class"}
    )
    tokenizer_save_jinja_files: bool | None = Field(
        default=True,  # match the default behavior from transformers
        json_schema_extra={
            "description": "Whether to save jinja files for tokenizer, transformers default is True"
        },
    )
    trust_remote_code: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Trust remote code for untrusted source"},
    )

    experimental_skip_move_to_device: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Don't move the model to the device before sharding. Set to `false` to revert to legacy behavior."
        },
    )

    use_kernels: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Use custom kernels, e.g. MegaBlocks."},
    )

    model_quantization_config: Literal["Mxfp4Config"] | None = Field(
        default=None,
        json_schema_extra={"description": "Model loading quantization config"},
    )
    model_quantization_config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={"description": "kwargs for model quantization config"},
    )

    @field_validator("trust_remote_code")
    @classmethod
    def hint_trust_remote_code(cls, trust_remote_code):
        if trust_remote_code:
            LOG.warning(
                "`trust_remote_code` is set to true. Please make sure that you reviewed the remote code/model."
            )
        return trust_remote_code


class ModelOutputConfig(BaseModel):
    """model save configuration subset"""

    output_dir: str = Field(
        default="./model-out",
        json_schema_extra={"description": "Where to save the full-finetuned model to"},
    )
    hub_model_id: str | None = Field(
        default=None, json_schema_extra={"description": "push checkpoints to hub"}
    )
    hub_strategy: str | None = Field(
        default=None,
        json_schema_extra={"description": "how to push checkpoints to hub"},
    )
    save_safetensors: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to save the model using safetensors format. Defaults to True."
        },
    )

    @field_validator("save_safetensors")
    @classmethod
    def validate_save_safetensors(cls, v):
        if v is False:
            raise ValueError(
                "save_safetensors=False is not supported in Transformers V5. "
                "Transformers V5 always uses safetensors format for model serialization. "
                "This field is deprecated and will be removed in a future version."
            )
        # Allow None and True, will default to True if None
        return True if v is None else v


class SpecialTokensConfig(BaseModel):
    """Special tokens configuration subset"""

    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    unk_token: str | None = None
    additional_special_tokens: list[str] | None = None
