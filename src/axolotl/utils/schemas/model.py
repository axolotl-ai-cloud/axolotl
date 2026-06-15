"""Pydantic models for model input / output, etc. configuration"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class BnbBaseQuantConfig(BaseModel):
    """bitsandbytes base-weight quantization for LoRA training.

    Replaces the older ``adapter: qlora`` + ``load_in_4bit: true`` boilerplate.
    ``nf4`` is 4-bit QLoRA (mirrored into ``load_in_4bit`` for the loader);
    ``int8`` is 8-bit LoRA (mirrored into ``load_in_8bit``).
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["bnb"]
    weight_dtype: Literal["nf4", "int8"] = Field(
        json_schema_extra={
            "description": "bnb base-weight dtype. nf4 → QLoRA 4-bit; int8 → 8-bit LoRA."
        }
    )


class TorchAoBaseQuantConfig(BaseModel):
    """torchao base-weight quantization for LoRA training.

    Compile- and FSDP2-friendly alternative to bitsandbytes. ``int4`` /
    ``nf4`` / ``nvfp4`` are 4-bit QLoRA-style base quants; ``int8`` / ``fp8``
    are weight-only LoRA. MXFP4 has no weight-only torchao config for
    arbitrary linears — for MoE experts use ``quantize_moe_experts: true``.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["torchao"]
    weight_dtype: Literal["int4", "nf4", "nvfp4", "int8", "fp8"] = Field(
        json_schema_extra={
            "description": (
                "torchao base-weight dtype. int4/nf4/nvfp4 → QLoRA; int8/fp8 "
                "→ weight-only LoRA."
            )
        }
    )
    group_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Quant group size. Defaults: int4/int8 → 128, nf4 → 64, "
                "nvfp4 → 16. Ignored for fp8."
            )
        },
    )


class Mxfp4BaseQuantConfig(BaseModel):
    """Structured form of ``model_quantization_config: Mxfp4Config``.

    Pass-through ``config_kwargs`` go straight to ``transformers.Mxfp4Config``.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["mxfp4"]
    config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={"description": "Forwarded to transformers.Mxfp4Config."},
    )


class FineGrainedFp8BaseQuantConfig(BaseModel):
    """Structured form of ``model_quantization_config: FineGrainedFP8Config``."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["fp8"]
    config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Forwarded to transformers.FineGrainedFP8Config."
        },
    )


# Discriminated on `backend` so exactly-one-backend and per-backend dtypes are
# structurally guaranteed, with precise pydantic errors for invalid combos.
ModelQuantizationConfig = Annotated[
    BnbBaseQuantConfig
    | TorchAoBaseQuantConfig
    | Mxfp4BaseQuantConfig
    | FineGrainedFp8BaseQuantConfig,
    Field(discriminator="backend"),
]

LEGACY_MQC_STRING_TO_BACKEND = {"Mxfp4Config": "mxfp4", "FineGrainedFP8Config": "fp8"}


def mqc_as_dict(mqc: Any, backend: str) -> dict | None:
    """Return model_quantization_config as a dict when it selects the given
    backend, tolerating both Pydantic-model and model_dump'd-dict forms.

    ``validate_config`` runs ``model_dump(exclude_none=True)`` and wraps the
    result in a ``DictDefault``, so downstream code sees dicts; tests and
    library users may pass Pydantic instances directly. Probe both.
    """
    if mqc is None or isinstance(mqc, str):
        return None
    if isinstance(mqc, dict):
        return mqc if mqc.get("backend") == backend else None
    if getattr(mqc, "backend", None) == backend:
        return mqc.model_dump(exclude_none=True)
    return None


def implies_bnb_4bit(data: dict) -> bool:
    """Whether raw config data requests a bnb 4-bit base, in any spelling."""
    bnb = mqc_as_dict(data.get("model_quantization_config"), "bnb") or {}
    return bool(
        data.get("load_in_4bit")
        or data.get("adapter") == "qlora"
        or bnb.get("weight_dtype") == "nf4"
    )


def implies_bnb_8bit(data: dict) -> bool:
    """Whether raw config data requests a bnb 8-bit base, in any spelling."""
    bnb = mqc_as_dict(data.get("model_quantization_config"), "bnb") or {}
    return bool(data.get("load_in_8bit") or bnb.get("weight_dtype") == "int8")


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
    processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "kwargs forwarded to the processor's from_pretrained(), overriding processor config (e.g. image_seq_length, min_pixels, etc.)."
        },
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

    model_quantization_config: (
        Literal["Mxfp4Config", "FineGrainedFP8Config"] | ModelQuantizationConfig | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Base-model quantization. Structured form discriminated on "
                "`backend` (`bnb` / `torchao` / `mxfp4` / `fp8`), e.g. "
                "`{backend: torchao, weight_dtype: int4}`. The legacy string "
                "form (`Mxfp4Config` / `FineGrainedFP8Config`) is deprecated."
            )
        },
    )
    model_quantization_config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "kwargs forwarded to the model quantization config (only "
                "with the legacy string form of model_quantization_config; "
                "the structured form carries kwargs inline)."
            )
        },
    )
    use_onebitllms: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to use `onebitllms` for 1.58bit training (only for bitnet models)."
        },
    )

    @field_validator("trust_remote_code")
    @classmethod
    def hint_trust_remote_code(cls, trust_remote_code):
        if trust_remote_code:
            LOG.warning(
                "`trust_remote_code` is set to true. Please make sure that you reviewed the remote code/model."
            )
        return trust_remote_code

    @field_validator("processor_kwargs")
    @classmethod
    def reject_reserved_processor_kwargs(cls, processor_kwargs):
        if not processor_kwargs:
            return processor_kwargs
        reserved = {"revision", "trust_remote_code"}
        conflicts = reserved.intersection(processor_kwargs)
        if conflicts:
            raise ValueError(
                "Do not set reserved keys "
                f"{sorted(conflicts)} inside `processor_kwargs`; "
                "use the top-level `revision_of_model` / `trust_remote_code` "
                "config keys instead."
            )
        return processor_kwargs


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
    hub_revision: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "branch/revision to push to on hub (default: main)"
        },
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
