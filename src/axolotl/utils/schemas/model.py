"""Pydantic models for model input / output, etc. configuration"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class BnbBaseQuantConfig(BaseModel):
    """bitsandbytes base-weight quantization for LoRA training.

    Replaces the older ``adapter: qlora`` + ``load_in_4bit: true`` boilerplate
    for the common case. ``nf4`` implies 4-bit QLoRA (auto-promotes
    ``adapter: lora`` to ``qlora`` and sets ``load_in_4bit``); ``int8`` stays
    as 8-bit LoRA (sets ``load_in_8bit``).
    """

    weight_dtype: Literal["nf4", "int8"] = Field(
        json_schema_extra={
            "description": "bnb base-weight dtype. nf4 → QLoRA 4-bit; int8 → 8-bit LoRA."
        }
    )


class TorchAoBaseQuantConfig(BaseModel):
    """torchao base-weight quantization for LoRA training.

    Compile- and FSDP2-friendly alternative to bitsandbytes. 4-bit dtypes
    (``int4`` / ``nf4`` / ``nvfp4``) auto-promote the adapter to ``qlora``;
    ``int8`` / ``fp8`` stay as weight-only LoRA. ``mxfp4`` is rejected here
    because torchao has no weight-only flavour for arbitrary linears — for
    MoE experts use ``quantize_moe_experts: true`` instead.
    """

    weight_dtype: Literal["int4", "nf4", "nvfp4", "int8", "fp8", "mxfp4"] = Field(
        json_schema_extra={
            "description": (
                "torchao base-weight dtype. int4/nf4/nvfp4 → QLoRA; int8/fp8 "
                "→ weight-only LoRA; mxfp4 is unsupported as a base-quant "
                "shorthand (use quantize_moe_experts for MoE MXFP4)."
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

    config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={"description": "Forwarded to transformers.Mxfp4Config."},
    )


class FineGrainedFp8BaseQuantConfig(BaseModel):
    """Structured form of ``model_quantization_config: FineGrainedFP8Config``."""

    config_kwargs: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Forwarded to transformers.FineGrainedFP8Config."
        },
    )


class ModelQuantizationConfig(BaseModel):
    """Structured discriminator for the base model's quantization scheme.

    Exactly one of ``bnb`` / ``torchao`` / ``mxfp4`` / ``fp8`` must be set.
    The legacy string form (``model_quantization_config: Mxfp4Config``) keeps
    working via a normalizer in the top-level validator.
    """

    bnb: BnbBaseQuantConfig | None = None
    torchao: TorchAoBaseQuantConfig | None = None
    mxfp4: Mxfp4BaseQuantConfig | None = None
    fp8: FineGrainedFp8BaseQuantConfig | None = None

    @model_validator(mode="after")
    def exactly_one_backend(self):
        chosen = [
            name
            for name, value in (
                ("bnb", self.bnb),
                ("torchao", self.torchao),
                ("mxfp4", self.mxfp4),
                ("fp8", self.fp8),
            )
            if value is not None
        ]
        if len(chosen) != 1:
            raise ValueError(
                "model_quantization_config must select exactly one of "
                "bnb / torchao / mxfp4 / fp8 (got: "
                f"{chosen or 'none'})."
            )
        return self

    @property
    def backend(self) -> str:
        """Name of the selected discriminator (one of bnb/torchao/mxfp4/fp8)."""
        for name in ("bnb", "torchao", "mxfp4", "fp8"):
            if getattr(self, name) is not None:
                return name
        raise RuntimeError("model_quantization_config has no backend set")


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
                "Base-model quantization. Accepts the legacy string form "
                "(`Mxfp4Config` / `FineGrainedFP8Config`) or a structured "
                "form selecting exactly one of `bnb` / `torchao` / `mxfp4` "
                "/ `fp8` (see ModelQuantizationConfig)."
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
