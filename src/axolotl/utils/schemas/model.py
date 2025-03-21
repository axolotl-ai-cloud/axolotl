"""Pydantic models for model input / output, etc. configuration"""

import logging

from pydantic import BaseModel, Field, field_validator

LOG = logging.getLogger(__name__)


class ModelInputConfig(BaseModel):
    """Model configuration subset"""

    model_config = {"protected_namespaces": ()}

    base_model: str
    base_model_config: str | None = None
    cls_model_config: str | None = None
    tokenizer_config: str | None = None
    tokenizer_use_fast: bool | None = None
    tokenizer_legacy: bool | None = None
    tokenizer_type: str | None = Field(
        default=None, json_schema_extra={"description": "transformers tokenizer class"}
    )
    processor_type: str | None = Field(
        default=None, json_schema_extra={"description": "transformers processor class"}
    )
    trust_remote_code: bool | None = None

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

    output_dir: str = Field(default="./model-out")
    hub_model_id: str | None = None
    hub_strategy: str | None = None
    save_safetensors: bool | None = True


class SpecialTokensConfig(BaseModel):
    """Special tokens configuration subset"""

    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    unk_token: str | None = None
    additional_special_tokens: list[str] | None = None
