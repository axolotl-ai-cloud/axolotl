"""
QAT Config Schema
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from axolotl.utils.schemas.enums import TorchAOQuantDType


def validate_ao_dtype(v: Any) -> TorchAOQuantDType | None:
    if v is None:
        return None
    if v == "int4":
        return TorchAOQuantDType.int4
    if v == "int8":
        return TorchAOQuantDType.int8
    if v in ["float8_e4m3fn", "fp8", "float8"]:
        return TorchAOQuantDType.float8_e4m3fn
    if v == "nvfp4":
        return TorchAOQuantDType.nvfp4
    raise ValueError(
        f"Invalid dtype: '{v}'. Must be one of: {[e.name for e in TorchAOQuantDType] + ['fp8', 'float8']}"
    )


class QATConfig(BaseModel):
    """
    QAT Config Schema
    """

    activation_dtype: TorchAOQuantDType | None = Field(
        default=None,
        description="Fake quantization layout to use for activation quantization.",
    )
    weight_dtype: TorchAOQuantDType = Field(
        default=TorchAOQuantDType.int8,
        description="Fake quantization layout to use for weight quantization.",
    )
    quantize_embedding: bool | None = Field(
        default=False, description="Quantize embedding"
    )
    group_size: int | None = Field(
        default=32,
        description="The number of elements in each group for per-group fake quantization",
    )
    fake_quant_after_n_steps: int | None = Field(
        default=None, description="The number of steps to apply fake quantization after"
    )

    @field_validator("activation_dtype", "weight_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: Any) -> TorchAOQuantDType | None:
        return validate_ao_dtype(v)


class PTQConfig(BaseModel):
    """
    PTQ Config Schema
    """

    weight_dtype: TorchAOQuantDType = Field(
        default=TorchAOQuantDType.int8,
        description="Fake quantization layout to use for weight quantization.",
    )
    activation_dtype: TorchAOQuantDType | None = Field(
        default=None,
        description="Fake quantization layout to use for activation quantization.",
    )
    quantize_embedding: bool | None = Field(
        default=None, description="Whether to quantize the embedding layer."
    )
    group_size: int | None = Field(
        default=32,
        description="The number of elements in each group for per-group fake quantization",
    )

    @field_validator("activation_dtype", "weight_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: Any) -> TorchAOQuantDType | None:
        return validate_ao_dtype(v)
