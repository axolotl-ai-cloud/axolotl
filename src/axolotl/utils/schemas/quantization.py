"""
QAT Config Schema
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from axolotl.utils.schemas.enums import TorchIntDType


class QATConfig(BaseModel):
    """
    QAT Config Schema
    """

    activation_dtype: TorchIntDType | None = Field(
        default=None,
        description='Fake quantization layout to use for activation quantization. Valid options are "int4" and "int8"',
    )
    weight_dtype: TorchIntDType = Field(
        default=TorchIntDType.int8,
        description='Fake quantization layout to use for weight quantization. Valid options are "int4" and "int8"',
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
    def validate_dtype(cls, v: Any) -> TorchIntDType | None:
        if v == "int4":
            return TorchIntDType.int4
        if v == "int8":
            return TorchIntDType.int8
        raise ValueError(f"Invalid dtype: '{v}'. Must be one of: ['int4', 'int8']")


class PTQConfig(BaseModel):
    """
    PTQ Config Schema
    """

    weight_dtype: TorchIntDType = Field(
        default=TorchIntDType.int8,
        description="Fake quantization layout to use for weight quantization. Valid options are uintX for X in [1, 2, 3, 4, 5, 6, 7], or int4, or int8",
    )
    activation_dtype: TorchIntDType | None = Field(
        default=None,
        description='Fake quantization layout to use for activation quantization. Valid options are "int4" and "int8"',
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
    def validate_dtype(cls, v: Any) -> TorchIntDType | None:
        if v == "int4":
            return TorchIntDType.int4
        if v == "int8":
            return TorchIntDType.int8
        raise ValueError(f"Invalid dtype: '{v}'. Must be one of: ['int4', 'int8']")
