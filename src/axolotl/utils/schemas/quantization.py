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
        default=None, description="Activation dtype"
    )
    weight_dtype: TorchIntDType = Field(
        default=TorchIntDType.int8, description="Weight dtype"
    )
    quantize_embedding: bool | None = Field(
        default=False, description="Quantize embedding"
    )
    group_size: int | None = Field(default=32, description="Group size")
    fake_quant_after_n_steps: int | None = Field(
        default=None, description="Fake quant after n steps"
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
        default=TorchIntDType.int8, description="Weight dtype"
    )
    activation_dtype: TorchIntDType | None = Field(
        default=None, description="Activation dtype"
    )
    quantize_embedding: bool | None = Field(
        default=None, description="Quantize embedding"
    )
    group_size: int | None = Field(default=32, description="Group size")

    @field_validator("activation_dtype", "weight_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: Any) -> TorchIntDType | None:
        if v == "int4":
            return TorchIntDType.int4
        if v == "int8":
            return TorchIntDType.int8
        raise ValueError(f"Invalid dtype: '{v}'. Must be one of: ['int4', 'int8']")
