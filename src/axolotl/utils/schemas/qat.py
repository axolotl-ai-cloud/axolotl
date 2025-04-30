"""
QAT Config Schema
"""

from enum import Enum
from pydantic import BaseModel, Field, field_validator
import torch
from typing import Any


class TorchDType(Enum):
    int1 = torch.int1
    uint1 = torch.uint1
    int2 = torch.int2
    uint2 = torch.uint2
    int3 = torch.int3
    uint3 = torch.uint3
    int4 = torch.int4
    uint4 = torch.uint4
    int5 = torch.int5
    uint5 = torch.uint5
    int6 = torch.int6
    uint6 = torch.uint6
    int7 = torch.int7
    uint7 = torch.uint7
    int8 = torch.int8
    uint8 = torch.uint8


class QATConfig(BaseModel):
    """
    QAT Config Schema
    """

    activation_dtype: TorchDType | None = Field(
        default=None, description="Activation dtype"
    )
    weight_dtype: TorchDType | None = Field(default=None, description="Weight dtype")
    quantize_embedding: bool | None = Field(
        default=False, description="Quantize embedding"
    )
    group_size: int | None = Field(default=32, description="Group size")

    @field_validator('activation_dtype', 'weight_dtype', mode='before')
    @classmethod
    def map_str_to_dtype_enum(cls, v: Any) -> TorchDType | None:
        try:
            return TorchDType[v]
        except KeyError as e:
            valid_keys = list(TorchDType.__members__.keys())
            raise ValueError(
                f"Invalid TorchDType string: '{v}'. Must be one of: {valid_keys}"
            ) from e