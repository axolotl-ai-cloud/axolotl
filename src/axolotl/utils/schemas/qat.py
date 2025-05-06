"""
QAT Config Schema
"""

from enum import Enum
from typing import Any

import torch
from pydantic import BaseModel, Field, field_validator, model_validator


class TorchIntDType(Enum):
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
    quantize_saved_model: bool | None = Field(
        default=False, description="Quantize saved model"
    )

    @field_validator("weight_dtype", mode="before")
    @classmethod
    def validate_weight_dtype(cls, v: Any) -> TorchIntDType | None:
        try:
            return TorchIntDType[v]
        except KeyError as e:
            valid_keys = list(TorchIntDType.__members__.keys())
            raise ValueError(
                f"Invalid weight_dtype: '{v}'. Must be one of: {valid_keys}"
            ) from e

    @field_validator("activation_dtype", mode="before")
    @classmethod
    def validate_activation_dtype(cls, v: Any) -> TorchIntDType | None:
        if v == "int4":
            return TorchIntDType.int4
        elif v == "int8":
            return TorchIntDType.int8
        else:
            raise ValueError(
                f"Invalid activation_dtype: '{v}'. Must be one of: ['int4', 'int8']"
            )

    @model_validator(mode="after")
    def validate_dtype_combination(self) -> "QATConfig":
        if (
            self.activation_dtype is not None
            and self.activation_dtype == TorchIntDType.int4
            and self.weight_dtype != TorchIntDType.int4
        ):
            import ipdb

            ipdb.set_trace()
            raise ValueError(
                "If 'activation_dtype' is specified as 'int4', 'weight_dtype' must also be specified as 'int4'."
            )

        return self
