"""
QAT Config Schema
"""

from pydantic import BaseModel, Field

from axolotl.utils.config import TorchDType


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
