"""module for gpu capabilities"""
from pydantic import BaseModel, Field


class GPUCapabilities(BaseModel):
    """model to manage the gpu capabilities statically"""

    bf16: bool = Field(default=False)
    fp8: bool = Field(default=False)
    n_gpu: int = Field(default=1)
