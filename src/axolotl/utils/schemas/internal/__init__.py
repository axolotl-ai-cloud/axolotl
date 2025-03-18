"""module for gpu capabilities"""

from typing import Optional

from pydantic import BaseModel, Field


class GPUCapabilities(BaseModel):
    """model to manage the gpu capabilities statically"""

    bf16: bool = Field(default=False)
    fp8: bool = Field(default=False)
    n_gpu: int = Field(default=1)
    n_node: int = Field(default=1)
    compute_capability: Optional[str] = Field(default=None)


class EnvCapabilities(BaseModel):
    """model to manage the environment capabilities statically"""

    torch_version: Optional[str] = Field(default=None)
