from pydantic import BaseModel, Field


class GPUCapabilities(BaseModel):
    bf16: bool = Field(default=False)
    fp8: bool = Field(default=False)
    n_gpu: int = Field(default=1)
