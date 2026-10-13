"""Configuration for CoreWeave Dedicated Inference."""

from pydantic import BaseModel, ConfigDict, Field


class CoreWeaveConfig(BaseModel):
    """Attach to an existing hot-load-enabled Dynamo vLLM deployment."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    path_prefix: str = Field(min_length=1)
    gateway_url: str
    model: str
    api_url: str = "https://api.coreweave.com"
    api_token_env: str = "CW_API_TOKEN"
    s3_endpoint_url: str | None = None
    s3_profile: str | None = None
    timeout: float = Field(default=900, gt=0)
    poll_interval: float = Field(default=5, gt=0)
    max_deltas: int = Field(default=128, ge=0, le=128)
    max_shard_size: int = Field(default=1 << 30, gt=0)
    rollouts: bool = False
    async_publish: bool = True


class CoreWeaveArgs(BaseModel):
    coreweave: CoreWeaveConfig | None = None
