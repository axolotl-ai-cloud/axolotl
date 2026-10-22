"""Configuration for CoreWeave Dedicated Inference."""

import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator

from axolotl.utils.schemas.checkpoint_publish import MAX_CHAIN_DELTAS

LEGACY_MAX_DELTAS = 128


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
    max_deltas: int = Field(default=MAX_CHAIN_DELTAS, ge=0, le=MAX_CHAIN_DELTAS)
    max_shard_size: int = Field(default=1 << 30, gt=0)
    rollouts: bool = False
    async_publish: bool = True

    @field_validator("max_deltas", mode="before")
    @classmethod
    def cap_legacy_max_deltas(cls, value):
        if isinstance(value, int) and MAX_CHAIN_DELTAS < value <= LEGACY_MAX_DELTAS:
            warnings.warn(
                f"coreweave.max_deltas={value} exceeds CoreWeave's {MAX_CHAIN_DELTAS}-delta "
                f"chain limit and is capped to {MAX_CHAIN_DELTAS}; set "
                "checkpoint_publish.max_deltas instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return MAX_CHAIN_DELTAS
        return value


class CoreWeaveArgs(BaseModel):
    coreweave: CoreWeaveConfig | None = None
