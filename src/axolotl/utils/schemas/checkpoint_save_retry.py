"""Schema for retrying checkpoint writes after disk errors."""

from pydantic import BaseModel, Field


class CheckpointSaveRetryConfig(BaseModel):
    """Wait and rewrite a checkpoint file whose write failed on a full disk, quota or I/O error."""

    wait_seconds: float = Field(
        default=300,
        ge=0,
        json_schema_extra={
            "description": "Seconds to wait before rewriting the file, for example while space is freed. Other ranks wait at their next collective meanwhile, so keep wait_seconds * retries below ddp_timeout."
        },
    )
    retries: int = Field(
        default=1,
        ge=1,
        json_schema_extra={
            "description": "Retries per file before the error is raised and the save fails."
        },
    )
