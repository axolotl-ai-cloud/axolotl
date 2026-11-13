"""Schema for publishing checkpoints (LoRA or merged) as full/delta chains."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CHAIN_DELTAS = 63


class CheckpointPublishConfig(BaseModel):
    """Publish adapter or merged weights as a full checkpoint followed by XOR deltas."""

    model_config = ConfigDict(extra="forbid")

    export: Literal["lora", "merged"] = Field(
        default="merged",
        json_schema_extra={
            "description": "Publish the LoRA adapter tensors or the base model with the "
            "adapter merged in."
        },
    )
    merged_format: Literal["preserve", "bf16"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Merged weight format: `preserve` keeps the base's own format "
            "(e.g. NVFP4 experts with fresh scales), `bf16` dequantizes a quantized "
            "base to BF16. Defaults to `preserve` for `export: merged`."
        },
    )
    delta: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Publish XOR/zstd/adler32 deltas against the last committed "
            "snapshot. When false every publish is a full checkpoint."
        },
    )
    max_deltas: int = Field(
        default=MAX_CHAIN_DELTAS,
        ge=0,
        le=MAX_CHAIN_DELTAS,
        json_schema_extra={
            "description": "Deltas per chain before a new full checkpoint (at most 63)."
        },
    )
    every: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Publish every N optimizer steps; 0 publishes only on save."
        },
    )
    merge_compile: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Use the compiled merge. Pinned for the whole chain: a change "
            "starts a new full checkpoint."
        },
    )
    merge_compile_precision: Literal["strict", "relaxed"] = Field(
        default="strict",
        json_schema_extra={"description": "Numerics of the compiled merge."},
    )
    destination: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "`s3://bucket/prefix` or a local directory, published to "
            "every `every` steps. Omit when a destination plugin such as `coreweave` "
            "is configured."
        },
    )
    workdir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Node-local directory for the committed baseline and staging. "
            "Defaults to a directory under `output_dir`."
        },
    )

    @model_validator(mode="after")
    def check_options(self):
        if self.export == "lora":
            if self.merged_format is not None:
                raise ValueError(
                    "checkpoint_publish.merged_format applies to export: merged only"
                )
            if self.merge_compile:
                raise ValueError(
                    "checkpoint_publish.merge_compile applies to export: merged only"
                )
        elif self.merged_format is None:
            self.merged_format = "preserve"
        if self.destination is not None:
            if not self.destination.strip():
                raise ValueError("checkpoint_publish.destination must not be empty")
            if self.destination.startswith("s3://"):
                bucket = self.destination[len("s3://") :].split("/", 1)[0]
                if not bucket:
                    raise ValueError(
                        "checkpoint_publish.destination must be s3://bucket/prefix"
                    )
        return self
