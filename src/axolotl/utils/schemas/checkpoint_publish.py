"""Schema for publishing checkpoints (LoRA or merged) as full/delta chains."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CHAIN_DELTAS = 63


class RecoveryCheckpointConfig(BaseModel):
    """Crash-recovery adapter checkpoints written from the staged LoRA snapshot."""

    model_config = ConfigDict(extra="forbid")

    every: int = Field(
        ge=1,
        json_schema_extra={
            "description": "Write a recovery checkpoint every N optimizer steps."
        },
    )
    keep: int = Field(
        default=2,
        ge=1,
        json_schema_extra={
            "description": "Recovery checkpoints kept in `dir`; older ones are deleted. "
            "The newest one with optimizer state is always kept as well."
        },
    )
    dir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Node-local directory for per-rank shards. Defaults to "
            "`recovery` under `checkpoint_publish.workdir` (or `output_dir`)."
        },
    )
    optimizer_every: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Also save optimizer state every N steps (a multiple of "
            "`every`); 0 never saves it."
        },
    )
    consolidate_every: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Consolidate every N-th step's checkpoint (a multiple of "
            "`every`) into a PEFT adapter under `consolidate_dir`; 0 consolidates only "
            "the final one."
        },
    )
    consolidate_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Durable directory for consolidated adapters. Defaults to "
            "`recovery-adapters` under `output_dir`."
        },
    )
    final: bool = Field(
        default=True,
        json_schema_extra={
            "description": "At the end of training, write and consolidate a recovery "
            "checkpoint of the final step."
        },
    )

    @model_validator(mode="after")
    def check_cadence(self):
        for name in ("optimizer_every", "consolidate_every"):
            value = getattr(self, name)
            if value and value % self.every:
                raise ValueError(
                    f"checkpoint_publish.recovery.{name} must be a multiple of every"
                )
        return self


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
            "(e.g. NVFP4 experts with fresh scales), `bf16` writes BF16. "
            "Defaults to `preserve` for `export: merged`."
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
            "description": "`s3://bucket/prefix` or a local directory. Omit when a "
            "destination plugin such as `coreweave` is configured."
        },
    )
    workdir: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Node-local directory for the committed baseline and staging. "
            "Defaults to a directory under `output_dir`."
        },
    )
    recovery: RecoveryCheckpointConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Asynchronous crash-recovery adapter checkpoints from the "
            "same LoRA snapshot as per-step publishing, with their own cadence and "
            "rotation."
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
