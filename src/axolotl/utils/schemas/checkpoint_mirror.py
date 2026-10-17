"""Schema for mirroring checkpoints to a shared filesystem in the background."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CheckpointMirrorConfig(BaseModel):
    """Save checkpoints to a fast local output_dir and copy them to `dir` without blocking training."""

    dir: str = Field(
        json_schema_extra={
            "description": "Directory on a shared filesystem (e.g. NFS) that receives a copy of every checkpoint and, at the end of training, the final outputs. A checkpoint appears there only once it is complete, so it can be resumed from."
        },
    )
    delete: Literal["mirror", "keep"] = Field(
        default="mirror",
        json_schema_extra={
            "description": "`mirror` keeps as many checkpoints in `dir` as save_total_limit keeps in output_dir. `keep` never deletes from `dir`, unless checkpoint_mirror.save_total_limit is set."
        },
    )
    save_total_limit: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "With `delete: keep`, the number of newest checkpoints to keep in `dir` (the best checkpoint is always kept). Unset keeps all."
        },
    )
    max_pending: int = Field(
        default=2,
        ge=1,
        json_schema_extra={
            "description": "Checkpoints waiting to be copied before a save blocks until the oldest copy finishes. Bounds the local disk used by queued checkpoints when the shared filesystem is slower than the save interval."
        },
    )
    copy_threads: int = Field(
        default=4,
        ge=1,
        json_schema_extra={"description": "Files copied in parallel per checkpoint."},
    )
    resume: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Let auto_resume_from_checkpoints resume from `dir` when it holds a newer checkpoint than output_dir, or when output_dir is local to each node of a multi-node run."
        },
    )

    @model_validator(mode="after")
    def check_save_total_limit(self):
        if self.delete == "mirror" and self.save_total_limit is not None:
            raise ValueError(
                "checkpoint_mirror.save_total_limit only applies with `delete: keep`; "
                "with `delete: mirror` the mirror follows save_total_limit"
            )
        return self
