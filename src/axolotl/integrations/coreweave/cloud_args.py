"""SUNK job submission configuration, independent of training imports."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from axolotl.utils.schemas.cloud import CloudImageConfig


class CoreWeaveCloudConfig(CloudImageConfig):
    """Submit an Axolotl container to an existing SUNK Slurm cluster."""

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(default=None, pattern=r"^[A-Za-z0-9/][^\s\x00]*$")

    provider: str = "coreweave"
    backend: Literal["sunk"] = "sunk"
    ssh_host: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.@-]*$"
    )
    shared_dir: str = Field(pattern=r"^/[^\s,:]*$")
    partition: str = Field(min_length=1)
    account: str | None = None
    qos: str | None = None
    nodes: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=8, ge=1)
    cpus_per_task: int = Field(default=64, ge=1)
    memory: str = "0"
    time_limit: str = "24:00:00"
    exclusive: bool = True
    master_port: int = Field(default=29500, ge=1024, le=65535)
    mounts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    env_file: str | None = None
    dry_run: bool = False

    @model_validator(mode="after")
    def validate_job(self):
        import re
        from pathlib import PurePosixPath

        if self.image is None and self.image_build is None:
            raise ValueError("SUNK requires image or image_build")
        if self.image_build and not self.image_build.tag:
            raise ValueError("SUNK image_build requires a registry tag")
        if ".." in PurePosixPath(self.shared_dir).parts:
            raise ValueError("shared_dir cannot contain '..'")
        for name in ("partition", "account", "qos", "memory", "time_limit"):
            value = getattr(self, name)
            if value is not None and not re.fullmatch(r"[A-Za-z0-9_:.+-]+", value):
                raise ValueError(f"Invalid Slurm {name}")
        for path in self.mounts:
            if (
                not re.fullmatch(r"/[^\s,:]+", path)
                or ".." in PurePosixPath(path).parts
            ):
                raise ValueError(
                    "mounts must be absolute shared paths without commas or colons"
                )
        for key in self.env:
            if (
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                or key.startswith("SLURM_")
                or key in {"MASTER_ADDR", "MASTER_PORT"}
            ):
                raise ValueError(
                    f"Invalid or launcher-owned environment variable: {key}"
                )
        if self.env_file is not None:
            if not re.fullmatch(r"/[^\s,:]+", self.env_file):
                raise ValueError(
                    "env_file must be an absolute path visible on allocated SUNK nodes"
                )
        return self
