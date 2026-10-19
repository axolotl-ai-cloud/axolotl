"""Nebius launcher configuration and resource validation."""

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from axolotl.utils.schemas.cloud import CloudImageConfig

NonemptyString = Annotated[
    str, StringConstraints(strict=True, min_length=1, pattern=r"^[^\x00]*\S[^\x00]*$")
]


class NebiusVolume(BaseModel):
    """An existing Nebius volume mounted into the training container."""

    model_config = ConfigDict(extra="forbid")

    source: NonemptyString
    mount: NonemptyString
    mode: Literal["ro", "rw"] = "ro"

    @model_validator(mode="after")
    def validate_paths(self):
        if ":" in self.source:
            raise ValueError("Volume source must be a Nebius resource ID or name")
        if not self.mount.startswith("/") or ":" in self.mount:
            raise ValueError("Volume mount must be an absolute container path")
        path = PurePosixPath("/" + self.mount.lstrip("/"))
        if ".." in path.parts:
            raise ValueError("Volume mount cannot contain parent traversal")
        self.mount = str(path)
        return self


class NebiusCloudConfig(CloudImageConfig):
    """Provider-owned compute, image, storage, and environment settings."""

    model_config = ConfigDict(extra="forbid")

    provider: StrictStr | None = None
    platform: NonemptyString
    preset: NonemptyString
    profile: NonemptyString | None = None
    parent_id: NonemptyString | None = None
    subnet_id: NonemptyString | None = None
    disk_size: NonemptyString | None = None
    output: NonemptyString | None = None
    timeout: Annotated[int, Field(strict=True, ge=3600, le=604800)] = 86400
    show_context: StrictBool = False
    dry_run: StrictBool = False
    volumes: list[NebiusVolume] = Field(default_factory=list)
    env: dict[str, StrictStr] = Field(default_factory=dict)
    env_secret: dict[str, StrictStr] = Field(default_factory=dict)

    @field_validator("env", "env_secret")
    @classmethod
    def validate_environment(cls, values, info):
        for name, value in values.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(
                    f"Invalid environment variable name in {info.field_name}"
                )
            if name in {
                "NEBIUS_OUTPUT_DIR",
                "AXOLOTL_NEBIUS_COMPLETION_FILE",
                "AXOLOTL_NEBIUS_EXPORT_DIR",
            }:
                raise ValueError(f"{name} is managed by the Nebius launcher")
            if "\x00" in value or (info.field_name == "env_secret" and not value):
                raise ValueError(f"Invalid value in {info.field_name}")
        return values

    @model_validator(mode="after")
    def validate_resources(self):
        if self.image is None and self.image_build is None:
            raise ValueError("Nebius requires image or image_build")
        if self.image_build:
            if not self.image_build.tag:
                raise ValueError("Nebius image_build requires a registry tag")
            if self.image_build.platform != "linux/amd64":
                raise ValueError("Nebius GPU images require linux/amd64")
            if self.show_context or self.dry_run:
                raise ValueError("Use a prebuilt image for show_context or dry_run")
        if self.show_context and self.dry_run:
            raise ValueError("Choose either show_context or dry_run")
        if self.env.keys() & self.env_secret.keys():
            raise ValueError("A variable cannot appear in both env and env_secret")
        mounts = [PurePosixPath("/outputs")]
        for volume in self.volumes:
            path = PurePosixPath(volume.mount)
            if any(path.is_relative_to(p) or p.is_relative_to(path) for p in mounts):
                raise ValueError(
                    "Volume mounts must not overlap each other or /outputs"
                )
            mounts.append(path)
        return self
