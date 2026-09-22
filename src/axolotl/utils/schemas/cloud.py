"""Image sources for whole-job cloud launchers."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    StrictStr,
    StringConstraints,
    model_validator,
)

ImageReference = Annotated[
    str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][^\s\x00]*$")
]


class CloudImageBuild(BaseModel):
    """Dockerfile build using an explicit local source context."""

    model_config = ConfigDict(extra="forbid")

    context: DirectoryPath
    dockerfile: Path = Path("Dockerfile")
    tag: ImageReference | None = None
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64"
    build_args: dict[str, StrictStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_paths(self):
        self.context = self.context.resolve()
        self.dockerfile = (self.context / self.dockerfile).resolve()
        if not self.dockerfile.is_relative_to(self.context):
            raise ValueError("image_build.dockerfile must be inside the build context")
        if not self.dockerfile.is_file():
            raise ValueError(f"Dockerfile does not exist: {self.dockerfile}")
        return self


class CloudImageConfig(BaseModel):
    """Select a prebuilt image or build one from a local Dockerfile."""

    image: ImageReference | None = None
    image_build: CloudImageBuild | None = None

    @model_validator(mode="after")
    def exclusive_source(self):
        if self.image is not None and self.image_build is not None:
            raise ValueError("Choose either image or image_build")
        return self


class ModalImageRegistry(BaseModel):
    """Modal-managed credentials for pulling a private registry image."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["registry", "aws_ecr", "gcp_artifact_registry"] = "registry"
    secret: Annotated[str, StringConstraints(strict=True, min_length=1)]


class ModalImageConfig(CloudImageConfig):
    """Image source and provider-native pull authentication for Modal."""

    image_registry: ModalImageRegistry | None = None

    @model_validator(mode="after")
    def registry_source(self):
        if self.image_build and self.image_build.platform != "linux/amd64":
            raise ValueError("Modal images must use linux/amd64")
        if (
            self.image_registry
            and not self.image
            and not (self.image_build and self.image_build.tag)
        ):
            raise ValueError(
                "image_registry requires image or image_build.tag; "
                "untagged builds use Modal's native builder"
            )
        return self


class BasetenImageConfig(CloudImageConfig):
    """Image source and native Truss DockerAuth configuration for Baseten."""

    docker_auth: dict[str, Any] | None = None
