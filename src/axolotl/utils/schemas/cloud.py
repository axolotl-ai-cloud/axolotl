"""Image sources for whole-job cloud launchers."""

from pathlib import Path
from typing import Annotated, Literal

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

    @classmethod
    def from_config(cls, config: dict, *, config_dir: Path | None = None):
        config = dict(config)
        if isinstance(config.get("image_build"), dict):
            build = dict(config["image_build"])
            context = build.get("context")
            if config_dir is not None and isinstance(context, (str, Path)):
                build["context"] = config_dir / context
            config["image_build"] = build
        return cls.model_validate(config)

    @model_validator(mode="after")
    def exclusive_source(self):
        if self.image is not None and self.image_build is not None:
            raise ValueError("Choose either image or image_build")
        return self
