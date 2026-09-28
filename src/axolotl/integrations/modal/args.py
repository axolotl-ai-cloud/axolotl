"""Modal image and registry configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from axolotl.utils.schemas.cloud import CloudImageConfig


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
