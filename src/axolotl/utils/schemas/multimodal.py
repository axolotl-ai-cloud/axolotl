"""Pydantic models for multimodal-related configuration"""

from typing import Literal

from PIL.Image import Resampling
from pydantic import BaseModel, Field, field_validator


class MultiModalConfig(BaseModel):
    """Multi-modal configuration subset"""

    image_size: int | tuple[int, int] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "The size of the image to resize to. It can be an integer (resized into padded-square image) or a tuple (width, height)."
                "If not provided, we will attempt to load from preprocessor.size, otherwise, images won't be resized."
            )
        },
    )
    image_resize_algorithm: (
        Literal["bilinear", "bicubic", "lanczos"] | Resampling | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "The resampling algorithm to use for image resizing. Default is bilinear. Please refer to PIL.Image.Resampling for more details."
        },
    )

    @field_validator("image_resize_algorithm", mode="before")
    @classmethod
    def convert_image_resize_algorithm(cls, image_resize_algorithm):
        """
        Convert the image resize algorithm to a PIL.Image.Resampling enum.
        """
        if isinstance(image_resize_algorithm, str):
            image_resize_algorithm = image_resize_algorithm.lower()
            if image_resize_algorithm == "bilinear":
                image_resize_algorithm = Resampling.BILINEAR
            elif image_resize_algorithm == "bicubic":
                image_resize_algorithm = Resampling.BICUBIC
            elif image_resize_algorithm == "lanczos":
                image_resize_algorithm = Resampling.LANCZOS
            else:
                raise ValueError(
                    f"Invalid image resize algorithm: {image_resize_algorithm}"
                )
        return image_resize_algorithm
