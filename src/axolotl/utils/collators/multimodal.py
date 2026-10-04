"""Collation for VLMs with flat image batches and per-image spatial sizes."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

_IMAGE_FIELDS = frozenset({"pixel_values", "image_sizes"})


def collate_image_inputs(
    inputs: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    """Pad CHW images in example order, retaining their unpadded spatial sizes.

    This layout is used by Pixtral-style models. Tiled images, grids, and
    cross-attention inputs require their own layout adapter.
    """
    images: list[torch.Tensor] = []
    sizes: list[torch.Tensor] = []
    for example in inputs:
        unknown = set(example) - _IMAGE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported image model inputs: {sorted(unknown)}")
        if not example:
            continue
        if set(example) != _IMAGE_FIELDS:
            raise ValueError("pixel_values and image_sizes must be supplied together")
        pixels = example["pixel_values"]
        image_sizes = torch.as_tensor(example["image_sizes"], dtype=torch.long)
        if image_sizes.ndim != 2 or image_sizes.shape[1] != 2:
            raise ValueError("image_sizes must have shape [images, 2]")
        if len(pixels) != image_sizes.shape[0]:
            raise ValueError(
                "pixel_values and image_sizes must have the same image count"
            )
        for pixel, size in zip(pixels, image_sizes, strict=True):
            image = torch.as_tensor(pixel, dtype=torch.float32)
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("each image must have shape [3, height, width]")
            if image.device.type != "cpu" or size.device.type != "cpu":
                raise ValueError("image collation expects CPU inputs")
            if bool((size <= 0).any()) or bool(
                (size > torch.tensor(image.shape[-2:])).any()
            ):
                raise ValueError(
                    "image_sizes must describe positive extents within pixels"
                )
            images.append(image)
            sizes.append(size)
    if not images:
        return {}
    height = max(image.shape[-2] for image in images)
    width = max(image.shape[-1] for image in images)
    return {
        "pixel_values": torch.stack(
            [
                F.pad(image, (0, width - image.shape[-1], 0, height - image.shape[-2]))
                for image in images
            ]
        ),
        "image_sizes": torch.stack(sizes),
    }


def image_model_inputs(
    inputs: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move only validated image fields into a model forward call."""
    if not inputs:
        return {}
    if set(inputs) != _IMAGE_FIELDS:
        raise ValueError("image model inputs require pixel_values and image_sizes only")
    return {key: value.to(device=device) for key, value in inputs.items()}
