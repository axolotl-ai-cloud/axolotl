"""Image loading/resizing helpers shared by the eager and prepared multimodal paths."""

from PIL import ImageOps
from PIL.Image import Resampling
from transformers.image_utils import load_image


def load_and_resize_image(
    image,
    image_size: int | tuple[int, int] | None,
    resample: Resampling | None = None,
):
    """Load an image and resize per ``cfg.image_size``: tuple = exact (width, height),
    int = aspect-preserving pad to a black square, None = load without resizing."""
    image = load_image(image)
    if image_size is None:
        return image

    assert hasattr(image, "resize"), "Image does not have a resize method"
    # `or` would clobber Resampling.NEAREST (value 0)
    resample = Resampling.BILINEAR if resample is None else resample

    if isinstance(image_size, tuple):
        return image.resize(image_size, resample)

    return ImageOps.pad(
        image,
        (image_size, image_size),
        method=resample,
        color=(0, 0, 0),
    )
