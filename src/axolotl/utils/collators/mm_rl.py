"""On-the-fly image loading for TRL's vision RL paths (DPO/IPO, KTO, GRPO).

TRL expects an ``images`` list of loaded images plus bare ``{"type": "image"}``
prompt placeholders; path/URL/base64 references are resolved here at collate time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL.Image import Resampling
from transformers.image_utils import load_image
from trl.trainer.dpo_trainer import DataCollatorForVisionPreference
from trl.trainer.kto_trainer import DataCollatorForVisionUnpairedPreference

from axolotl.processing_strategies import resize_image
from axolotl.utils.dict import remove_none_values

IMAGE_REF_KEYS = ("image", "url", "path", "base64")
MESSAGE_FIELDS = ("prompt", "chosen", "rejected", "completion")


@dataclass
class MultimodalRLExampleNormalizer:
    """Resolve image refs into loaded, optionally resized, images in TRL's format."""

    image_size: int | tuple[int, int] | None = None
    image_resize_algorithm: Resampling | None = None

    def __call__(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.normalize(example) for example in examples]

    def _load(self, ref: Any):
        image = load_image(ref)
        if self.image_size is not None:
            image = resize_image(image, self.image_size, self.image_resize_algorithm)
        return image

    def normalize(self, example: dict[str, Any]) -> dict[str, Any]:
        example = dict(example)
        images = example.pop("images", None)
        # Merging an ``image`` dataset with an ``images`` one leaves the other key None.
        single = example.pop("image", None)
        column_images = images or ([single] if single is not None else [])
        column_images = [img for img in column_images if img is not None]

        for key in MESSAGE_FIELDS:
            if isinstance(example.get(key), list):
                # Arrow schema merges add None keys that TRL would read as image payloads.
                example[key] = remove_none_values(example[key])

        prompt = example.get("prompt")
        if isinstance(prompt, list):
            example["images"] = self._fill_prompt_images(prompt, column_images)
        else:
            example["images"] = [self._load(ref) for ref in column_images]
        return example

    def _fill_prompt_images(
        self, prompt: list[dict[str, Any]], column_images: list[Any]
    ) -> list[Any]:
        """Return prompt images in document order, leaving bare placeholders for TRL.

        Column images fill bare placeholders; leftovers are prepended to the first user
        turn, matching TRL's handling of string content."""
        image_parts = [
            part
            for message in prompt
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image"
        ]
        if not column_images and not image_parts:
            return []
        # TRL adds placeholders to the first *string* user turn it finds, which would
        # double-count ours; structuring every user turn opts out of that.
        for message in prompt:
            if message.get("role") == "user" and isinstance(
                message.get("content"), str
            ):
                message["content"] = [{"type": "text", "text": message["content"]}]

        num_bare = sum(
            1 for part in image_parts if not any(k in part for k in IMAGE_REF_KEYS)
        )
        num_leftover = len(column_images) - num_bare
        if num_leftover > 0:
            first_user = next((m for m in prompt if m.get("role") == "user"), None)
            if first_user is None:
                raise ValueError(
                    "Sample has images but its prompt has no user turn to attach them to."
                )
            first_user["content"] = [
                *({"type": "image"} for _ in range(num_leftover)),
                *(first_user.get("content") or []),
            ]

        column_iter = iter(column_images)
        images = []
        for message in prompt:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for i, part in enumerate(content):
                if part.get("type") != "image":
                    continue
                ref = next((part[k] for k in IMAGE_REF_KEYS if k in part), None)
                if ref is None:
                    ref = next(column_iter, None)
                if ref is None:
                    # More placeholders than images: TRL raises a count-mismatch error.
                    continue
                images.append(self._load(ref))
                content[i] = {"type": "image"}
        return images


@dataclass
class AxolotlVisionPreferenceCollator(DataCollatorForVisionPreference):
    """TRL's DPO vision collator with axolotl image loading/resizing."""

    normalizer: MultimodalRLExampleNormalizer = field(
        default_factory=MultimodalRLExampleNormalizer
    )

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return super().torch_call(self.normalizer(examples))


@dataclass
class AxolotlVisionUnpairedPreferenceCollator(DataCollatorForVisionUnpairedPreference):
    """TRL's KTO vision collator with axolotl image loading/resizing."""

    normalizer: MultimodalRLExampleNormalizer = field(
        default_factory=MultimodalRLExampleNormalizer
    )

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return super().torch_call(self.normalizer(examples))
