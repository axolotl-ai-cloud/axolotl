"""Source-backed image prompt preparation for Nemotron Diffusion VLM."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers.processing_utils import ProcessorMixin

from .compat import resolve_nemotron_vlm_source


def _image_processing_module(model_source: str | Path, revision: str | None):
    """Load the image processor beside the native model implementation."""
    source = resolve_nemotron_vlm_source(model_source, revision=revision)
    from transformers.dynamic_module_utils import (
        HF_MODULES_CACHE,
        get_cached_module_file,
    )

    try:
        module_file = Path(
            get_cached_module_file(
                str(source), "image_processing.py", local_files_only=True
            )
        )
    except ImportError as error:
        if "cv2" not in str(error):
            raise
        raise ImportError(
            "Nemotron VLM image preprocessing requires OpenCV. Install the "
            "Axolotl vision extra: `pip install 'axolotl[vision]'`."
        ) from error
    if module_file.is_absolute():
        try:
            relative_module = module_file.relative_to(HF_MODULES_CACHE).with_suffix("")
        except ValueError as error:
            raise RuntimeError(
                "transformers returned an image processor outside its dynamic module cache"
            ) from error
    else:
        relative_module = module_file.with_suffix("")
    return importlib.import_module(".".join(relative_module.parts))


def _serialize_state(state: Any) -> str:
    if state is None:
        raise ValueError("decision state is required")
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _decision_messages(
    system_text: str, state: Any, images: Sequence[str]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image}} for image in images
    ]
    content.append({"type": "text", "text": _serialize_state(state)})
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def _image_sources(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    sources = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "image_url":
                image = block.get("image_url", {})
                source = image.get("url") if isinstance(image, Mapping) else None
            elif block.get("type") == "image":
                source = next(
                    (block[key] for key in ("url", "path", "image") if key in block),
                    None,
                )
            else:
                source = None
            if not isinstance(source, str) or not source:
                if block.get("type") in {"image", "image_url"}:
                    raise ValueError(
                        "Nemotron VLM image content needs a non-empty URL or path"
                    )
                continue
            sources.append(source)
    return sources


def prepare_messages(
    tokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    model_source: str | Path,
    revision: str | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    return_tensors: str | None = "pt",
    max_image_size: int | None = None,
) -> Mapping[str, Any]:
    """Expand native image markers and preprocess image inputs.

    The upstream image processor owns image loading, geometry, normalization, and
    marker expansion. This adapter only preserves ordered message/image ownership
    while avoiding the upstream stack operation for differently sized images.
    """
    images = _image_sources(messages)
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise TypeError("decision images must be an ordered sequence of paths or URLs")
    if not images:
        raise ValueError("Nemotron VLM preparation requires at least one image")
    if any(not isinstance(image, str) or not image for image in images):
        raise ValueError("decision images must contain non-empty paths or URLs")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    return _prepare_rendered_prompt(
        tokenizer,
        prompt,
        images,
        model_source=model_source,
        revision=revision,
        return_tensors=return_tensors,
        max_image_size=max_image_size,
    )


def _prepare_rendered_prompt(
    tokenizer,
    prompt: str,
    images: Sequence[str],
    *,
    model_source: str | Path,
    revision: str | None,
    return_tensors: str | None,
    max_image_size: int | None,
) -> Mapping[str, Any]:
    processor = _image_processing_module(model_source, revision)
    marker = processor.IMG_START_TOKEN
    pieces = prompt.split(marker)
    if len(pieces) != len(images) + 1:
        raise ValueError("Nemotron VLM prompt image placeholders do not match images")
    pixel_values = []
    image_sizes = []
    expanded = []
    for image in images:
        image_kwargs = (
            {} if max_image_size is None else {"max_image_size": max_image_size}
        )
        width_tokens, height_tokens, pixels = processor.encode_image(
            processor.load_image(image), **image_kwargs
        )
        expanded.append(processor.build_image_token_str(width_tokens, height_tokens))
        pixel_values.append(torch.from_numpy(pixels).to(dtype=torch.float32))
        image_sizes.append(
            (
                height_tokens
                * processor.DEFAULT_PATCH_SIZE
                * processor.DEFAULT_SPATIAL_MERGE_SIZE,
                width_tokens
                * processor.DEFAULT_PATCH_SIZE
                * processor.DEFAULT_SPATIAL_MERGE_SIZE,
            )
        )
    expanded_prompt = "".join(
        piece + (expanded[index] if index < len(expanded) else "")
        for index, piece in enumerate(pieces)
    )
    input_ids = torch.as_tensor(
        tokenizer(expanded_prompt, return_tensors=return_tensors).input_ids
    )
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.size(0) != 1
    ):
        raise ValueError(
            "Nemotron VLM tokenizer must return one [1, sequence] input_ids tensor"
        )

    image_token_ids = tuple(
        int(getattr(processor, name))
        for name in ("IMG_START_ID", "IMG_PAD_ID", "IMG_BREAK_ID", "IMG_END_ID")
    )
    return {
        "input_ids": input_ids[0].tolist(),
        "pixel_values": pixel_values,
        "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
        "image_token_ids": image_token_ids,
    }


def prepare_decision_prompt(
    tokenizer,
    system_text: str,
    state: Any,
    images: Sequence[str],
    *,
    model_source: str | Path,
    revision: str | None = None,
    max_image_size: int | None = None,
) -> Mapping[str, Any]:
    """Expand image markers before constructing a decision canvas."""
    return prepare_messages(
        tokenizer,
        _decision_messages(system_text, state, images),
        model_source=model_source,
        revision=revision,
        add_generation_prompt=True,
        enable_thinking=False,
        max_image_size=max_image_size,
    )


class NemotronVLMProcessor(ProcessorMixin):
    """Processor facade that keeps native Nemotron VLM image preparation together."""

    attributes = ["tokenizer"]
    tokenizer_class = ("PreTrainedTokenizerBase",)

    def __init__(
        self,
        tokenizer,
        *,
        model_source: str | Path,
        revision: str | None = None,
        max_image_size: int | None = None,
    ):
        super().__init__(tokenizer=tokenizer)
        self.model_source = str(model_source)
        self.revision = revision
        self.max_image_size = max_image_size

    @classmethod
    def from_pretrained(cls, model_source, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        unset = object()
        revision = kwargs.pop("revision", unset)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        max_image_size = kwargs.pop("max_image_size", unset)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                f"unsupported Nemotron VLM processor settings: {unsupported}"
            )
        metadata_path = Path(model_source) / "nemotron_vlm_processor.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        )
        source = metadata.get("model_source", model_source)
        if revision is unset:
            revision = metadata.get("revision")
        if max_image_size is unset:
            max_image_size = metadata.get("max_image_size")
        if tokenizer is None:
            from transformers import AutoTokenizer

            if not trust_remote_code:
                raise ValueError(
                    "Nemotron VLM processor requires trust_remote_code: true when "
                    "loading its tokenizer."
                )
            tokenizer = AutoTokenizer.from_pretrained(
                model_source, revision=revision, trust_remote_code=trust_remote_code
            )
        return cls(
            tokenizer,
            model_source=source,
            revision=revision,
            max_image_size=max_image_size,
        )

    def apply_chat_template(self, conversation, **kwargs):
        tokenize = kwargs.get("tokenize", False)
        images = _image_sources(conversation)
        if not images:
            return self.tokenizer.apply_chat_template(conversation, **kwargs)
        if not tokenize:
            return self.tokenizer.apply_chat_template(conversation, **kwargs)
        output = prepare_messages(
            self.tokenizer,
            conversation,
            model_source=self.model_source,
            revision=self.revision,
            add_generation_prompt=kwargs.get("add_generation_prompt", False),
            enable_thinking=kwargs.get("enable_thinking", True),
            return_tensors=kwargs.get("return_tensors", "pt"),
            max_image_size=self.max_image_size,
        )
        return output if kwargs.get("return_dict", False) else output["input_ids"]

    def __call__(self, text=None, images=None, **kwargs):
        if images is None:
            return self.tokenizer(text, **kwargs)
        image_list = (
            images
            if isinstance(images, Sequence) and not isinstance(images, (str, bytes))
            else [images]
        )
        if not isinstance(text, str):
            raise TypeError("Nemotron VLM processor requires rendered text with images")
        return _prepare_rendered_prompt(
            self.tokenizer,
            text,
            image_list,
            model_source=self.model_source,
            revision=self.revision,
            return_tensors=kwargs.get("return_tensors", "pt"),
            max_image_size=self.max_image_size,
        )

    def save_pretrained(self, save_directory, **kwargs):
        saved = self.tokenizer.save_pretrained(save_directory, **kwargs)
        metadata = Path(save_directory) / "nemotron_vlm_processor.json"
        metadata.write_text(
            json.dumps(
                {
                    "model_source": self.model_source,
                    "revision": self.revision,
                    "max_image_size": self.max_image_size,
                },
                sort_keys=True,
            )
        )
        return (*saved, str(metadata))
