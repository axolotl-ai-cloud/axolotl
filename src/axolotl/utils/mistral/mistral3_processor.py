"""Processor for Mistral3 multimodal models with image support"""

from typing import Any, Dict, Optional, Union

import torch
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from axolotl.utils.mistral.mistral_tokenizer import HFMistralTokenizer


class Mistral3ProcessorKwargs(ProcessingKwargs):
    _defaults: Dict[str, Dict[str, Any]] = {
        "text_kwargs": {
            "padding": True,
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "return_dict": True,
            "tokenize": True,
        },
    }


class Mistral3Processor(ProcessorMixin):
    """
    Processor for Mistral3 multimodal models that handles text and images.
    Wraps HFMistralTokenizer and adds image processing capabilities.
    """

    # TODO(nano): This should be removed in transformers V5
    attributes = ["tokenizer"]
    tokenizer_class = "HFMistralTokenizer"

    def __init__(self, tokenizer: HFMistralTokenizer):
        # Don't call super().__init__ to avoid the class validation issue
        self.tokenizer = tokenizer

    @property
    def chat_template(self) -> None:
        """Chat template is not supported. Dummy method to satisfy HuggingFace API."""
        return None

    @property
    def audio_tokenizer(self) -> None:
        """Audio tokenizer is not supported. Dummy method to satisfy HuggingFace API."""
        return None

    def _merge_kwargs(
        self, processor_kwargs_class: Any, **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Merge kwargs with defaults similar to ProcessorMixin"""
        defaults = processor_kwargs_class._defaults
        output_kwargs: Dict[str, Dict[str, Any]] = {}

        for kwarg_type, default_values in defaults.items():
            output_kwargs[kwarg_type] = {**default_values}

        # Update with provided kwargs
        for key, value in kwargs.items():
            # Try to match key to appropriate kwarg type
            if key in ["padding", "truncation", "max_length"]:
                output_kwargs.setdefault("text_kwargs", {}).update({key: value})
            elif key in ["return_tensors", "return_dict", "tokenize"]:
                output_kwargs.setdefault("common_kwargs", {}).update({key: value})
            else:
                # Add to text_kwargs by default
                output_kwargs.setdefault("text_kwargs", {}).update({key: value})

        return output_kwargs

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        **kwargs: Any,
    ) -> Union[BatchFeature, str, list[str]]:
        """
        Apply chat template with image support for Mistral3.

        Similar to VoxtralProcessor, this method extracts images from the conversation,
        calls the tokenizer's apply_chat_template, then adds pixel_values and image_sizes
        to the result.
        """
        output_kwargs = self._merge_kwargs(Mistral3ProcessorKwargs, **kwargs)
        text_kwargs = output_kwargs["text_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        return_tensors = common_kwargs.pop("return_tensors", "pt")
        if return_tensors != "pt":
            raise ValueError(
                f"{self.__class__.__name__} only supports `return_tensors='pt'`."
            )

        return_dict = common_kwargs.pop("return_dict", False)
        tokenize = common_kwargs.pop("tokenize", False)

        # Determine if batched
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple))
            or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]  # type: ignore

        # Call tokenizer's apply_chat_template
        tokenizer_kwargs = {**text_kwargs, **common_kwargs}
        tokenizer_kwargs["return_tensors"] = return_tensors
        tokenizer_kwargs["tokenize"] = tokenize
        tokenizer_kwargs["return_dict"] = return_dict

        encoded_instruct_inputs = self.tokenizer.apply_chat_template(
            conversations,
            **tokenizer_kwargs,
        )

        if tokenize:
            if return_dict:
                # The tokenizer already handles pixel_values, we just need to add image_sizes
                if hasattr(encoded_instruct_inputs, "items"):
                    data: Dict[str, Any] = dict(encoded_instruct_inputs)  # type: ignore
                elif hasattr(encoded_instruct_inputs, "data"):
                    data = encoded_instruct_inputs.data  # type: ignore
                else:
                    raise ValueError("Unknown data type")

                if "pixel_values" in data:
                    pixel_values = data["pixel_values"]

                    # MistralTokenizer returns a Double, so we convert to fp32
                    data["pixel_values"] = pixel_values.to(dtype=torch.float32)

                    # Always batched: [B, C, H, W] -> image_sizes: [B, 2]
                    # Since tensor is homogeneous, all images have same H, W
                    batch_size = pixel_values.shape[0]
                    image_sizes = torch.tensor([pixel_values.shape[-2:]] * batch_size)
                    data["image_sizes"] = image_sizes

                return BatchFeature(data=data, tensor_type=return_tensors)

        if not is_batched:
            return encoded_instruct_inputs[0]

        return encoded_instruct_inputs

    def __call__(
        self,
        text: Optional[
            Union[
                TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
            ]
        ],
        **kwargs: Any,
    ) -> BatchFeature:
        """
        Forward text processing to the tokenizer.
        This method does not support images - use apply_chat_template instead.
        """
        output_kwargs = self._merge_kwargs(Mistral3ProcessorKwargs, **kwargs)
        text_kwargs = output_kwargs["text_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        out = self.tokenizer(text, **text_kwargs)
        return BatchFeature(
            data=out, tensor_type=common_kwargs.pop("return_tensors", None)
        )
