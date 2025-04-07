"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

from copy import deepcopy
from typing import Optional

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch import Tensor
from transformers import ProcessorMixin
from transformers.image_utils import load_image


class ProcessingStrategy:
    """Base Processing Strategy class"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.image_token = None
        self.image_token_id = None

        self.image_size = image_size
        self.image_resize_algorithm = (
            image_resize_algorithm or Image.Resampling.BILINEAR
        )

        if hasattr(processor, "image_token"):
            self.image_token = processor.image_token
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
                self.image_token
            )

    def __call__(self, examples: list[dict]) -> list[dict]:
        """
        Preprocess conversation examples to ensure consistent format.
        Converts different conversation formats to OpenAI format with 'messages'.
        Supports two formats:
        1. OpenAI format with 'messages'
        2. Legacy format with 'conversations'

        Args:
            examples: list of conversation dictionaries

        Returns:
            list of dicts in OpenAI format with 'messages' key

        Raises:
            ValueError: If the conversation format is not supported
        """
        role_mapping = {
            "human": "user",
            "gpt": "assistant",
        }

        def normalize_role(role: str) -> str:
            """Normalize role names to OpenAI format. Default to original role if not found."""
            return role_mapping.get(role, role)

        def convert_legacy_format(example: dict) -> dict:
            """Convert legacy 'conversations' format to OpenAI 'messages' format."""
            messages = [
                {"role": normalize_role(convo["from"]), "content": convo["value"]}
                for convo in example["conversations"]
            ]

            # Create new dict without 'conversations' key
            result = deepcopy(example)
            result.pop("conversations")
            result["messages"] = messages
            return result

        def convert_messages_to_multimedia_messages(messages: list[dict]) -> list[dict]:
            """Convert regular messages format to Messages format with content type"""

            new_messages = []
            for message in messages:
                if isinstance(message["content"], str):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": message["content"],
                                }
                            ],
                        }
                    )
                elif isinstance(message["content"], list):
                    content = message["content"]

                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": content,
                        }
                    )

            return new_messages

        processed_examples = []
        for example in examples:
            if not ("messages" in example or "conversations" in example):
                raise ValueError(
                    "Only `messages` and `conversations` message keys are currently supported."
                )

            processed_example = None
            if "messages" in example:  # OpenAI format
                processed_example = example
            else:  # Legacy format
                processed_example = convert_legacy_format(example)

            # convert regular messages format to Messages format with content type
            # for compatibility with apply_chat_template
            processed_example["messages"] = convert_messages_to_multimedia_messages(
                processed_example["messages"]
            )

            # find the image key if it exists
            possible_image_keys = ["images", "image"]
            image_key = None
            for key in possible_image_keys:
                if key in processed_example:
                    image_key = key
                    break

            # if the image key exists, add the image to the first message
            if image_key is not None:
                # TODO: check if it's normal to be single image only for common datasets
                # From observation, it's usually a list of single image but some datasets may have several columns for images
                # Temporary solution: take the first image and suggest people convert their datasets to use multi-content Messages
                image_value = processed_example[image_key][0]

                # Handle image loading (Image, url, path, base64)
                image_value = load_image(image_value)

                if self.image_size is not None:
                    assert hasattr(
                        image_value, "resize"
                    ), "Image does not have a resize method"

                    if isinstance(self.image_size, tuple):
                        image_value = image_value.resize(
                            self.image_size, self.image_resize_algorithm
                        )
                    else:
                        # Set the padding value; here we use black (0, 0, 0) for RGB images
                        padding_color = (0, 0, 0)

                        # When image_size is an int (square target), preserve aspect ratio then pad
                        # This is to prevent aspect ratio distortion when resizing to square
                        image_value = ImageOps.pad(
                            image_value,
                            (self.image_size, self.image_size),
                            method=self.image_resize_algorithm,
                            color=padding_color,
                        )

                # Look for any image type in the first message
                # some dataset have an {type: "image"} in the first message
                ind_to_add = None

                for i, content in enumerate(
                    processed_example["messages"][0]["content"]
                ):
                    # Usually datasets created with image columns, don't have it in the messages itself
                    if content["type"] == "image" and all(
                        k not in content for k in ["image", "url", "path", "base64"]
                    ):
                        ind_to_add = i
                        break

                # If an image type is found, add the image to that index
                if ind_to_add is not None:
                    processed_example["messages"][0]["content"][ind_to_add][
                        "image"
                    ] = image_value
                else:
                    # if no image type is found, add it to end of the first message
                    processed_example["messages"][0]["content"].append(
                        {
                            "type": "image",
                            "image": image_value,
                        }
                    )

            processed_examples.append(processed_example)

        return processed_examples

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = input_ids.clone()

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Ignore the image token index in the loss computation (model specific)
        labels[labels == self.image_token_id] = -100

        return labels


class Qwen2VLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Qwen2-VL"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.image_token = "<|image_pad|>"  # nosec
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.image_token
        )


class Gemma3ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Gemma3"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.image_token = processor.tokenizer.special_tokens_map["boi_token"]
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.image_token
        )

    def process_labels(self, input_ids):
        labels = input_ids.clone()

        # Follows https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        labels[labels == 262144] = -100  # corresponds to <image_soft_token>

        return labels


def get_processing_strategy(
    processor: ProcessorMixin,
    chat_template,
    chat_template_type,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Resampling | None = None,
):
    if chat_template_type == "qwen2_vl":
        return Qwen2VLProcessingStrategy(
            processor, chat_template, image_size, image_resize_algorithm
        )
    if chat_template_type == "gemma3":
        return Gemma3ProcessingStrategy(
            processor, chat_template, image_size, image_resize_algorithm
        )
    if chat_template_type in [
        "llama3_2_vision",
        "llama4",
        "llava",
        "mistral_v7_tekken",
        "pixtral",
    ]:
        return ProcessingStrategy(
            processor, chat_template, image_size, image_resize_algorithm
        )
    raise ValueError(f"Unsupported chat template type: {chat_template_type}")
