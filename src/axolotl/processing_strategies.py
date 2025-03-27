"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

import ast
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

        def convert_multiple_choice_to_multimedia_messages(
            messages: dict,
        ) -> list[dict]:

            def construct_prompt(sample):
                question = sample["question"]
                options = sample["options"]
                if isinstance(options, str):
                    options = ast.literal_eval(options)

                example = ""
                start_chr = "A"
                prediction_range = []
                index2ans = {}
                for option in options:
                    prediction_range.append(start_chr)
                    example += f"({start_chr}) {option}\n"
                    index2ans[start_chr] = option
                    start_chr = chr(ord(start_chr) + 1)

                empty_prompt_sample_structure = "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly."
                empty_prompt = empty_prompt_sample_structure.format(question, example)

                return empty_prompt

            new_messages = []

            user_content = construct_prompt(messages)
            assistant_response = messages["answer"]

            new_messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_content}]}
            )

            new_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_response}],
                }
            )

            return new_messages

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
                # convert regular messages format to Messages format with content type
                # for compatibility with apply_chat_template
                processed_example["messages"] = convert_messages_to_multimedia_messages(
                    processed_example["messages"]
                )
            elif "question" in example:  # Multiple choice format
                processed_example = {}
                processed_example["messages"] = (
                    convert_multiple_choice_to_multimedia_messages(example)
                )
            else:  # Legacy format
                processed_example = convert_legacy_format(example)
                processed_example["messages"] = convert_messages_to_multimedia_messages(
                    processed_example["messages"]
                )

            # find the image key if it exists

            image_keys = []
            for key in example.keys():
                if "image" in key:
                    image_keys.append(key)

            for im_key in image_keys:
                if example[im_key] is None:
                    continue
                if isinstance(example[im_key], list):
                    if len(example[im_key] == 0):
                        continue
                    image_value = example[im_key][0]
                else:
                    image_value = example[im_key]

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
        "llava",
        "mistral_v7_tekken",
        "pixtral",
    ]:
        return ProcessingStrategy(
            processor, chat_template, image_size, image_resize_algorithm
        )
    raise ValueError(f"Unsupported chat template type: {chat_template_type}")
