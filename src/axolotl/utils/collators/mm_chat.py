"""
Collators for multi-modal chat messages and packing
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Union

from PIL import Image
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy


@dataclass
class MultiModalChatDataCollator(DataCollatorMixin):
    """
    Collator for multi-modal chat messages
    """

    tokenizer: PreTrainedTokenizerBase
    processor: ProcessorMixin
    return_tensors: str = "pt"
    chat_template: Optional[str] = None
    packing: bool = False
    max_images: int = -1
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.packing:
            raise ValueError("Packing is currently not supported.")

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.

        return self.__class__.process_rows(
            examples, self.processor, self.chat_template, self.max_images
        )

    @staticmethod
    def process_rows(examples, processor, chat_template, max_images, length_only=False):
        # HINT: use `_torch_collate_batch` to stack and pad tensors
        # see also DataCollatorWithFlattening and DefaultDataCollator

        # *** This is COPIED from the trl example sft_vlm.py code ***
        # use this as a starting point

        def _preprocess(examples: list[dict]) -> list[dict]:
            """
            Preprocess conversation examples to ensure consistent format.

            Converts different conversation formats to OpenAI format with 'messages'.
            Supports two formats:
            1. OpenAI format with 'messages'
            2. Legacy format with 'conversations'

            Args:
                examples: list of conversation dictionaries

            Returns:
                dict in OpenAI format with 'messages' key

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
                    {
                        "role": normalize_role(convo["from"]),
                        "content": convo["value"],
                    }
                    for convo in example["conversations"]
                ]

                # Create new dict without 'conversations' key
                result = deepcopy(example)
                result.pop("conversations")
                return {"messages": messages, **result}

            processed_examples = []
            for example in examples:
                # OpenAI format
                if "messages" in example:
                    processed_examples.append(example)

                # Legacy format
                elif "conversations" in example:
                    processed_examples.append(convert_legacy_format(example))

                else:
                    raise ValueError(
                        "Only `messages` and `conversations` message keys are currently supported."
                    )

            return processed_examples

        def _process_images(examples, max_images):
            """
            Process images from examples, ensuring consistency in image presence and applying max_images limit.

            Args:
                examples: List of dictionaries that may contain 'images' key
                max_images: Maximum number of images to keep per example (0 means no limit)

            Returns:
                Either None (if no images) or List[Image objects] (if all examples have images)

            Raises:
                ValueError: If there's a mix of None and non-None images
            """

            def get_image(example):
                if "images" not in example:
                    return None
                images = example["images"]
                if isinstance(images, str):
                    return Image.open(images)
                return images

            images = [get_image(example) for example in examples]

            # Count None and non-None images
            none_count = sum(1 for img in images if img is None)

            # All images are None
            if none_count == len(images):
                return None

            # Mix of None and non-None images
            if none_count > 0:
                raise ValueError(
                    "All images should be either None or not None. "
                    "Please provide images for all examples or None."
                )

            # Apply max_images limit if specified
            if max_images > 0:
                images = [
                    (
                        img_batch[:max_images]
                        if isinstance(img_batch, (list, tuple))
                        else img_batch
                    )
                    for img_batch in images
                ]

            return images

        # Preprocess the examples
        examples = _preprocess(examples)

        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(
                example["messages"], chat_template=chat_template, tokenize=False
            )
            for example in examples
        ]

        images = _process_images(examples, max_images=max_images)

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        if length_only:
            return {
                "length": [len(sample["input_ids"]) for sample in batch["input_ids"]]
            }
        return batch
