from copy import deepcopy
from typing import Optional

from PIL import Image
from transformers import ProcessorMixin


class ProcessingStrategy:
    def __init__(self, processor: ProcessorMixin, chat_template: Optional[str] = None):
        self.processor = processor
        self.chat_template = chat_template
        try:
            self.image_token = processor.image_token
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
                self.image_token
            )
        except AttributeError:
            pass

    @staticmethod
    def preprocess(examples: list[dict]) -> list[dict]:
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
            result["messages"] = messages
            return result

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

    @staticmethod
    def process_images(examples, max_images):
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

    def process_texts(self, examples):
        texts = [self.processor.apply_chat_template(example["messages"], chat_template=self.chat_template, tokenize=False)for example in examples]
        return texts


class PixtralProcessingStrategy(ProcessingStrategy):
    @staticmethod
    def pixtral_chat_conversion(messages):
        is_single_message = not isinstance(messages, list)
        if is_single_message:
            messages = [messages]

        for i, message in enumerate(messages):
            if message["role"] == "user":
                for j, content in enumerate(message["content"]):
                    if "type" in content and content["type"] == "text":
                        messages[i]["content"][j] = {
                            "type": "text",
                            "content": content["text"],
                        }

            if message["role"] == "assistant":
                messages[i]["content"] = message["content"][0]["text"]

        if is_single_message:
            return messages[0]
        return messages

    def process_texts(self, examples):
        texts = [
            self.processor.apply_chat_template(__class__.pixtral_chat_conversion(example["messages"]), chat_template=self.chat_template, tokenize=False,)
            for example in examples
        ]
        return texts


class Qwen2VLProcessingStrategy(ProcessingStrategy):

    def __init__(self, processor: ProcessorMixin, chat_template: Optional[str] = None):
        super().__init__(processor, chat_template)
        self.image_token = "<|image_pad|>"


class LlavaProcessingStrategy(ProcessingStrategy):

    @staticmethod
    def process_images(examples, max_images):
        images = ProcessingStrategy.process_images(examples, max_images)
        images = [image[0] for image in images]
        return images


def get_processing_strategy(processor: ProcessorMixin, chat_template, chat_template_type):
    if chat_template_type == "pixtral":
        return PixtralProcessingStrategy(processor, chat_template)
    if chat_template_type == "llava":
        return LlavaProcessingStrategy(processor, chat_template)
    return ProcessingStrategy(processor, chat_template)