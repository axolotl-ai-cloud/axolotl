"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

from copy import deepcopy
from typing import Optional

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch import Tensor, zeros_like
from transformers import ProcessorMixin
from transformers.image_utils import load_image
from transformers.models.internvl import InternVLProcessor
from transformers.models.smolvlm import SmolVLMProcessor
from transformers.models.voxtral import VoxtralProcessor

from axolotl.utils.dict import remove_none_values
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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
            if (
                "messages" in example and example["messages"] is not None
            ):  # OpenAI format
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

            # if the image key exists, add the image to the first user message
            if image_key is not None and processed_example[image_key] is not None:
                # TODO: check if it's normal to be single image only for common datasets
                # From observation, it's usually a list of single image but some datasets may have several columns for images
                # Temporary solution: take the first image and suggest people convert their datasets to use multi-content Messages
                if len(processed_example[image_key]) > 1:
                    LOG.warning(
                        f"Found {len(processed_example[image_key])} images in a sample. Using the first one."
                        "If you are using a dataset with multiple images per sample, please convert it to use multi-content Messages."
                        "See https://docs.axolotl.ai/docs/multimodal.html#dataset-format"
                    )

                image_value = processed_example[image_key][0]

                # Handle image loading (Image, url, path, base64)
                image_value = load_image(image_value)

                if self.image_size is not None:
                    assert hasattr(image_value, "resize"), (
                        "Image does not have a resize method"
                    )

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
                msg_ind_to_add = None
                ind_to_add = None
                first_user_idx = None

                for msg_idx, msg_content in enumerate(processed_example["messages"]):
                    if first_user_idx is None and msg_content["role"] == "user":
                        first_user_idx = msg_idx
                    for i, content in enumerate(
                        processed_example["messages"][msg_idx]["content"]
                    ):
                        # Usually datasets created with image columns, don't have it in the messages itself
                        if content["type"] == "image" and all(
                            k not in content for k in ["image", "url", "path", "base64"]
                        ):
                            msg_ind_to_add = msg_idx
                            ind_to_add = i
                            break

                # If an image type is found, add the image to that index
                if ind_to_add is not None and msg_ind_to_add is not None:
                    processed_example["messages"][msg_ind_to_add]["content"][
                        ind_to_add
                    ]["image"] = image_value
                else:
                    # if no image type is found, add it to end of the first user message
                    if first_user_idx is None:
                        first_user_idx = 0
                    processed_example["messages"][first_user_idx]["content"].append(
                        {
                            "type": "image",
                            "image": image_value,
                        }
                    )

            processed_examples.append(remove_none_values(processed_example))

        return processed_examples

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        """
        Mask non assistant regions to -100.
        To be implemented per subclass.
        """
        return labels

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = input_ids.clone()

        labels = self._mask_non_assistant(labels)

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


class Gemma3nProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Gemma3n"""

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        def _find_token_sequence(label, start_pos, token_sequence):
            """Check if token_sequence appears at start_pos in label"""
            if start_pos + len(token_sequence) > len(label):
                return False
            if label[start_pos] != token_sequence[0]:
                return False
            return (
                label[start_pos : start_pos + len(token_sequence)].tolist()
                == token_sequence
            )

        def _find_assistant_end(label, start_pos, assistant_end_tok, mask, i):
            """
            Find the end of assistant response and update mask accordingly

            Returns new position to continue from and whether the end seq is found
            """
            k = start_pos
            while k < len(label):
                if not _find_token_sequence(label, k, assistant_end_tok):
                    mask[i][k] = 1
                    k += 1
                    continue

                return k + len(assistant_end_tok), True

            return k, False

        mask = zeros_like(labels)

        assistant_start_str = "<start_of_turn>model"
        assistant_end_str = "<end_of_turn>"
        include_assistant_start_tok = False
        include_assistant_end_tok = True

        # str to tokens
        assistant_start_tok = self.processor.tokenizer.encode(
            assistant_start_str, add_special_tokens=False
        )
        assistant_end_tok = self.processor.tokenizer.encode(
            assistant_end_str, add_special_tokens=False
        )

        for i, label in enumerate(labels):
            j = 0
            # while loop through each tok index in labels[i]
            while j < len(label):
                # Check until match start seq
                if not _find_token_sequence(label, j, assistant_start_tok):
                    j += 1
                    continue

                if include_assistant_start_tok:
                    mask[i][j : j + len(assistant_start_tok)] = 1

                # Find where the assistant response ends
                start_of_content = j + len(assistant_start_tok)
                end_pos, found_end_seq = _find_assistant_end(
                    label, start_of_content, assistant_end_tok, mask, i
                )

                # Include end token if requested
                if include_assistant_end_tok and found_end_seq:
                    mask[i][end_pos - len(assistant_end_tok) : end_pos] = 1

                j = end_pos

            labels[i][mask[i] == 0] = -100

        return labels

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        # Follows https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_t4.ipynb
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        if hasattr(self.processor.tokenizer, "image_token_id"):
            labels[labels == self.processor.tokenizer.image_token_id] = -100
        if hasattr(self.processor.tokenizer, "audio_token_id"):
            labels[labels == self.processor.tokenizer.audio_token_id] = -100
        if hasattr(self.processor.tokenizer, "boi_token_id"):
            labels[labels == self.processor.tokenizer.boi_token_id] = -100
        if hasattr(self.processor.tokenizer, "eoi_token_id"):
            labels[labels == self.processor.tokenizer.eoi_token_id] = -100

        return labels


class VoxtralProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Voxtral"""

    def __init__(
        self,
        processor: VoxtralProcessor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids
        )

        self.audio_token = special_ids.audio
        self.begin_audio_token = special_ids.begin_audio

    def process_labels(self, input_ids):
        labels = input_ids.clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.audio_token] = -100
        labels[labels == self.begin_audio_token] = -100

        return labels


class SmolVLM2ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for SmolVLM2"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.image_token = "<image>"  # nosec

        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index(self.image_token)
        ]


class Mistral3ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Mistral3"""

    def __init__(
        self,
        processor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.image_encoder.special_ids
        )

        self.image_token = special_ids.img
        self.image_break_token = special_ids.img_break
        self.image_end_token = special_ids.img_end

    def process_labels(self, input_ids):
        labels = input_ids.clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token] = -100
        labels[labels == self.image_break_token] = -100
        labels[labels == self.image_end_token] = -100

        return labels


class InternVLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for InternVL"""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)

        if not hasattr(processor, "image_ids"):
            raise ValueError("'image_ids' missing from InternVL Processor.")

        self.image_token_ids = processor.image_ids

    def process_labels(self, input_ids):
        labels = input_ids.clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        for ids in self.image_token_ids:
            labels[labels == ids] = -100

        # Note: Check if need to mask 'video_token' as it gets converted to
        # image patches during media processing

        return labels


def get_processing_strategy(
    processor: ProcessorMixin,
    chat_template,
    chat_template_type,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Resampling | None = None,
):
    from axolotl.utils.mistral.mistral3_processor import Mistral3Processor

    processing_kwargs = {
        "processor": processor,
        "chat_template": chat_template,
        "image_size": image_size,
        "image_resize_algorithm": image_resize_algorithm,
    }

    if chat_template_type in [None, "tokenizer_default"] and hasattr(
        processor.tokenizer, "chat_template"
    ):
        processing_kwargs["chat_template"] = processor.tokenizer.chat_template

    if chat_template_type == "qwen2_vl":
        return Qwen2VLProcessingStrategy(
            **processing_kwargs,
        )
    if chat_template_type == "gemma3":
        return Gemma3ProcessingStrategy(
            **processing_kwargs,
        )
    if chat_template_type == "gemma3n":
        return Gemma3nProcessingStrategy(
            **processing_kwargs,
        )

    if isinstance(processor, VoxtralProcessor):
        return VoxtralProcessingStrategy(
            **processing_kwargs,
        )

    if isinstance(processor, SmolVLMProcessor):
        return SmolVLM2ProcessingStrategy(
            **processing_kwargs,
        )

    if isinstance(processor, Mistral3Processor):
        return Mistral3ProcessingStrategy(
            **processing_kwargs,
        )

    if isinstance(processor, InternVLProcessor):
        return InternVLProcessingStrategy(
            **processing_kwargs,
        )

    # llama3_2_vision, llama4, llava
    # mistral_v7_tekken, pixtral, lfm2vl
    return ProcessingStrategy(
        **processing_kwargs,
    )
