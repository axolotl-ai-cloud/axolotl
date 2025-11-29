"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

import json
from copy import deepcopy
from typing import Optional

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch import Tensor, zeros_like
from transformers import ProcessorMixin
from transformers.image_utils import load_image
from transformers.models.smolvlm import SmolVLMProcessor
from transformers.models.voxtral import VoxtralProcessor

from axolotl.utils.dict import remove_none_values
from axolotl.utils.logging import get_logger
from axolotl.utils.mistral.mistral3_processor import Mistral3Processor

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
        """
        Initialize the processing strategy with a processor, optional chat template, and image resizing configuration.
        
        Parameters:
            processor (ProcessorMixin): The processor/tokenizer wrapper used for token and image handling.
            chat_template (Optional[str]): Optional chat template identifier or content to guide message formatting.
            image_size (int | tuple[int, int] | None): Target image size; an int indicates square padding to that size, a (width, height) tuple specifies explicit resize dimensions, and None disables resizing.
            image_resize_algorithm (Resampling | None): PIL resampling algorithm to use when resizing images; defaults to Image.Resampling.BILINEAR if not provided.
        
        Notes:
            - The instance attribute `supports_multi_images` is initialized to False and may be set to True by subclasses that support multiple images.
            - If the provided processor exposes an `image_token` attribute, this token is stored on the instance and its token id is computed and stored as `image_token_id`.
        """
        self.processor = processor
        self.chat_template = chat_template
        self.image_token = None
        self.image_token_id = None
        self.supports_multi_images = False  # Override in subclasses that support multiple images

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
        Normalize and preprocess conversation examples into a unified multimedia OpenAI-like `messages` format.
        
        Converts legacy `conversations` entries to `messages`, normalizes each message's content into typed content blocks (e.g., text or image), deserializes JSON-encoded message content when multi-image Qwen2-VL processing is enabled, loads and optionally resizes image values, and attaches loaded images into message content (mapping to image placeholders when present or appending to the first user message). Multi-image handling is controlled by the instance's `supports_multi_images` attribute.
        
        Parameters:
            examples (list[dict]): Input conversation examples. Each example must contain either a `messages` key (OpenAI format) or a `conversations` key (legacy format). If present, an `images` or `image` key may contain one or more image references (PIL Image, URL, path, or base64) to be loaded.
        
        Returns:
            list[dict]: Processed examples where each example contains a `messages` key with content items normalized to typed content blocks (e.g., {"type": "text", "text": ...} or {"type": "image", "image": <PIL.Image>}), and where image values have been loaded and resized according to the instance configuration.
        
        Raises:
            ValueError: If an example contains neither `messages` nor `conversations`.
        """
        role_mapping = {
            "human": "user",
            "gpt": "assistant",
        }

        def normalize_role(role: str) -> str:
            """Normalize role names to OpenAI format. Default to original role if not found."""
            return role_mapping.get(role, role)

        def convert_legacy_format(example: dict) -> dict:
            """
            Convert a legacy example using a 'conversations' list into an OpenAI-style 'messages' list.
            
            Parameters:
                example (dict): Input example containing a "conversations" key where each item has "from" and "value".
            
            Returns:
                dict: A shallow copy of `example` with the "conversations" key removed and a "messages" key added.
                      Each message is a dict with "role" (normalized from the conversation "from") and "content" (the conversation "value").
            """
            messages = [
                {"role": normalize_role(convo["from"]), "content": convo["value"]}
                for convo in example["conversations"]
            ]

            # Create new dict without 'conversations' key
            result = deepcopy(example)
            result.pop("conversations")
            result["messages"] = messages
            return result

        def convert_messages_to_multimedia_messages(messages: list[dict], is_qwen2_vl: bool = False) -> list[dict]:
            """
            Normalize a sequence of messages into a multimedia-aware messages format.
            
            Parameters:
                messages (list[dict]): Input messages where each item contains at least "role" and "content".
                    Content may be a string or a list of content blocks.
                is_qwen2_vl (bool): If True, attempt to JSON-deserialize string content that begins with '[' or '{'
                    so JSON-encoded mixed content becomes native Python structures; invalid JSON remains a string.
            
            Returns:
                list[dict]: A list of messages where each message has the same "role" and a "content" value that is a
                list of content blocks. Plain string content is converted to a single text block:
                    {"type": "text", "text": <original string>}.
                If the original content is already a list, it is returned unchanged.
            """

            new_messages = []
            for message in messages:
                content = message["content"]
                
                # Only try to deserialize JSON-encoded content for qwen2_vl models
                # This is because we normalized mixed content to JSON strings during loading
                if is_qwen2_vl and isinstance(content, str) and (content.startswith('[') or content.startswith('{')):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        # Not JSON, treat as regular string
                        pass
                
                if isinstance(content, str):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": content,
                                }
                            ],
                        }
                    )
                elif isinstance(content, list):
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
            # Check if this model supports multi-images and needs special handling
            processed_example["messages"] = convert_messages_to_multimedia_messages(
                processed_example["messages"], is_qwen2_vl=self.supports_multi_images
            )

            # find the image key if it exists
            possible_image_keys = ["images", "image"]
            image_key = None
            for key in possible_image_keys:
                if key in processed_example:
                    image_key = key
                    break

            # if the image key exists, add the images to the message
            if image_key is not None and processed_example[image_key] is not None:
                # Check if we should handle multiple images
                # Debug logging
                if len(processed_example[image_key]) > 1:
                    LOG.debug(f"Multiple images detected. Strategy type: {type(self).__name__}, supports_multi_images={self.supports_multi_images}")
                
                if self.supports_multi_images and len(processed_example[image_key]) > 1:
                    # Qwen2-VL: Load all images
                    loaded_images = []
                    for img in processed_example[image_key]:
                        loaded_img = load_image(img)
                        loaded_images.append(loaded_img)
                    
                    # Log multi-image usage for debugging
                    LOG.debug(f"Processing {len(loaded_images)} images in sample for Qwen2-VL")
                else:
                    # Original behavior: take first image and warn if multiple
                    if len(processed_example[image_key]) > 1:
                        LOG.warning(
                            f"Found {len(processed_example[image_key])} images in a sample. Using the first one."
                            "If you are using a dataset with multiple images per sample, please convert it to use multi-content Messages."
                            "See https://docs.axolotl.ai/docs/multimodal.html#dataset-format"
                        )
                    
                    image_value = processed_example[image_key][0]
                    # Handle image loading (Image, url, path, base64)
                    image_value = load_image(image_value)
                    loaded_images = [image_value]

                # Resize all loaded images if needed
                if self.image_size is not None:
                    resized_images = []
                    for image_value in loaded_images:
                        assert hasattr(image_value, "resize"), (
                            "Image does not have a resize method"
                        )

                        if isinstance(self.image_size, tuple):
                            resized_img = image_value.resize(
                                self.image_size, self.image_resize_algorithm
                            )
                        else:
                            # Set the padding value; here we use black (0, 0, 0) for RGB images
                            padding_color = (0, 0, 0)

                            # When image_size is an int (square target), preserve aspect ratio then pad
                            # This is to prevent aspect ratio distortion when resizing to square
                            resized_img = ImageOps.pad(
                                image_value,
                                (self.image_size, self.image_size),
                                method=self.image_resize_algorithm,
                                color=padding_color,
                            )
                        resized_images.append(resized_img)
                    loaded_images = resized_images

                # Look for image placeholders in messages
                if self.supports_multi_images and len(loaded_images) > 1:
                    # Qwen2-VL: Map multiple images to their placeholders
                    image_placeholders = []
                    first_user_idx = None

                    for msg_idx, msg_content in enumerate(processed_example["messages"]):
                        if first_user_idx is None and msg_content["role"] == "user":
                            first_user_idx = msg_idx
                        for i, content in enumerate(
                            processed_example["messages"][msg_idx]["content"]
                        ):
                            # Find image placeholders
                            if content["type"] == "image" and all(
                                k not in content for k in ["image", "url", "path", "base64"]
                            ):
                                image_placeholders.append((msg_idx, i))

                    # Map loaded images to placeholders
                    if image_placeholders:
                        # If we have placeholders, map images to them in order
                        for idx, (msg_idx, content_idx) in enumerate(image_placeholders):
                            if idx < len(loaded_images):
                                processed_example["messages"][msg_idx]["content"][content_idx]["image"] = loaded_images[idx]
                    else:
                        # If no placeholders found, add all images to end of first user message
                        if first_user_idx is None:
                            first_user_idx = 0
                        for image_value in loaded_images:
                            processed_example["messages"][first_user_idx]["content"].append(
                                {
                                    "type": "image",
                                    "image": image_value,
                                }
                            )
                else:
                    # Original single image behavior
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
                        ]["image"] = loaded_images[0]
                    else:
                        # if no image type is found, add it to end of the first user message
                        if first_user_idx is None:
                            first_user_idx = 0
                        processed_example["messages"][first_user_idx]["content"].append(
                            {
                                "type": "image",
                                "image": loaded_images[0],
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
        """
        Initialize a Qwen2-VL-specific processing strategy and configure multi-image and image token settings.
        
        Sets supports_multi_images to True, sets the image token placeholder to "<|image_pad|>", and computes its token id from the provided processor's tokenizer.
        
        Parameters:
            processor (ProcessorMixin): Tokenizer/processor used to derive special token ids.
            chat_template (Optional[str]): Optional chat template identifier or content.
            image_size (int | tuple[int, int] | None): Optional target image size used for resizing.
            image_resize_algorithm (Resampling | None): Optional resampling algorithm for image resizing.
        """
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.supports_multi_images = True  # Qwen2-VL supports multiple images
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
        processor: Mistral3Processor,
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


def get_processing_strategy(
    processor: ProcessorMixin,
    chat_template,
    chat_template_type,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Resampling | None = None,
):
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

    # llama3_2_vision, llama4, llava
    # mistral_v7_tekken, pixtral, lfm2vl
    return ProcessingStrategy(
        **processing_kwargs,
    )