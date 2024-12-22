"""
Collators for multi-modal chat messages and packing
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy
from .mm_processing_strategies import ProcessingStrategy


@dataclass
class MultiModalChatDataCollator(DataCollatorMixin):
    """
    Collator for multi-modal chat messages
    """

    tokenizer: PreTrainedTokenizerBase
    processing_strategy: ProcessingStrategy
    return_tensors: str = "pt"
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
            examples,
            self.processing_strategy,
            self.max_images,
        )

    @staticmethod
    def process_rows(
        examples,
        processing_strategy: ProcessingStrategy,
        max_images,
        length_only=False,
    ):
        # HINT: use `_torch_collate_batch` to stack and pad tensors
        # see also DataCollatorWithFlattening and DefaultDataCollator

        # *** This is COPIED from the trl example sft_vlm.py code ***
        # use this as a starting point

        # Preprocess the examples
        examples = processing_strategy.preprocess(examples)

        # Get the texts and images, and apply the chat template
        texts = processing_strategy.process_texts(examples)
        images = processing_strategy.process_images(examples, max_images)

        # Tokenize the texts and process the images
        batch = processing_strategy.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processing_strategy.processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        labels[labels == processing_strategy.image_token_id] = -100
        batch["labels"] = labels

        if length_only:
            return {
                "length": [len(sample["input_ids"]) for sample in batch["input_ids"]]
            }
        return batch
