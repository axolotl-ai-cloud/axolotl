"""
Collators for multi-modal chat messages and packing
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
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

        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(
                example["messages"], chat_template=chat_template, tokenize=False
            )
            for example in examples
        ]
        images = [
            Image.open(example["images"])
            if isinstance(example["images"], str)
            else example["images"]
            for example in examples
        ]

        if max_images > 0:
            images = [img_batch[:max_images] for img_batch in images]

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
