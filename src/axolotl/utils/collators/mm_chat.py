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
    sequence_length: Optional[int] = None
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
        if self.packing:
            return self.__class__.process_rows_packing(
                examples, self.processor, self.chat_template, self.max_images, self.sequence_length
            )

        return self.__class__.process_rows(
            examples, self.processor, self.chat_template, self.max_images
        )

    @staticmethod
    def process_rows_packing(examples, processor, chat_template, max_images, sequence_length, length_only=False):
        import torch
        # Perform sample packing within a batch

        if processor.tokenizer.sep_token is None:
            sep_token = '[SEP]'
            processor.tokenizer.add_tokens([sep_token])
            processor.tokenizer.sep_token = sep_token
        sep_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.sep_token)
        pad_token_id = processor.tokenizer.pad_token_id
        
        texts = [
            processor.apply_chat_template(
                example["messages"], chat_template=chat_template, tokenize=False
            )
            for example in examples
        ]
        images = [example["images"] for example in examples]

        batch = processor(text=texts, images=images, padding=False)
        
        n_sequence = len(examples)
        n = 0
        pack_len = 0
        features_pack = {}
        packed = {}
        features = list[batch.keys()]
        for feature in features:
            features_pack[feature] = []
            packed[feature] = []
        features.remove("input_ids")

        for ii in range(n_sequence):
            next_seq_len = len(batch["input_ids"][ii])
            if not pack_len + next_seq_len + 1 < sequence_length:
                n += 1
                pack_len += next_seq_len + 1
                features_pack["input_ids"] += batch["input_ids"][ii] + [sep_token_id]

                '''
                Do something with attention mask and cross-attention
                '''

                for feature in features:
                    features_pack[feature] += batch[feature][ii]

            else:
                for _ in range(sequence_length - pack_len):
                    features_pack["input_ids"] += [pad_token_id]

                packed["input_ids"].append(torch.tensor(features_pack["input_ids"].copy()))

                for feature in features:
                    packed[feature].append(torch.tensor(features_pack[feature].copy()))
                    features_pack[feature] = []

                pack_len = 0

        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )
        labels = [pack.clone() for pack in packed["input_ids"]]
        for ii , label in enumerate(labels):
            labels[ii][label == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        
            labels[ii][label == image_token_id] = -100
        packed["labels"] = labels

        if length_only:
            return {
                "length": [len(sample["input_ids"]) for sample in batch["input_ids"]]
            }
        return packed
    
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
