"""
Collators for multi-modal chat messages and packing
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

from axolotl.processing_strategies import ProcessingStrategy


@dataclass
class MultiModalChatDataCollator(DataCollatorMixin):
    """
    Collator for multi-modal chat messages
    """

    tokenizer: PreTrainedTokenizerBase
    processing_strategy: ProcessingStrategy
    packing: bool = False
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.packing:
            raise ValueError("Packing is currently not supported.")

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        return self.process_rows(examples)

    def process_rows(
        self,
        examples: list[dict],
    ) -> dict[str, Tensor]:
        # Preprocess the examples
        examples = self.processing_strategy(examples)

        # Initialize batch
        messages = [ex["messages"] for ex in examples]

        batch = self.processing_strategy.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            chat_template=self.processing_strategy.chat_template,
        )

        # Process the labels
        batch["labels"] = self.processing_strategy.process_labels(batch["input_ids"])

        return batch
