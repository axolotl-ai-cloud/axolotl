import json
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy
from typing import Any, Optional, Union

from axolotl.processing_strategies import ProcessingStrategy


@dataclass
class MultiModalChatDataCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    processing_strategy: ProcessingStrategy
    packing: bool = False
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None

    @staticmethod
    def _to_text_dict(ex: Any) -> dict:
        """
        Convert input into a dict with a 'messages' or 'conversations' field if possible.
        If input is not a dict or lacks these fields, convert it into a minimal dict format.
        """
        if isinstance(ex, dict):
            if ("messages" in ex and isinstance(ex["messages"], list)) or (
                "conversations" in ex and isinstance(ex["conversations"], list)
            ):
                return ex
        # If input is a string, try to parse as JSON, else fallback to plain text
        try:
            ex = json.loads(ex) if not isinstance(ex, dict) else ex
        except Exception:
            pass
        # Fallback: wrap input as a single-message dict
        if not isinstance(ex, dict) or (
            "messages" not in ex and "conversations" not in ex
        ):
            ex = {"messages": [{"role": "user", "content": str(ex)}]}
        return ex

    def torch_call(self, examples: list[Any]):
        """
        Main entry point for data collation.
        """
        return self.process_rows(examples)

    def process_rows(self, examples: list[Any]):
        """
        Process each example in the batch:
        - Normalize each item to a dict (with messages/conversations)
        - Apply processing strategy
        - Prepare tensors for input_ids, attention_mask, (optional) audio features
        - Apply padding and final formatting for model input
        """
        # Normalize all examples to dict format
        raw_examples = [self._to_text_dict(e) for e in examples]
        # Process raw examples through the main processing strategy
        processed_examples = self.processing_strategy(raw_examples)

        # Check if every example contains audio
        has_audio = all("audio" in e for e in raw_examples)
        batch = {"input_ids": [], "attention_mask": []}
        if has_audio:
            batch["input_audio_embeds"] = []
            batch["audio_attention_mask"] = []

        # Tokenize and collate each example
        for txt_ex, raw_ex in zip(processed_examples, raw_examples):
            # Apply chat template and tokenize the textual content
            result = self.processing_strategy.processor.apply_chat_template(
                raw_ex,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
                chat_template=self.processing_strategy.chat_template,
            )
            batch["input_ids"].append(result["input_ids"].squeeze(0))
            batch["attention_mask"].append(result["attention_mask"].squeeze(0))

            if has_audio:
                audio = raw_ex["audio"]
                audio_embeds = self.processing_strategy.processor.audio_processor(
                    audios=[(audio["array"], audio["sampling_rate"])],
                    return_tensors="pt",
                )
                batch["input_audio_embeds"].append(
                    audio_embeds["input_audio_embeds"].squeeze(0)
                )
                batch["audio_attention_mask"].append(
                    audio_embeds["audio_attention_mask"].squeeze(0)
                )
        pad = torch.nn.utils.rnn.pad_sequence

        # Pad text inputs
        input_ids = pad(
            batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad(batch["attention_mask"], batch_first=True, padding_value=0)

        # Process labels as required by the strategy
        labels = self.processing_strategy.process_labels(input_ids)

        final = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Pad and collate audio inputs if present
        if has_audio:
            final["input_audio_embeds"] = pad(
                batch["input_audio_embeds"], batch_first=True, padding_value=0.0
            )
            final["audio_attention_mask"] = pad(
                batch["audio_attention_mask"], batch_first=True, padding_value=0
            )

        return final
