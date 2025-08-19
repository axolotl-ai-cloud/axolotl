from dataclasses import dataclass
from typing import Any, List

import torch
from transformers import PreTrainedTokenizerBase, default_data_collator
from transformers.utils import PaddingStrategy

from axolotl.prompters import Prompter
from axolotl.utils.dict import DictDefault


@dataclass
class StreamingDataCollator:
    tokenizer: PreTrainedTokenizerBase
    cfg: DictDefault
    prompter: Prompter | None = None
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.max_length is None:
            self.max_length = self.cfg.sequence_len

    def __call__(self, raw_batch: List[dict]) -> dict[str, Any]:
        processed_samples = []

        for raw_sample in raw_batch:
            formatted_sample = raw_sample
            if self.prompter:
                formatted_sample = self._apply_prompt_formatting(raw_sample)

            tokenized_sample = self._tokenize_sample(formatted_sample)

            if len(tokenized_sample["input_ids"]) > self.max_length:
                tokenized_sample = self._truncate_sample(tokenized_sample)

            if tokenized_sample.get("input_ids"):
                processed_samples.append(tokenized_sample)

        return self._pad_and_batch(processed_samples)

    def _apply_prompt_formatting(self, raw_sample: dict) -> dict:
        formatted_text = self.prompter.build_prompt(
            instruction=raw_sample.get("instruction", ""),
            input=raw_sample.get("input", ""),
            output=raw_sample.get("output", ""),
        )
        return {"text": formatted_text}

    def _tokenize_sample(self, sample: dict) -> dict:
        text = sample.get("text", sample.get("content", ""))

        if not text:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output_text = sample.get("output", "")

            parts = []
            if instruction:
                parts.append(f"Instruction: {instruction}")
            if input_text:
                parts.append(f"Input: {input_text}")
            if output_text:
                parts.append(f"Output: {output_text}")
            text = "\n".join(parts)

        if not text:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        tokenized = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def _truncate_sample(self, tokenized_sample: dict) -> dict:
        max_len = self.max_length
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in tokenized_sample:
                tokenized_sample[key] = tokenized_sample[key][:max_len]
        return tokenized_sample

    def _pad_and_batch(self, processed_samples: List[dict]) -> dict[str, Any]:
        if not processed_samples:
            processed_samples = [
                {
                    "input_ids": [self.tokenizer.eos_token_id],
                    "attention_mask": [1],
                    "labels": [self.tokenizer.eos_token_id],
                }
            ]

        batch_samples = []
        for sample in processed_samples:
            batch_sample = {}
            for key, value in sample.items():
                if key in ["input_ids", "attention_mask", "labels"]:
                    batch_sample[key] = torch.tensor(value, dtype=torch.long)
            batch_samples.append(batch_sample)

        if self.padding:
            max_len_in_batch = max(len(sample["input_ids"]) for sample in batch_samples)

            for sample in batch_samples:
                current_len = len(sample["input_ids"])
                pad_len = max_len_in_batch - current_len

                if pad_len > 0:
                    pad_token_id = (
                        self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )

                    sample["input_ids"] = torch.cat(
                        [
                            sample["input_ids"],
                            torch.full((pad_len,), pad_token_id, dtype=torch.long),
                        ]
                    )
                    sample["attention_mask"] = torch.cat(
                        [
                            sample["attention_mask"],
                            torch.zeros(pad_len, dtype=torch.long),
                        ]
                    )
                    sample["labels"] = torch.cat(
                        [
                            sample["labels"],
                            torch.full(
                                (pad_len,), self.label_pad_token_id, dtype=torch.long
                            ),
                        ]
                    )

        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch_samples[0]:
                batch[key] = torch.stack([sample[key] for sample in batch_samples])

        return batch
