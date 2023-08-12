"""Module containing Dataset functionality"""

import logging
import os
from typing import List

import torch
from datasets import Dataset, IterableDataset

from .prompt_tokenizers import PromptTokenizingStrategy

# We want this to be a wrapper for an existing dataset that we have loaded
# lets use the concept of middlewares to wrap each dataset, for example
# ConstantLengthDataset(ShuffledDataset([TokenizedPromptDataset(alpaca_dataset)]))
# let's check to ensure we don't truncate an item in the middle, we'll use
# the collators later on to pad the datasets

LOG = logging.getLogger("axolotl")


class TokenizedPromptDataset(Dataset):
    """
    Dataset that returns tokenized prompts from a stream of text files.
        Args:
            prompt_tokenizer (PromptTokenizingStrategy): The prompt tokenizing method for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: IterableDataset,
        **kwargs,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        super().__init__(self.process(dataset).data, **kwargs)

    def process(self, dataset):
        features = dataset.features.keys()
        num_proc = min(64, os.cpu_count())
        return dataset.map(
            self.prompt_tokenizer.tokenize_prompt,
            num_proc=num_proc,
            remove_columns=features,
        )


# TODO this isn't the best since it can't interleave datasets
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            seq_length (int): Length of token sequences to return.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        tokenizer,
        datasets,
        seq_length=2048,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.datasets: List[IterableDataset] = datasets
        self.seq_length = seq_length

        vocab_size = len(tokenizer.get_vocab())

        if vocab_size <= torch.iinfo(torch.int16).max:
            self.tokens_dtype = torch.int16
        elif vocab_size <= torch.iinfo(torch.int32).max:
            self.tokens_dtype = torch.int32
        else:
            self.tokens_dtype = torch.int64

    def __iter__(self):
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "position_ids": [],
        }
        buffer_len = 0
        for dataset in self.datasets:
            idx = 0
            iterator = iter(dataset)
            more_examples = True
            while more_examples:
                try:
                    example = next(iterator)
                    idx += 1
                except StopIteration:
                    more_examples = False
                    example = None

                add_concat_token = False
                if example:
                    example_len = len(example["input_ids"])
                    add_concat_token = example["input_ids"][-1] != self.concat_token_id
                else:
                    example_len = 0

                if not example_len or (
                    buffer_len + int(add_concat_token) + example_len > self.seq_length
                ):
                    if buffer["input_ids"]:
                        input_ids = torch.cat(buffer["input_ids"], dim=-1)[
                            : self.seq_length
                        ]
                        attention_mask = torch.cat(buffer["attention_mask"], dim=-1)[
                            : self.seq_length
                        ]
                        position_ids = torch.cat(buffer["position_ids"], dim=-1)[
                            : self.seq_length
                        ]
                        labels = torch.cat(buffer["labels"], dim=-1)[: self.seq_length]
                        if labels.size() == input_ids.size() and (
                            attention_mask.size() == input_ids.size()
                        ):
                            yield {
                                "input_ids": input_ids,
                                "labels": labels,
                                "attention_mask": attention_mask,
                                "position_ids": position_ids,
                            }
                        else:
                            LOG.warning(
                                f"dropping batch due to tensor size mismatch input_ids: {input_ids.size()}, labels: {labels.size()}, attention_mask: {attention_mask.size()}"
                            )
                    buffer = {
                        "input_ids": [],
                        "attention_mask": [],
                        "labels": [],
                        "position_ids": [],
                    }
                    buffer_len = 0
                    idx = 1

                if example:
                    # FIXME
                    # just going to drop data points that are too long
                    if len(example["input_ids"]) <= self.seq_length:
                        input_ids = example["input_ids"]
                        attention_mask = example["attention_mask"]
                        labels = example["labels"]

                        if add_concat_token:
                            input_ids.append(self.concat_token_id)
                            attention_mask.append(1)
                            labels.append(self.concat_token_id)

                        input_ids_with_concat = torch.tensor(
                            input_ids, dtype=self.tokens_dtype
                        )
                        attention_mask_with_concat = torch.tensor(
                            [idx * m for m in attention_mask], dtype=torch.int16
                        )
                        labels_with_concat = torch.tensor(
                            labels, dtype=self.tokens_dtype
                        )
                        position_ids = torch.arange(
                            len(input_ids), dtype=self.tokens_dtype
                        )

                        buffer["input_ids"].append(input_ids_with_concat)
                        buffer["attention_mask"].append(attention_mask_with_concat)
                        buffer["labels"].append(labels_with_concat)
                        buffer["position_ids"].append(position_ids)
                        buffer_len += len(input_ids)
