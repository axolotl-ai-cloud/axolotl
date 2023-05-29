"""Module containing Dataset functionality"""

import logging
from typing import List

import torch
from datasets import IterableDataset
from rathe import AbstractPromptParser, AbstractPromptFormatter

from transformers import PreTrainedTokenizerBase

# We want this to be a wrapper for an existing dataset that we have loaded
# lets use the concept of middlewares to wrap each dataset, for example
# ConstantLengthDataset(ShuffledDataset([TokenizedPromptDataset(alpaca_dataset)]))
# let's check to ensure we don't truncate an item in the middle, we'll use
# the collators later on to pad the datasets


def TokenizedPromptDataset(
    parser: AbstractPromptParser,
    formatter: AbstractPromptFormatter,
    tokenizer: PreTrainedTokenizerBase,
    dataset: IterableDataset,
) -> IterableDataset:
    old_columns = dataset.column_names

    def format_and_tokenize(row):
        prompt = parser.parse(row)
        formatted = formatter.format(prompt, tokenizer.special_tokens_map)
        item = formatted.to_tokens(tokenizer)
        for key in item:
            item[key] = item[key].squeeze(0)
        return item
        
    return dataset.map(format_and_tokenize, remove_columns=old_columns, num_proc=8)


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
        buffer = {"input_ids": [], "attention_mask": [], "labels": []}
        buffer_len = 0
        for dataset in self.datasets:
            iterator = iter(dataset)
            more_examples = True
            while more_examples:
                try:
                    example = next(iterator)
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
                        labels = torch.cat(buffer["labels"], dim=-1)[: self.seq_length]
                        if labels.size() == input_ids.size() and (
                            attention_mask.size() == input_ids.size()
                        ):
                            yield {
                                "input_ids": input_ids,
                                "labels": labels,
                                "attention_mask": attention_mask,
                            }
                        else:
                            logging.warning(
                                f"dropping batch due to tensor size mismatch input_ids: {input_ids.size()}, labels: {labels.size()}, attention_mask: {attention_mask.size()}"
                            )
                    buffer = {
                        "input_ids": [],
                        "attention_mask": [],
                        "labels": [],
                    }
                    buffer_len = 0

                if example:
                    # just going to drop data points that are too long
                    if len(example["input_ids"]) <= self.seq_length:
                        input_ids = example["input_ids"]
                        attention_mask = example["attention_mask"]
                        labels = example["labels"]
                        if (
                            buffer["input_ids"]
                            and input_ids[0] == self.tokenizer.bos_token_id
                        ):
                            attention_mask[0] = 0

                        if add_concat_token:
                            input_ids.append(self.concat_token_id)
                            attention_mask.append(1)
                            labels.append(self.concat_token_id)

                        input_ids_with_concat = torch.tensor(
                            input_ids, dtype=self.tokens_dtype
                        )
                        attention_mask_with_concat = torch.tensor(
                            attention_mask, dtype=self.tokens_dtype
                        )
                        labels_with_concat = torch.tensor(
                            labels, dtype=self.tokens_dtype
                        )

                        buffer["input_ids"].append(input_ids_with_concat)
                        buffer["attention_mask"].append(attention_mask_with_concat)
                        buffer["labels"].append(labels_with_concat)
                        buffer_len += len(input_ids)
