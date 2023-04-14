from typing import List

import torch
from datasets import IterableDataset
from .prompt_tokenizers import PromptTokenizingStrategy, InvalidDataException


# We want this to be a wrapper for an existing dataset that we have loaded
# lets use the concept of middlewares to wrap each dataset, for example
# ConstantLengthDataset(ShuffledDataset([TokenizedPromptDataset(alpaca_dataset)]))
# let's check to ensure we don't truncate an item in the middle, we'll use
# the collators later on to pad the datasets


class TokenizedPromptDataset(IterableDataset):
    def __init__(
        self,
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: IterableDataset,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        self.dataset = dataset

    def __iter__(self):
        iterator = iter(self.dataset)
        # Loop through the entire dataset
        for example in iterator:
            try:
                yield self.prompt_tokenizer.tokenize_prompt(example)
            except InvalidDataException:
                pass


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            seq_length (int): Length of token sequences to return.
    """

    def __init__(
        self,
        tokenizer,
        datasets,
        seq_length=2048,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.datasets: List[IterableDataset] = datasets
        self.seq_length = seq_length

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

                if (
                    not example_len
                    or buffer_len + int(add_concat_token) + example_len
                    > self.seq_length
                ):
                    if buffer["input_ids"]:
                        input_ids = torch.cat(buffer["input_ids"], dim=-1)[
                            : self.seq_length
                        ]
                        attention_mask = torch.cat(buffer["attention_mask"], dim=-1)[
                            : self.seq_length
                        ]
                        labels = torch.cat(buffer["labels"], dim=-1)[: self.seq_length]
                        yield {
                            "input_ids": input_ids,
                            "labels": labels,
                            "attention_mask": attention_mask,
                        }
                    buffer = {"input_ids": [], "attention_mask": [], "labels": []}
                    buffer_len = 0

                if example:
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    labels = example["labels"]

                    if add_concat_token:
                        input_ids.append(self.concat_token_id)
                        attention_mask.append(1)
                        labels.append(self.concat_token_id)

                    input_ids_with_concat = torch.tensor(input_ids, dtype=torch.long)
                    attention_mask_with_concat = torch.tensor(
                        attention_mask, dtype=torch.long
                    )
                    labels_with_concat = torch.tensor(labels, dtype=torch.long)

                    buffer["input_ids"].append(input_ids_with_concat)
                    buffer["attention_mask"].append(attention_mask_with_concat)
                    buffer["labels"].append(labels_with_concat)
                    buffer_len += len(input_ids)
