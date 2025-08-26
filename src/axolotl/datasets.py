"""
Module containing dataset functionality.

We want this to be a wrapper for an existing dataset that we have loaded. Lets use the
concept of middlewares to wrap each dataset. We'll use the collators later on to pad the
datasets.
"""

from datasets import Dataset, IterableDataset

from axolotl.utils.logging import get_logger

from .prompt_tokenizers import PromptTokenizingStrategy

LOG = get_logger(__name__)


class TokenizedPromptDataset(Dataset):
    """Dataset that returns tokenized prompts from a stream of text files.

    Args:
        prompt_tokenizer: The prompt tokenizing method for processing the data.
        dataset: Dataset with text files.
        process_count: Number of processes to use for tokenizing.
        keep_in_memory: Whether to keep the tokenized dataset in memory.
    """

    def __init__(
        self,
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: Dataset,
        process_count: int | None = None,
        keep_in_memory: bool | None = False,
        **kwargs,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        self.process_count = process_count
        self.keep_in_memory = keep_in_memory
        super().__init__(
            self.process(dataset).data,
            **kwargs,
        )

    def process(self, dataset):
        features = dataset.features.keys()

        map_kwargs = {}
        if self.prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = 1_000

        if (
            hasattr(self.prompt_tokenizer, "filter_rows")
            and self.prompt_tokenizer.filter_rows
        ):
            dataset = dataset.filter(
                self.prompt_tokenizer.filter_rows,
                num_proc=self.process_count,
                desc="Strategy Filtering Rows",
            )

        return dataset.map(
            self.prompt_tokenizer.tokenize_prompt,
            num_proc=self.process_count,
            remove_columns=features,
            keep_in_memory=self.keep_in_memory,
            desc="Tokenizing Prompts",
            **map_kwargs,
        )


def wrap_dataset_for_tokenized_prompt(
    prompt_tokenizer: PromptTokenizingStrategy,
    dataset: Dataset | IterableDataset,
    **kwargs,
):
    if isinstance(dataset, IterableDataset):
        map_kwargs = {}
        if prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
        features = list(dataset.features.keys())
        return dataset.map(
            prompt_tokenizer.tokenize_prompt,
            remove_columns=features,
            **map_kwargs,
        )
    return TokenizedPromptDataset(prompt_tokenizer, dataset, **kwargs)
