"""
Module containing dataset functionality.

We want this to be a wrapper for an existing dataset that we have loaded. Lets use the
concept of middlewares to wrap each dataset, for example:
ConstantLengthDataset(ShuffledDataset([TokenizedPromptDataset(alpaca_dataset)])).
Let's check to ensure we don't truncate an item in the middle. We'll use the collators
later on to pad the datasets.
"""

from typing import Any

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

    def __init__(  # pylint: disable=super-init-not-called
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

    def process(self, dataset: Dataset | IterableDataset) -> Dataset | IterableDataset:
        """Apply filtering and tokenization."""
        # For IterableDataset, we can't access features up front. Anyways, we don't care
        # to remove unused columns from streaming datasets.
        features = None
        if not isinstance(dataset, IterableDataset):
            features = dataset.features.keys()

        map_kwargs: dict[str, Any] = {}
        if self.prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = 1_000

        if (
            hasattr(self.prompt_tokenizer, "filter_rows")
            and self.prompt_tokenizer.filter_rows
        ):
            filter_kwargs: dict[str, Any] = {"desc": "Strategy Filtering Rows"}
            if not isinstance(dataset, IterableDataset):
                filter_kwargs["num_proc"] = self.process_count

            dataset = dataset.filter(
                self.prompt_tokenizer.filter_rows,
                **filter_kwargs,
            )

        map_kwargs = {
            **map_kwargs,
            "desc": "Tokenizing Prompts",
        }

        # Only add remove_columns for regular datasets
        if not isinstance(dataset, IterableDataset):
            map_kwargs["remove_columns"] = features
            map_kwargs["num_proc"] = self.process_count
            map_kwargs["keep_in_memory"] = self.keep_in_memory

        return dataset.map(
            self.prompt_tokenizer.tokenize_prompt,
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
        return dataset.map(
            prompt_tokenizer.tokenize_prompt,
            **map_kwargs,
        )
    return TokenizedPromptDataset(prompt_tokenizer, dataset, **kwargs)
