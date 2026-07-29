"""
Module containing dataset functionality.

We want this to be a wrapper for an existing dataset that we have loaded. Lets use the
concept of middlewares to wrap each dataset. We'll use the collators later on to pad the
datasets.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, IterableDataset, concatenate_datasets

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
        batch_size: Rows per batched tokenization call.
        writer_batch_size: Rows buffered in RAM before each Arrow cache flush.
    """

    def __init__(
        self,
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: Dataset,
        process_count: int | None = None,
        keep_in_memory: bool | None = False,
        batch_size: int | None = None,
        writer_batch_size: int | None = None,
        **kwargs,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        self.process_count = process_count
        self.keep_in_memory = keep_in_memory
        self.batch_size = batch_size
        self.writer_batch_size = writer_batch_size
        super().__init__(
            self.process(dataset).data,
            **kwargs,
        )

    def process(self, dataset):
        features = dataset.features.keys()

        map_kwargs = {}
        if self.prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = self.batch_size or 1_000
        if self.writer_batch_size:
            map_kwargs["writer_batch_size"] = self.writer_batch_size

        if (
            hasattr(self.prompt_tokenizer, "filter_rows")
            and self.prompt_tokenizer.filter_rows
        ):
            dataset = dataset.filter(
                self.prompt_tokenizer.filter_rows,
                num_proc=self.process_count,
                desc="Strategy Filtering Rows",
            )

        def _tokenize(part):
            return part.map(
                self.prompt_tokenizer.tokenize_prompt,
                num_proc=self.process_count,
                remove_columns=features,
                keep_in_memory=self.keep_in_memory,
                desc="Tokenizing Prompts",
                **map_kwargs,
            )

        split_indices = self._split_media_indices(dataset)
        if split_indices is not None:
            # Media rows emit processor columns (pixel_values, ...) that text rows
            # don't, and the Arrow writer locks its schema on the first batch it
            # sees; tokenize each modality separately and restore the row order.
            with_media, without_media, order = split_indices
            merged = concatenate_datasets(
                [_tokenize(dataset.select(with_media))]
                + [_tokenize(dataset.select(without_media))]
            )
            return merged.select(order).flatten_indices(
                writer_batch_size=self.writer_batch_size or 1_000
            )

        return _tokenize(dataset)

    def _split_media_indices(self, dataset):
        """Row indices (with_media, without_media, restore_order) for mixed
        media/text datasets, or None when no split is needed."""
        prompter = getattr(self.prompt_tokenizer, "prompter", None)
        if getattr(prompter, "processor", None) is None:
            return None
        media_field = getattr(self.prompt_tokenizer, "images", None)
        if not media_field or media_field not in dataset.column_names:
            return None

        # Arrow-level presence check: never decodes the images themselves.
        column = (
            dataset.select_columns([media_field])
            .with_format("arrow")[:]
            .column(media_field)
        )
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            mask = pc.greater(pc.fill_null(pc.list_value_length(column), 0), 0)
        else:
            mask = pc.is_valid(column)
        mask = np.asarray(mask.combine_chunks())

        with_media = np.flatnonzero(mask)
        without_media = np.flatnonzero(~mask)
        if not len(with_media) or not len(without_media):
            return None
        order = np.argsort(np.concatenate([with_media, without_media]), kind="stable")
        return with_media, without_media, order


def wrap_dataset_for_tokenized_prompt(
    prompt_tokenizer: PromptTokenizingStrategy,
    dataset: Dataset | IterableDataset,
    **kwargs,
):
    if isinstance(dataset, IterableDataset):
        map_kwargs = {}
        if prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
            if kwargs.get("batch_size"):
                map_kwargs["batch_size"] = kwargs["batch_size"]
        features = list(dataset.features.keys())
        return dataset.map(
            prompt_tokenizer.tokenize_prompt,
            remove_columns=features,
            **map_kwargs,
        )
    return TokenizedPromptDataset(prompt_tokenizer, dataset, **kwargs)
