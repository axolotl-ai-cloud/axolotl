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
            with_media, without_media = split_indices
            tokenize_prompt = self.prompt_tokenizer.tokenize_prompt

            batched = self.prompt_tokenizer.supports_batched

            def _tokenize_row_tracked(batch, indices):
                # row-by-row (batches of one) so dropped rows can't desync the
                # order restore: survivors carry their part-relative source index
                feature_names = list(batch.keys())
                rows, row_ids = [], []
                for idx, values in zip(
                    indices, zip(*batch.values(), strict=False), strict=False
                ):
                    if batched:
                        result = tokenize_prompt(
                            {
                                k: [v]
                                for k, v in zip(feature_names, values, strict=False)
                            }
                        )
                        row = {k: v[0] for k, v in result.items()} if result else None
                    else:
                        row = tokenize_prompt(
                            dict(zip(feature_names, values, strict=False))
                        )
                    if row:
                        rows.append(row)
                        row_ids.append(idx)
                keys: list[str] = []
                for row in rows:
                    for key in row:
                        if key not in keys:
                            keys.append(key)
                out = {key: [row.get(key) for row in rows] for key in keys}
                out["__mm_source_row"] = row_ids
                return out

            def _tokenize_part(indices):
                part_kwargs = dict(map_kwargs)
                part_kwargs["batched"] = True
                part_kwargs.setdefault("batch_size", self.batch_size or 1_000)
                return dataset.select(indices).map(
                    _tokenize_row_tracked,
                    with_indices=True,
                    num_proc=self.process_count,
                    remove_columns=features,
                    keep_in_memory=self.keep_in_memory,
                    desc="Tokenizing Prompts",
                    **part_kwargs,
                )

            parts, source_rows = [], []
            for indices in (with_media, without_media):
                part = _tokenize_part(indices)
                if len(part):
                    parts.append(part)
                    source_rows.append(
                        np.asarray(indices)[np.asarray(part["__mm_source_row"])]
                    )
            if not parts:
                return _tokenize(dataset.select([]))

            merged = concatenate_datasets(parts).remove_columns("__mm_source_row")
            dropped = len(with_media) + len(without_media) - len(merged)
            if dropped:
                LOG.warning(
                    "%d row(s) were dropped while tokenizing a mixed media/text "
                    "dataset; surviving rows keep their original order.",
                    dropped,
                )
            restore = np.argsort(np.concatenate(source_rows), kind="stable")
            return merged.select(restore).flatten_indices(
                keep_in_memory=bool(self.keep_in_memory),
                writer_batch_size=self.writer_batch_size or 1_000,
            )

        return _tokenize(dataset)

    def _split_media_indices(self, dataset):
        """Row indices (with_media, without_media) for mixed media/text
        datasets, or None when no split is needed."""
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
        return with_media, without_media


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
