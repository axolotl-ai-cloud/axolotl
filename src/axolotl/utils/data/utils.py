"""Data handling helpers"""

import contextlib
import functools
import hashlib
import time
from enum import Enum
from typing import Callable

import huggingface_hub
import numpy as np
import requests
from datasets import Dataset, IterableDataset

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers.utils import get_dataset_lengths
from axolotl.utils.trainer import filter_sequences_by_length

LOG = get_logger(__name__)


class RetryStrategy(Enum):
    """Enum for retry strategies."""

    CONSTANT = 1
    LINEAR = 2
    EXPONENTIAL = 3


def retry_on_request_exceptions(
    max_retries=3, delay=1, retry_strategy: RetryStrategy = RetryStrategy.LINEAR
) -> Callable:
    """Decorator that retries function calls on specific request exceptions.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Base delay between retries in seconds.
        retry_strategy: Strategy for calculating retry delays.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError,
                    huggingface_hub.errors.HfHubHTTPError,
                ) as exc:
                    if attempt < max_retries - 1:
                        if retry_strategy == RetryStrategy.EXPONENTIAL:
                            step_delay = delay * 2**attempt
                        elif retry_strategy == RetryStrategy.LINEAR:
                            step_delay = delay * (attempt + 1)
                        else:
                            step_delay = delay  # Use constant delay.
                        time.sleep(step_delay)
                    else:
                        raise exc

        return wrapper

    return decorator


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    """Generate MD5 hash of a string."""
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def sha256(to_hash: str, encoding: str = "utf-8") -> str:
    """Generate SHA256 hash of a string."""
    return hashlib.sha256(to_hash.encode(encoding)).hexdigest()


def _deduplicate_dataset(
    dataset: Dataset,
    seen_hashes: set[str] | None = None,
) -> tuple[Dataset, set[str]]:
    """Remove duplicate rows from a dataset using SHA256 hashes.

    Args:
        dataset: Dataset to deduplicate.
        seen_hashes: Set of previously seen row hashes (for cross-deduplication).

    Returns:
        Tuple of deduplicated dataset and the set of seen hashes.
    """
    if seen_hashes is None:
        seen_hashes = set()

    unique_indices = []
    for idx, row in enumerate(dataset):
        row_hash = sha256(str(row))  # Using SHA256 for collision resistance
        if row_hash not in seen_hashes:
            seen_hashes.add(row_hash)
            unique_indices.append(idx)

    return dataset.select(unique_indices), seen_hashes


def deduplicate_and_log_datasets(
    dataset: Dataset,
    other_dataset: Dataset | None = None,
    dataset_name: str | None = "train",
    other_name: str | None = "eval",
) -> tuple[Dataset, Dataset | None]:
    """Deduplicate datasets, with optional cross-dataset deduplication.

    Args:
        dataset: Primary dataset to deduplicate.
        other_dataset: Optional second dataset to deduplicate against the first.
        dataset_name: Name for the primary dataset (for logging).
        other_name: Name for the second dataset (for logging).

    Returns:
        Tuple of (deduplicated_dataset, deduplicated_other_dataset).
    """
    # Deduplicate primary dataset
    LOG.info(
        f"Starting deduplication for {dataset_name} dataset. Original size: {len(dataset)}"
    )
    dataset, seen_rows = _deduplicate_dataset(dataset)
    LOG.info(
        f"Deduplication complete for {dataset_name} dataset. New size: {len(dataset)}"
    )

    # Deduplicate second dataset if provided
    if other_dataset is not None:
        LOG.info(
            f"Starting deduplication for {other_name} dataset. Original size: {len(other_dataset)}"
        )
        other_dataset, _ = _deduplicate_dataset(other_dataset, seen_rows)
        LOG.info(
            f"Deduplication complete for {other_name} dataset. New size: {len(other_dataset)}"
        )

    return dataset, other_dataset


def keep_min_len(sample, min_sequence_len=2):
    """
    Batched filter function that keeps only samples with sequence length >= min_sequence_len.
    Returns a list of booleans indicating which samples to keep.
    """
    min_sequence_len = min_sequence_len or 2

    input_ids = sample["input_ids"]

    # Batched (input_ids is a list of lists)
    results = []
    for seq in input_ids:
        results.append(len(seq) >= min_sequence_len)
    return results


def truncate_long_seq(sample, sequence_len=2048):
    """
    Truncate samples whose sequence length is too long (> sequence_len).
    Modifies the sample in-place and returns the modified sample.
    """
    input_ids = sample["input_ids"]

    # Batched (input_ids is a list of lists)
    for i, seq in enumerate(input_ids):
        length = len(seq)
        if length > sequence_len:
            sample["input_ids"][i] = seq[:sequence_len]
            if "attention_mask" in sample:
                sample["attention_mask"][i] = sample["attention_mask"][i][:sequence_len]
            if "labels" in sample:
                sample["labels"][i] = sample["labels"][i][:sequence_len]
            if "position_ids" in sample:
                sample["position_ids"][i] = sample["position_ids"][i][:sequence_len]
    return sample


def _should_skip_processing(dataset: Dataset) -> bool:
    """Check if dataset should skip long sequence handling."""
    if (
        hasattr(dataset, "column_names")
        and dataset.column_names
        and "input_ids" not in dataset.column_names
    ):
        LOG.warning(
            "Dataset does not contain 'input_ids' column. Skip drop long seq. This is "
            "expected for reward modeling."
        )
        return True
    elif not hasattr(dataset, "column_names") or dataset.column_names is None:
        LOG.info(
            "Dataset is streaming (IterableDataset), skipping long sequence handling"
        )
        return True
    return False


def _log_dataset_stats(dataset: Dataset) -> None:
    """Log min/max sequence lengths for debugging."""
    with contextlib.suppress(AttributeError):
        ds_lengths = get_dataset_lengths(dataset, from_arrow=True)
        LOG.info(f"min_input_len: {np.min(ds_lengths)}")
        LOG.info(f"max_input_len: {np.max(ds_lengths)}")


def _build_filter_kwargs(dataset: Dataset, cfg: DictDefault) -> dict:
    """Build kwargs for dataset filter/map operations."""
    kwargs = {}
    if not isinstance(dataset, IterableDataset):
        kwargs["num_proc"] = cfg.dataset_num_proc
        kwargs["load_from_cache_file"] = not cfg.is_preprocess
    return kwargs


def _filter_short_sequences(
    dataset: Dataset, min_len: int, filter_kwargs: dict
) -> tuple[Dataset, int]:
    """Filter out sequences shorter than min_len. Returns (dataset, num_dropped)."""
    prior_len = len(dataset) if hasattr(dataset, "__len__") else None

    desc_kwargs = {}
    if filter_kwargs:
        desc_kwargs["desc"] = f"Filtering Short Sequences (<{min_len})"

    dataset = dataset.filter(
        functools.partial(keep_min_len, min_sequence_len=min_len),
        batched=True,
        **filter_kwargs,
        **desc_kwargs,
    )

    dropped = 0
    if prior_len:
        dropped = prior_len - len(dataset)
        if dropped > 0:
            LOG.info(f"Dropped {dropped} short sequences (<{min_len} tokens)")

    return dataset, dropped


def _truncate_long_sequences(
    dataset: Dataset, max_len: int, map_kwargs: dict
) -> Dataset:
    """Truncate sequences longer than max_len."""
    desc_kwargs = {}
    if map_kwargs:
        desc_kwargs["desc"] = f"Truncating Sequences (target_len={max_len})"

    dataset = dataset.map(
        functools.partial(truncate_long_seq, sequence_len=max_len),
        batched=True,
        **map_kwargs,
        **desc_kwargs,
    )
    LOG.info(f"Truncated long sequences to max length {max_len}")
    return dataset


def _drop_outside_range(
    dataset: Dataset,
    max_len: int,
    min_len: int,
    raise_on_long: bool,
    filter_kwargs: dict,
) -> tuple[Dataset, int]:
    """Drop sequences outside valid length range [min_len, max_len].

    Returns (dataset, num_dropped)."""
    prior_len = len(dataset) if hasattr(dataset, "__len__") else None

    desc_kwargs = {}
    if filter_kwargs:
        action = (
            "Checking Sequence Lengths"
            if raise_on_long
            else "Dropping Invalid Sequences"
        )
        desc_kwargs["desc"] = f"{action} (<{min_len} or >{max_len})"

    dataset = dataset.filter(
        functools.partial(
            filter_sequences_by_length,
            sequence_len=max_len,
            min_sequence_len=min_len,
            raise_on_drop=raise_on_long,
        ),
        batched=True,
        **filter_kwargs,
        **desc_kwargs,
    )

    dropped = 0
    if not raise_on_long and prior_len:
        dropped = prior_len - len(dataset)
        if dropped > 0:
            LOG.info(
                f"Dropped {dropped} sequences outside valid range "
                f"([{min_len}, {max_len}])"
            )

    return dataset, dropped


def handle_long_seq_in_dataset(
    dataset: Dataset, sequence_len: int, cfg: DictDefault
) -> Dataset:
    """Remove sequences longer than configured maximum from dataset.

    Args:
        dataset: Dataset to filter.
        sequence_len: Maximum length for sequences to keep
        cfg: Dictionary mapping `axolotl` config keys to values.

    Returns:
        Filtered dataset with long sequences handled according to the excess_length_strategy value:
            'drop' (default)    excludes any sequence longer than sequence_len
            'truncate'          truncates them down to sequence_len
            'raise'             raises a ValueError if any sequence was found that was longer than sequence_len
    """
    # Early returns for special cases
    if _should_skip_processing(dataset):
        return dataset

    excess_length_strategy = (cfg.excess_length_strategy or "drop").lower()

    _log_dataset_stats(dataset)

    # Setup kwargs
    filter_kwargs = _build_filter_kwargs(dataset, cfg)

    # Handle sequences based on strategy
    if excess_length_strategy == "truncate":
        dataset, _ = _filter_short_sequences(dataset, cfg.min_sample_len, filter_kwargs)
        dataset = _truncate_long_sequences(dataset, sequence_len, filter_kwargs)
    else:
        raise_on_long = excess_length_strategy == "raise"
        dataset, _ = _drop_outside_range(
            dataset, sequence_len, cfg.min_sample_len, raise_on_long, filter_kwargs
        )

    return dataset
