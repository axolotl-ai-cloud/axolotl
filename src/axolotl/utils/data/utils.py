"""data handling helpers"""

import functools
import hashlib
import time
from enum import Enum
from typing import Optional

import huggingface_hub
import numpy as np
import requests
from datasets import Dataset, IterableDataset

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers.utils import get_dataset_lengths
from axolotl.utils.trainer import drop_long_seq

LOG = get_logger(__name__)


class RetryStrategy(Enum):
    """
    Enum for retry strategies.
    """

    CONSTANT = 1
    LINEAR = 2
    EXPONENTIAL = 3


def retry_on_request_exceptions(
    max_retries=3, delay=1, retry_strategy: RetryStrategy = RetryStrategy.LINEAR
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
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
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def sha256(to_hash: str, encoding: str = "utf-8") -> str:
    return hashlib.sha256(to_hash.encode(encoding)).hexdigest()


def compute_row_hash(example):
    return {"row_hash": sha256(str(example))}


def deduplicate_dataset(
    dataset: Dataset,
    other_dataset: Dataset = None,
    num_proc: Optional[int] = None,
) -> Dataset:
    hashes, other_hashes, seen_hashes, unique_indices = [], set(), set(), set()

    if dataset is not None:
        hashed = dataset.map(
            compute_row_hash, remove_columns=dataset.column_names, num_proc=num_proc
        )
        hashes = hashed["row_hash"]
        del hashed

    if other_dataset is not None:
        other_hashed = other_dataset.map(
            compute_row_hash, remove_columns=dataset.column_names, num_proc=num_proc
        )
        other_hashes = set(other_hashed["row_hash"])
        del other_hashed

    for idx, row_hash in enumerate(hashes):
        if row_hash in seen_hashes or row_hash in other_hashes:
            continue
        seen_hashes.add(row_hash)
        unique_indices.add(idx)

    del hashes, other_hashes, seen_hashes

    return dataset.select(unique_indices)


def deduplicate_and_log_datasets(
    *,
    train_dataset: Dataset = None,
    eval_dataset: Dataset = None,
    dataset: Dataset = None,
    num_proc: Optional[int] = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Deduplicates train, eval, and an optional dataset if provided, logging original and new sizes.

    Returns:
        tuple: Deduplicated train, eval, and additional datasets.
    """
    # Handle cases where datasets are None
    if train_dataset is not None:
        LOG.info(
            f"Starting deduplication for train dataset. Original size: {len(train_dataset)}"
        )
        train_dataset = deduplicate_dataset(dataset=train_dataset, num_proc=num_proc)
        LOG.info(
            f"Deduplication complete for train dataset. New size: {len(train_dataset)}"
        )
    else:
        LOG.info("Train dataset is None. Skipping deduplication.")

    if eval_dataset is not None:
        LOG.info(
            f"Starting deduplication for eval dataset. Original size: {len(eval_dataset)}"
        )
        eval_dataset = deduplicate_dataset(
            dataset=eval_dataset, other_dataset=train_dataset, num_proc=num_proc
        )
        LOG.info(
            f"Deduplication complete for eval dataset. New size: {len(eval_dataset)}"
        )
    else:
        LOG.info("Eval dataset is None. Skipping deduplication.")

    if dataset is not None and (eval_dataset is None and train_dataset is None):
        LOG.info(
            f"Starting deduplication for combined dataset. Original size: {len(dataset)}"
        )
        dataset = deduplicate_dataset(dataset=dataset, num_proc=num_proc)
        LOG.info(
            f"Deduplication complete for combined dataset. New size: {len(dataset)}"
        )

    return train_dataset, eval_dataset, dataset


def drop_long_seq_in_dataset(dataset: Dataset, cfg: DictDefault):
    if "input_ids" not in dataset.column_names:
        LOG.warning(
            "Dataset does not contain 'input_ids' column. Skip drop long seq. This is expected for RewardModeling."
        )
        return dataset

    drop_long = functools.partial(
        drop_long_seq,
        sequence_len=cfg.sequence_len,
        min_sequence_len=cfg.min_sample_len,
    )

    try:
        ds_lengths = get_dataset_lengths(dataset, from_arrow=True)
        min_input_len = np.min(ds_lengths)
        LOG.info(f"min_input_len: {min_input_len}")
        max_input_len = np.max(ds_lengths)
        LOG.info(f"max_input_len: {max_input_len}")
    except AttributeError:
        pass

    try:
        prior_len = len(dataset)
    except TypeError:
        # handle iterable datasets case
        prior_len = None

    filter_map_kwargs = {}
    if not isinstance(dataset, IterableDataset):
        filter_map_kwargs["num_proc"] = cfg.dataset_processes
        filter_map_kwargs["load_from_cache_file"] = not cfg.is_preprocess

    drop_long_kwargs = {}
    if filter_map_kwargs:
        drop_long_kwargs["desc"] = "Dropping Long Sequences"

    dataset = dataset.filter(
        drop_long,
        batched=True,
        **filter_map_kwargs,
        **drop_long_kwargs,
    )
    if prior_len:
        dropped = prior_len - len(dataset)
        if dropped:
            LOG.warning(f"Dropped {dropped} long samples from dataset")

    return dataset
