"""data handling helpers"""
import functools
import hashlib
import logging
import time
from enum import Enum

import huggingface_hub
import requests
from datasets import Dataset

LOG = logging.getLogger("axolotl")


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


def deduplicate_dataset(
    dataset: Dataset, seen_hashes: dict[str, list[int]], other_dataset: Dataset = None
) -> Dataset:
    unique_indices = []

    for idx, row in enumerate(dataset):
        row_hash = sha256(str(row))  # Using SHA256 for collision resistance.
        if row_hash not in seen_hashes:
            seen_hashes[row_hash] = [idx]
            unique_indices.append(idx)
        else:
            # Check for collision by looking up the original dataset indices
            original_indices = seen_hashes[row_hash]
            is_duplicate = False
            for original_idx in original_indices:
                if (
                    not idx == original_idx
                    and original_idx < len(dataset)
                    and str(dataset[original_idx]) == str(row)
                ):
                    is_duplicate = True
                    break
                # Check in the other dataset if provided
                if other_dataset is not None:
                    if original_idx < len(other_dataset) and str(
                        other_dataset[original_idx]
                    ) == str(row):
                        is_duplicate = True
                        break
            if not is_duplicate:
                seen_hashes[row_hash].append(idx)
                unique_indices.append(idx)
                continue
    return dataset.select(unique_indices)


def deduplicate_and_log_datasets(
    *,
    train_dataset: Dataset = None,
    eval_dataset: Dataset = None,
    dataset: Dataset = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Deduplicates train, eval, and an optional dataset if provided, logging original and new sizes.

    Returns:
        tuple: Deduplicated train, eval, and additional datasets.
    """
    seen_hashes: dict[str, list[int]] = {}

    # Handle cases where datasets are None
    if train_dataset is not None:
        LOG.info(
            f"Starting deduplication for train dataset. Original size: {len(train_dataset)}"
        )
        train_dataset = deduplicate_dataset(
            dataset=train_dataset, seen_hashes=seen_hashes
        )
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
            dataset=eval_dataset, seen_hashes=seen_hashes, other_dataset=train_dataset
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
        dataset = deduplicate_dataset(dataset=dataset, seen_hashes=seen_hashes)
        LOG.info(
            f"Deduplication complete for combined dataset. New size: {len(dataset)}"
        )

    return train_dataset, eval_dataset, dataset
