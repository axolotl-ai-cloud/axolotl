"""data handling helpers"""

import hashlib
import logging

from datasets import Dataset

LOG = logging.getLogger("axolotl")


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def deduplicate_dataset(dataset: Dataset, seen_hashes: set[str]) -> Dataset:
    unique_indices = []

    for idx, row in enumerate(dataset):
        row_hash = hashlib.sha256(
            str(row).encode("utf-8")
        ).hexdigest()  # using SHA256 for the low risk of collision in large datasets.
        if row_hash not in seen_hashes:
            seen_hashes.add(row_hash)
            unique_indices.append(idx)

    return dataset.select(unique_indices)


def deduplicate_and_log_datasets(
    *,
    train_dataset: Dataset = None,
    eval_dataset: Dataset = None,
    dataset: Dataset = None,
) -> tuple[Dataset, Dataset, Dataset]:  # type: ignore
    """
    Deduplicates train, eval, and an optional dataset if provided, logging original and new sizes.

    Returns:
        tuple: Deduplicated train, eval, and additional datasets.
    """
    # Handle cases where datasets are None
    seen_hashes: set[str] = set()
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
    elif dataset is None:
        LOG.info("Train dataset is None. Skipping deduplication.")

    if eval_dataset is not None:
        LOG.info(
            f"Starting deduplication for eval dataset. Original size: {len(eval_dataset)}"
        )
        eval_dataset = deduplicate_dataset(
            dataset=eval_dataset, seen_hashes=seen_hashes
        )
        LOG.info(
            f"Deduplication complete for eval dataset. New size: {len(eval_dataset)}"
        )
    elif dataset is None:
        LOG.info("Eval dataset is None. Skipping deduplication.")

    if dataset is not None and (eval_dataset is None and train_dataset is None):
        LOG.info(
            f"Starting deduplication for combined dataset (train and eval). Original size: {len(dataset)}"
        )
        dataset = deduplicate_dataset(dataset=dataset, seen_hashes=seen_hashes)
        LOG.info(
            f"Deduplication complete for combined dataset (train and eval). New size: {len(dataset)}"
        )

    return train_dataset, eval_dataset, dataset
