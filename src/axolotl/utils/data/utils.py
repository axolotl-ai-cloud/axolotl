"""data handling helpers"""

import hashlib
from datasets import Dataset
import logging


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec



def collect_unique_rows(
    *, dataset: Dataset
) -> set:
    """Converts each row to a tuple, keeping only unique rows."""
    unique_row_tuples = set(tuple(row.values()) for row in dataset)
    return unique_row_tuples

def convert_tuples_to_dict(
    *, unique_row_tuples: set, column_names: list
) -> dict:
    """Converts unique row tuples back to dictionary format."""
    unique_data_dict = {column: [] for column in column_names}
    for row_tuple in unique_row_tuples:
        for i, column in enumerate(column_names):
            unique_data_dict[column].append(row_tuple[i])
    return unique_data_dict

def deduplicate_dataset(
    *, dataset: Dataset
) -> Dataset:
    """Returns a deduplicated Hugging Face Dataset."""
    unique_rows = collect_unique_rows(dataset=dataset)
    unique_data_dict = convert_tuples_to_dict(
        unique_row_tuples=unique_rows, column_names=dataset.column_names
    )
    return Dataset.from_dict(unique_data_dict)


def deduplicate_and_log_datasets(
    *, train_dataset: Dataset, eval_dataset: Dataset
) -> tuple[Dataset, Dataset]: # type: ignore
    """
    Deduplicates train and eval datasets if provided, logging original and new sizes.
    
    Returns:
        tuple: Deduplicated train and eval datasets.
    """
    
    if train_dataset is not None:
        logging.info(f"Starting deduplication for train dataset. Original size: {len(train_dataset)}")
        train_dataset = deduplicate_dataset(dataset=train_dataset)
        logging.info(f"Deduplication complete for train dataset. New size: {len(train_dataset)}")
    else:
        logging.info("Train dataset is None. Skipping deduplication.")

    if eval_dataset is not None:
        logging.info(f"Starting deduplication for eval dataset. Original size: {len(eval_dataset)}")
        eval_dataset = deduplicate_dataset(dataset=eval_dataset)
        logging.info(f"Deduplication complete for eval dataset. New size: {len(eval_dataset)}")
    else:
        logging.info("Eval dataset is None. Skipping deduplication.")
    
    return train_dataset, eval_dataset

