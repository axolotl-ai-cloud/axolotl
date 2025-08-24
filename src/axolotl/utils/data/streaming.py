"""Utilities for handling streaming datasets."""

import functools
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, IterableDataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import add_position_ids

LOG = get_logger(__name__)


def wrap_streaming_sft_dataset(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg,
    dataset_config,
    d_base_type: str,
    d_prompt_style: str | None,
    processor: Any | None,
    max_tokens: int = 2048,
    buffer_size: int = 10_000,
) -> IterableDataset:
    """
    Wrap a streaming SFT dataset with tokenization and optional packing.

    This is similar to wrap_pretraining_dataset but for SFT datasets.

    Args:
        dataset: The streaming dataset to wrap
        tokenizer: Tokenizer to use
        cfg: Configuration object
        dataset_config: Dataset configuration
        d_base_type: Base dataset type
        d_prompt_style: Prompt style
        processor: Optional processor for multimodal
        max_tokens: Maximum sequence length
        buffer_size: Buffer size for shuffling

    Returns:
        Wrapped streaming dataset ready for training
    """

    # Import here to avoid circular imports
    from axolotl.utils.data.wrappers import get_dataset_wrapper

    # Apply shuffling if configured
    if cfg.shuffle_merged_datasets:
        LOG.info(f"Shuffling streaming dataset with buffer_size={buffer_size}")
        dataset = dataset.shuffle(seed=cfg.seed, buffer_size=buffer_size)

    # For streaming datasets, we need to get column names from the first sample
    remove_columns = []
    for first_row in dataset:
        remove_columns = list(first_row.keys())
        break

    # Reset dataset after peeking
    if cfg.shuffle_merged_datasets:
        dataset = dataset.shuffle(seed=cfg.seed, buffer_size=buffer_size)

    # Define the encoding function - always add position_ids for compatibility
    if cfg.sample_packing:
        # For sample packing, we need to handle position_ids
        def encode_streaming_packed(examples: Dict[str, List]) -> Dict[str, List]:
            """Encode examples for streaming with sample packing."""
            # Convert the batch dict to a temporary Dataset for processing
            temp_dataset = Dataset.from_dict(examples)

            # Apply the dataset wrapper to tokenize
            wrapped_dataset, _ = get_dataset_wrapper(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
                cfg=cfg,
                dataset_base_type=d_base_type,
                dataset=temp_dataset,
                dataset_prompt_style=d_prompt_style,
                processor=processor,
            )

            # Convert to dict for processing
            result = {}
            if hasattr(wrapped_dataset, "to_dict"):
                result = wrapped_dataset.to_dict()
            else:
                for key in wrapped_dataset.column_names:
                    result[key] = wrapped_dataset[key]

            # Add position_ids using the existing function
            result = add_position_ids(result)

            # For multipack attention, we may need to drop attention_mask
            if cfg.pretrain_multipack_attn and "attention_mask" in result:
                del result["attention_mask"]

            return result

        encode_fn = encode_streaming_packed
    else:
        # Regular encoding without packing - still add position_ids for compatibility
        def encode_streaming(examples: Dict[str, List]) -> Dict[str, List]:
            """Encode examples for streaming."""
            # Convert the batch dict to a temporary Dataset for processing
            temp_dataset = Dataset.from_dict(examples)

            # Apply the dataset wrapper to tokenize
            wrapped_dataset, _ = get_dataset_wrapper(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
                cfg=cfg,
                dataset_base_type=d_base_type,
                dataset=temp_dataset,
                dataset_prompt_style=d_prompt_style,
                processor=processor,
            )

            # Convert to dict format
            result = {}
            if hasattr(wrapped_dataset, "to_dict"):
                result = wrapped_dataset.to_dict()
            else:
                for key in wrapped_dataset.column_names:
                    result[key] = wrapped_dataset[key]

            # Add position_ids even without packing for compatibility
            result = add_position_ids(result)

            return result

        encode_fn = encode_streaming

    # Map the encoding function over the streaming dataset
    dataset = dataset.map(
        encode_fn,
        batched=True,
        batch_size=buffer_size,
        remove_columns=remove_columns,
    )

    # Set format for PyTorch
    dataset = dataset.with_format("torch")

    return dataset
