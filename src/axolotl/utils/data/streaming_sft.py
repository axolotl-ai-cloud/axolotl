"""Optimized streaming SFT with multipack support that avoids repeated preprocessing."""

import functools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset, IterableDataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import add_position_ids, drop_long_seq

LOG = get_logger(__name__)


def encode_packed_streaming_sft(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
) -> Dict[str, List]:
    """
    Encode streaming SFT data with packing, avoiding repeated preprocessing logs.

    This is similar to encode_packed_pretraining but skips the verbose
    process_pretraining_datasets_for_packing call that logs repeatedly.
    """
    # Tokenize all the examples
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]

    # Apply filtering and preprocessing directly without verbose logging
    # Filter out long sequences
    def should_keep(sample):
        return drop_long_seq(sample, sequence_len=max_seq_length)

    # Convert to list for filtering (since we need to iterate anyway)
    filtered_samples = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        if should_keep(sample):
            # Add position_ids if needed
            if not multipack_attn:  # skip_position_ids=False when multipack_attn=True
                sample = add_position_ids(sample)
            filtered_samples.append(sample)

    # Convert back to dataset
    if not filtered_samples:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    train_dataset = Dataset.from_list(filtered_samples)

    # Remove attention_mask if needed for multipack
    if multipack_attn and "attention_mask" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("attention_mask")

    # Use MultipackBatchSampler to create efficient packed batches
    sampler = MultipackBatchSampler(
        sampler=RandomSampler(train_dataset),
        lengths=get_dataset_lengths(train_dataset),
        batch_size=1,
        batch_max_len=batch_size * max_seq_length,
        drop_last=True,
        num_processes=1,
    )

    # Collect packed data
    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            features = train_dataset[data]
            # Clean up unnecessary fields
            for field in ["num_truncated_tokens", "overflow_to_sample_mapping"]:
                if field in features:
                    del features[field]
            # Ensure labels exist
            if "labels" not in features:
                features["labels"] = features["input_ids"].copy()
            # Apply collator
            collated_features = collate_fn(features)

            # Collect features
            for feature in features.keys():
                if feature == "length":
                    continue
                chunked_data[feature].append(collated_features[feature].squeeze(0))

    return chunked_data


def wrap_streaming_sft_dataset_optimized(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg,
    ds_wrapper_fn: Callable,
    max_tokens: int = 2048,
    batch_size: int = 4,
    seed: int = 42,
    buffer_size: int = 1000,
) -> IterableDataset:
    """
    Wrap a streaming SFT dataset with optimized multipack batching.

    This avoids the repeated preprocessing logs by using a custom encoder
    that applies filtering and position_ids directly.
    """
    # Create collator for packing
    collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=max_tokens,
        multipack_attn=cfg.pretrain_multipack_attn,
    )

    # Create optimized encode function
    encode = functools.partial(
        encode_packed_streaming_sft,
        collate_fn,
        ds_wrapper_fn,
        max_seq_length=max_tokens,
        batch_size=batch_size,
        multipack_attn=cfg.pretrain_multipack_attn,
    )

    # Apply shuffling if configured
    if cfg.shuffle_merged_datasets:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    else:
        LOG.debug("NOT shuffling merged streaming datasets")

    # Get column names to remove
    remove_columns = []
    for first_row in dataset:
        remove_columns = list(first_row.keys())
        break

    # Reset dataset after peeking
    if cfg.shuffle_merged_datasets:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    # Map the optimized encoding function
    dataset = dataset.map(
        encode,
        batched=True,
        batch_size=buffer_size,
        remove_columns=remove_columns,
    )

    # Set micro_batch_size to 1 since we've already packed
    cfg.micro_batch_size = 1

    return dataset
