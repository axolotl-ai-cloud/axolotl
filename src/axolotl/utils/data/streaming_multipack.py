"""Streaming dataset with multipack support for SFT."""

import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, IterableDataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = get_logger(__name__)


def encode_packed_sft_streaming(
    collate_fn,
    ds_wrapper_fn,
    examples: Dict[str, List],
    dataset_config,
    tokenizer,
    cfg,
    d_base_type: str,
    d_prompt_style: str | None,
    processor: Any | None,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
) -> Dict[str, List]:
    """
    Encode and pack streaming SFT data similar to how pretraining does it.

    This function:
    1. Tokenizes the examples using the dataset wrapper
    2. Adds position_ids for each sequence
    3. Uses MultipackBatchSampler to efficiently pack sequences
    4. Applies the collator to handle attention masks properly

    Args:
        collate_fn: Collator function for handling batches
        ds_wrapper_fn: Function to get the dataset wrapper for tokenization
        examples: Dict of lists containing the raw examples
        dataset_config: Configuration for the dataset
        tokenizer: Tokenizer to use
        cfg: Main configuration
        d_base_type: Dataset base type
        d_prompt_style: Prompt style
        processor: Optional processor for multimodal
        max_seq_length: Maximum sequence length
        batch_size: Batch size for packing
        multipack_attn: Whether to use multipack attention

    Returns:
        Dict of packed and processed data ready for training
    """
    # Import here to avoid circular imports
    from axolotl.utils.data.wrappers import get_dataset_wrapper

    # Convert examples to Dataset for processing
    temp_dataset = Dataset.from_dict(examples)

    # Apply the dataset wrapper to tokenize
    train_dataset, _ = get_dataset_wrapper(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        cfg=cfg,
        dataset_base_type=d_base_type,
        dataset=temp_dataset,
        dataset_prompt_style=d_prompt_style,
        processor=processor,
    )

    # Process for packing - add position_ids and filter long sequences
    train_dataset = process_pretraining_datasets_for_packing(
        train_dataset,
        max_seq_length,
        skip_position_ids=not multipack_attn,
        drop_attention_mask=multipack_attn,
    )

    # Use MultipackBatchSampler to create efficient packed batches
    sampler = MultipackBatchSampler(
        sampler=RandomSampler(train_dataset),
        lengths=get_dataset_lengths(train_dataset),
        batch_size=1,  # We pack multiple sequences into one "batch"
        batch_max_len=batch_size * max_seq_length,  # Total tokens in packed batch
        drop_last=True,
        num_processes=1,  # Single process for streaming
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

            # Apply collator to handle padding and attention masks
            collated_features = collate_fn(features)

            # Collect features
            for feature in features.keys():
                if feature == "length":
                    continue
                chunked_data[feature].append(collated_features[feature].squeeze(0))

    return chunked_data


def wrap_streaming_sft_dataset_with_packing(
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
    Wrap a streaming SFT dataset with tokenization and multipack batching.

    This creates properly packed batches with:
    - Multiple sequences concatenated together
    - Position IDs that reset for each sequence
    - Attention masks that prevent cross-attention between sequences

    Args:
        dataset: The streaming dataset to wrap
        tokenizer: Tokenizer to use
        cfg: Configuration object
        dataset_config: Dataset configuration
        d_base_type: Base dataset type
        d_prompt_style: Prompt style
        processor: Optional processor for multimodal
        max_tokens: Maximum sequence length
        buffer_size: Buffer size for streaming/shuffling

    Returns:
        Wrapped streaming dataset with multipack batching
    """

    # Apply shuffling if configured
    if cfg.shuffle_merged_datasets:
        LOG.info(f"Shuffling streaming dataset with buffer_size={buffer_size}")
        dataset = dataset.shuffle(seed=cfg.seed, buffer_size=buffer_size)

    # Get column names from first sample
    remove_columns = []
    for first_row in dataset:
        remove_columns = list(first_row.keys())
        break

    # Reset dataset after peeking
    if cfg.shuffle_merged_datasets:
        dataset = dataset.shuffle(seed=cfg.seed, buffer_size=buffer_size)

    # Create the collator for multipack
    collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=max_tokens,
        multipack_attn=cfg.pretrain_multipack_attn,
    )

    # Create the encoding function
    # batch_size here refers to how many sequences to pack together to fill max_tokens
    # The actual batching happens at the DataLoader level with micro_batch_size=1
    pack_batch_size = max(
        1, max_tokens // 512
    )  # Estimate based on typical sequence lengths

    encode_fn = functools.partial(
        encode_packed_sft_streaming,
        collate_fn,
        None,  # ds_wrapper_fn will be created inside
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        cfg=cfg,
        d_base_type=d_base_type,
        d_prompt_style=d_prompt_style,
        processor=processor,
        max_seq_length=max_tokens,
        batch_size=pack_batch_size,
        multipack_attn=cfg.pretrain_multipack_attn,
    )

    # Map the encoding function over the streaming dataset
    # This will process data in batches and apply packing
    dataset = dataset.map(
        encode_fn,
        batched=True,
        batch_size=buffer_size,  # Process large batches for efficiency
        remove_columns=remove_columns,
    )

    # Set format for PyTorch
    dataset = dataset.with_format("torch")

    # IMPORTANT: Set micro_batch_size to 1 since we've already packed
    # This prevents the trainer from trying to batch our packed sequences
    cfg.micro_batch_size = 1

    return dataset
