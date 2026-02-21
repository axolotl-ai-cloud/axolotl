"""Data handling specific to streaming datasets."""

import functools
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = get_logger(__name__)


def encode_streaming(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    text_column: str = "text",
    concatenate: bool = True,
) -> Dict[str, List]:
    """
    Encode streaming examples with auto-chunking support.

    Tokenizes text without truncation, appends EOS/PAD document boundary tokens,
    then splits into max_tokens-sized chunks. This ensures no data is lost from
    long sequences while preserving document boundaries for pretraining.

    Note: When concatenate=False, individual samples that exceed max_tokens will
    be split into multiple chunks (rather than truncated), so the output may have
    more rows than the input.

    Note: Tokenization is performed without truncation so that long documents can
    be chunked rather than silently lost. The streaming buffer size
    (streaming_multipack_buffer_size) controls how many examples are batched at
    once, limiting peak memory usage.

    Args:
        examples: Dictionary containing text samples
        tokenizer: The tokenizer to use
        max_tokens: Maximum sequence length for each chunk
        text_column: Name of the text column in examples
        concatenate: If True, concatenate all samples before chunking

    Returns:
        Dictionary with input_ids, labels, and attention_mask lists
    """
    # Tokenize without truncation to preserve all data
    full_inputs = tokenizer(
        examples[text_column],
        add_special_tokens=True,
    )

    # Convert to PyTorch tensors
    input_ids = [
        torch.tensor(sample, dtype=torch.long) for sample in full_inputs["input_ids"]
    ]
    targets = [
        torch.tensor(sample, dtype=torch.long) for sample in full_inputs["input_ids"]
    ]
    attention_mask = [
        torch.tensor(sample, dtype=torch.long)
        for sample in full_inputs["attention_mask"]
    ]

    if not concatenate:
        # Without concatenation, chunk each sample independently into max_tokens pieces
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )
        new_input_ids = []
        new_labels = []
        new_attention_mask = []

        for ids, tgts, mask in zip(input_ids, targets, attention_mask, strict=False):
            sample_len = ids.numel()
            for start in range(0, sample_len, max_tokens):
                end = min(start + max_tokens, sample_len)
                chunk_len = end - start

                chunk_ids = torch.full((max_tokens,), pad_id, dtype=torch.long)
                chunk_labels = torch.full((max_tokens,), -100, dtype=torch.long)
                chunk_mask = torch.zeros((max_tokens,), dtype=torch.long)

                chunk_ids[:chunk_len] = ids[start:end]
                chunk_labels[:chunk_len] = tgts[start:end]
                chunk_mask[:chunk_len] = mask[start:end]

                new_input_ids.append(chunk_ids)
                new_labels.append(chunk_labels)
                new_attention_mask.append(chunk_mask)

        LOG.debug("encode_streaming (no concat): created %d chunks", len(new_input_ids))
        return {
            "input_ids": [seq.tolist() for seq in new_input_ids],
            "labels": [seq.tolist() for seq in new_labels],
            "attention_mask": [seq.tolist() for seq in new_attention_mask],
        }

    # --- concatenate=True path ---

    # Append EOS and PAD tokens to mark document boundaries (before concatenation)
    for i, _ in enumerate(input_ids):
        input_ids[i] = torch.cat(
            (
                input_ids[i],
                torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id]),
            ),
            dim=0,
        )
        targets[i] = torch.cat(
            (
                targets[i],
                torch.tensor([tokenizer.eos_token_id, -100]),
            ),
            dim=0,
        )
        attention_mask[i] = torch.cat((attention_mask[i], torch.tensor([1, 0])), dim=0)

    # Concatenate all samples into a single stream
    all_input_ids = torch.cat(input_ids, dim=0)
    all_targets = torch.cat(targets, dim=0)
    all_attention_mask = torch.cat(attention_mask, dim=0)

    total_len = all_input_ids.numel()

    # Resolve a safe pad token id for chunk padding
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )

    if tokenizer.pad_token_id is None:
        LOG.warning(
            "tokenizer.pad_token_id is None; falling back to %s for padding", pad_id
        )

    new_input_ids = []
    new_labels = []
    new_attention_mask = []

    # Split the concatenated stream into max_tokens-sized chunks
    for start in range(0, total_len, max_tokens):
        end = min(start + max_tokens, total_len)
        chunk_len = end - start

        chunk_ids = torch.full((max_tokens,), pad_id, dtype=torch.long)
        chunk_labels = torch.full((max_tokens,), -100, dtype=torch.long)
        chunk_mask = torch.zeros((max_tokens,), dtype=torch.long)

        chunk_ids[:chunk_len] = all_input_ids[start:end]
        chunk_labels[:chunk_len] = all_targets[start:end]
        chunk_mask[:chunk_len] = all_attention_mask[start:end]

        new_input_ids.append(chunk_ids)
        new_labels.append(chunk_labels)
        new_attention_mask.append(chunk_mask)

    LOG.debug("encode_streaming: created %d chunks", len(new_input_ids))

    return {
        "input_ids": [seq.tolist() for seq in new_input_ids],
        "labels": [seq.tolist() for seq in new_labels],
        "attention_mask": [seq.tolist() for seq in new_attention_mask],
    }


def wrap_streaming_dataset(
    dataset,
    tokenizer,
    cfg,
    ds_wrapper_fn,
):
    """Wrap a streaming dataset with encoding/packing logic."""
    if cfg.sample_packing:
        # For SFT (non-pretraining) datasets, always use multipack_attn=True to ensure
        # attention isolation between packed sequences
        multipack_attn = (
            True if not cfg.pretraining_dataset else cfg.pretrain_multipack_attn
        )

        collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=cfg.sequence_len,
            multipack_attn=multipack_attn,
        )
        encode = functools.partial(
            encode_packed_streaming,
            collate_fn,
            ds_wrapper_fn,
            max_seq_length=cfg.sequence_len,
            batch_size=cfg.micro_batch_size,
            multipack_attn=multipack_attn,
            bin_size=cfg.sample_packing_bin_size,
            is_pretraining=bool(cfg.pretraining_dataset),
        )

        # Set this to 1 so downstream data_loader doesn't try to increase the batch size
        # again
        cfg.micro_batch_size = 1
    else:
        # NOTE: This is not reachable for SFT datasets since we use the pre-existing
        # loading function for non-packed streaming datasets. Refer to
        # _prepare_streaming_datasets in sft.py for that code path.
        text_column = (
            getattr(cfg.pretraining_dataset[0], "text_column", "text") or "text"
        )
        encode = functools.partial(
            encode_streaming,
            tokenizer=tokenizer,
            max_tokens=cfg.sequence_len,
            text_column=text_column,
            concatenate=cfg.pretraining_sample_concatenation is True,
        )

    if cfg.shuffle_merged_datasets:
        dataset = dataset.shuffle(
            seed=cfg.seed, buffer_size=cfg.streaming_multipack_buffer_size
        )
    else:
        LOG.debug("NOT shuffling merged pretraining datasets")

    # remove all the existing columns after mapping since they end up having
    # a different length than the encoded/tokenized column
    # this is empty during streaming/pretraining
    remove_columns = []
    if dataset.features is None:
        for first_row in dataset:
            remove_columns = list(first_row.keys())
            break
    else:
        remove_columns = list(dataset.features.keys())

    dataset = dataset.map(
        encode,
        batched=True,
        batch_size=cfg.streaming_multipack_buffer_size,
        remove_columns=remove_columns,
    )
    return dataset


def _chunk_long_sequences(
    train_dataset: Dataset,
    max_seq_length: int,
) -> Dict[str, list]:
    """
    Chunk sequences longer than max_seq_length into multiple smaller sequences.

    Instead of dropping long sequences (which loses data), this function splits
    them into max_seq_length-sized chunks. Uses numpy for fast array splitting
    to avoid the overhead of pure-Python list slicing on very long sequences
    (100K+ tokens).

    This is especially useful for pretraining datasets with very long samples
    (e.g., millions of tokens per example).

    Note: This should only be used for pretraining. For SFT, long sequences should
    be dropped to maintain complete instruction-response pairs.

    Note: The last chunk of a split sequence may be shorter than max_seq_length.
    This is intentional to preserve all data; the downstream packer handles
    variable-length sequences.

    Args:
        train_dataset: Dataset with input_ids, attention_mask, and optionally labels
        max_seq_length: Maximum sequence length for each chunk

    Returns:
        Dict of lists with all sequences <= max_seq_length
    """
    columns = train_dataset.column_names
    has_labels = "labels" in columns
    has_attention_mask = "attention_mask" in columns

    # Batch column access avoids per-element __getitem__ overhead
    all_input_ids: list = train_dataset["input_ids"]
    all_attention_mask: list = (
        train_dataset["attention_mask"] if has_attention_mask else []
    )
    all_labels: list = train_dataset["labels"] if has_labels else []

    total_samples = len(all_input_ids)
    seq_lengths = [len(ids) for ids in all_input_ids]
    n_long = sum(1 for s in seq_lengths if s > max_seq_length)

    if n_long == 0:
        result: dict = {"input_ids": list(all_input_ids)}
        if has_attention_mask:
            result["attention_mask"] = list(all_attention_mask)
        if has_labels:
            result["labels"] = list(all_labels)
        return result

    new_input_ids: list = []
    new_attention_mask_list: list = []
    new_labels_list: list = []
    total_chunks = 0
    long_samples = 0

    for i in range(total_samples):
        seq_len = seq_lengths[i]

        if seq_len <= max_seq_length:
            new_input_ids.append(all_input_ids[i])
            if has_attention_mask:
                new_attention_mask_list.append(all_attention_mask[i])
            if has_labels:
                new_labels_list.append(all_labels[i])
            total_chunks += 1
        else:
            long_samples += 1
            # Use numpy for fast splitting: converts to contiguous array once,
            # then np.split creates views (no copy), and tolist() converts back
            split_points = list(range(max_seq_length, seq_len, max_seq_length))

            ids_arr = np.array(all_input_ids[i], dtype=np.int64)
            id_chunks = np.split(ids_arr, split_points)
            new_input_ids.extend(c.tolist() for c in id_chunks)

            if has_attention_mask:
                mask_arr = np.array(all_attention_mask[i], dtype=np.int64)
                mask_chunks = np.split(mask_arr, split_points)
                new_attention_mask_list.extend(c.tolist() for c in mask_chunks)

            if has_labels:
                labels_arr = np.array(all_labels[i], dtype=np.int64)
                label_chunks = np.split(labels_arr, split_points)
                new_labels_list.extend(c.tolist() for c in label_chunks)

            total_chunks += len(id_chunks)

    LOG.info(
        "Chunked %d/%d sequences exceeding max_seq_length=%d: %d samples -> %d chunks",
        long_samples,
        total_samples,
        max_seq_length,
        total_samples,
        total_chunks,
    )

    new_data: dict = {"input_ids": new_input_ids}
    if has_attention_mask:
        new_data["attention_mask"] = new_attention_mask_list
    if has_labels:
        new_data["labels"] = new_labels_list

    return new_data


def encode_packed_streaming(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    bin_size: int,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
    is_pretraining: bool = False,
) -> Dict[str, List]:
    """
    Encode examples for sample packing with streaming support.

    This function tokenizes examples, optionally chunks long sequences (for pretraining),
    and then packs them together efficiently using MultipackBatchSampler.

    For pretraining: long sequences are chunked to preserve data.
    For SFT: long sequences are dropped to maintain complete instruction-response pairs.

    Args:
        collate_fn: Collator function for batching
        ds_wrapper: Dataset wrapper function for tokenization
        examples: Raw examples to process
        bin_size: Bin size for multipack sampler
        max_seq_length: Maximum sequence length
        batch_size: Micro batch size
        multipack_attn: Whether to use multipack attention
        is_pretraining: If True, chunk long sequences; if False, let them be dropped
    """
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]

    if is_pretraining:
        chunked = _chunk_long_sequences(train_dataset, max_seq_length)

        # Build plain list-of-dicts directly instead of going through HF Dataset
        # filter/map (process_pretraining_datasets_for_packing). This avoids
        # Arrow serialization overhead which is significant with 10K+ chunks.
        # Replicates the same logic: drop short/long seqs, add position_ids,
        # drop attention_mask when using multipack attention.
        samples: list = []
        lengths: list = []
        skip_position_ids = not multipack_attn
        all_ids = chunked["input_ids"]
        all_labels = chunked.get("labels")
        all_mask = chunked.get("attention_mask")

        for idx in range(len(all_ids)):
            ids = all_ids[idx]
            seq_len = len(ids)
            if seq_len < 2 or seq_len > max_seq_length:
                continue
            sample: dict = {"input_ids": ids}
            if all_labels:
                sample["labels"] = all_labels[idx]
            if not skip_position_ids:
                sample["position_ids"] = list(range(seq_len))
            if all_mask and not multipack_attn:
                sample["attention_mask"] = all_mask[idx]
            sample["length"] = seq_len
            samples.append(sample)
            lengths.append(seq_len)

        lengths_arr = np.array(lengths, dtype=np.int64)
    else:
        # SFT path: use HF Dataset processing as before
        train_dataset = process_pretraining_datasets_for_packing(
            train_dataset,
            max_seq_length,
            skip_position_ids=not multipack_attn,
            drop_attention_mask=multipack_attn,
        )
        lengths_arr = get_dataset_lengths(train_dataset)
        samples = [train_dataset[i] for i in range(len(train_dataset))]

    sampler = MultipackBatchSampler(
        sampler=RandomSampler(range(len(samples))),
        lengths=lengths_arr,
        batch_size=1,
        batch_max_len=batch_size * max_seq_length,
        drop_last=True,
        num_processes=1,
        bin_size=bin_size,
    )

    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            if isinstance(data, list):
                features = {
                    key: [samples[i][key] for i in data] for key in samples[data[0]]
                }
            else:
                features = samples[data]
            if "num_truncated_tokens" in features:
                del features["num_truncated_tokens"]
            if "overflow_to_sample_mapping" in features:
                del features["overflow_to_sample_mapping"]
            if "labels" not in features:
                features["labels"] = features["input_ids"].copy()
            collated_features = collate_fn(features)

            for feature in features.keys():
                if feature == "length":
                    continue
                chunked_data[feature].append(collated_features[feature].squeeze(0))

    return chunked_data
