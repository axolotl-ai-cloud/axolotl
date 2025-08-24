"""Simple streaming SFT with multipack support."""

import functools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from datasets import Dataset, IterableDataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = get_logger(__name__)


class StreamingMultipackDataset:
    """Dataset that handles streaming with multipack on-the-fly."""

    def __init__(
        self,
        base_dataset: IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        cfg,
        dataset_config,
        d_base_type: str,
        d_prompt_style: str | None,
        processor: Any | None,
        max_tokens: int = 2048,
        pack_length: int = 4,  # How many sequences to collect before packing
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.max_tokens = max_tokens
        self.pack_length = pack_length

        # Create the dataset wrapper once
        from axolotl.utils.data.wrappers import get_dataset_wrapper

        # Create a dummy dataset to get the wrapper
        dummy_data = {"text": ["dummy"], "instruction": ["dummy"], "output": ["dummy"]}
        dummy_dataset = Dataset.from_dict(dummy_data)

        self.dataset_wrapper, _ = get_dataset_wrapper(
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            cfg=cfg,
            dataset_base_type=d_base_type,
            dataset=dummy_dataset,
            dataset_prompt_style=d_prompt_style,
            processor=processor,
        )

        # Create collator for packing
        self.collator = PretrainingBatchSamplerDataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=max_tokens,
            multipack_attn=cfg.pretrain_multipack_attn,
        )

    def __iter__(self):
        """Iterator that yields packed samples."""
        buffer = []

        for sample in self.base_dataset:
            # Convert single sample to dataset for processing
            temp_dataset = Dataset.from_dict({k: [v] for k, v in sample.items()})

            # Tokenize using the dataset wrapper
            try:
                tokenized = self.dataset_wrapper.__class__(
                    temp_dataset,
                    **{
                        k: v
                        for k, v in self.dataset_wrapper.__dict__.items()
                        if not k.startswith("_")
                    },
                )

                # Get the tokenized sample
                if len(tokenized) > 0:
                    tokenized_sample = tokenized[0]

                    # Add to buffer
                    buffer.append(tokenized_sample)

                    # When buffer is full, pack and yield
                    if len(buffer) >= self.pack_length:
                        packed_sample = self._pack_buffer(buffer)
                        if packed_sample:
                            yield packed_sample
                        buffer = []

            except Exception as e:
                LOG.warning(f"Failed to process sample: {e}")
                continue

        # Process remaining buffer
        if buffer:
            packed_sample = self._pack_buffer(buffer)
            if packed_sample:
                yield packed_sample

    def _pack_buffer(self, buffer: List[Dict]) -> Optional[Dict]:
        """Pack a buffer of tokenized samples."""
        if not buffer:
            return None

        try:
            # Create dataset from buffer
            temp_dataset = Dataset.from_list(buffer)

            # Add position_ids and process for packing
            temp_dataset = process_pretraining_datasets_for_packing(
                temp_dataset,
                self.max_tokens,
                skip_position_ids=not self.cfg.pretrain_multipack_attn,
                drop_attention_mask=self.cfg.pretrain_multipack_attn,
            )

            # Use MultipackBatchSampler to create packed batches
            sampler = MultipackBatchSampler(
                sampler=RandomSampler(temp_dataset),
                lengths=get_dataset_lengths(temp_dataset),
                batch_size=1,
                batch_max_len=self.max_tokens,
                drop_last=False,
                num_processes=1,
            )

            # Get packed data
            for batch in sampler:
                if batch and batch[0]:  # Check if we have data
                    features = []
                    for idx in batch[0]:  # batch[0] contains the indices
                        sample = temp_dataset[idx]
                        if "labels" not in sample:
                            sample["labels"] = sample["input_ids"].copy()
                        features.append(sample)

                    # Apply collator to create final packed sample
                    if features:
                        packed = self.collator(features)
                        return {
                            k: v.squeeze(0) if v.dim() > 1 else v
                            for k, v in packed.items()
                        }

            return None

        except Exception as e:
            LOG.warning(f"Failed to pack buffer: {e}")
            return None


def wrap_streaming_sft_dataset_simple(
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
    Wrap a streaming SFT dataset with simple multipack batching.

    This approach processes samples in small groups rather than large batches,
    avoiding the repeated processing issue.
    """

    # Apply shuffling if configured
    if cfg.shuffle_merged_datasets:
        LOG.info(f"Shuffling streaming dataset with buffer_size={buffer_size}")
        dataset = dataset.shuffle(seed=cfg.seed, buffer_size=buffer_size)

    # Create the streaming multipack dataset
    multipack_dataset = StreamingMultipackDataset(
        dataset,
        tokenizer,
        cfg,
        dataset_config,
        d_base_type,
        d_prompt_style,
        processor,
        max_tokens,
        pack_length=max(1, max_tokens // 512),  # Estimate sequences per pack
    )

    # Convert back to IterableDataset
    class IterableWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            return iter(self.dataset)

    wrapped = IterableWrapper(multipack_dataset)

    # Set micro_batch_size to 1 since sequences are already packed
    cfg.micro_batch_size = 1

    return wrapped
