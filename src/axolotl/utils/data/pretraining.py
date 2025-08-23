"""data handling specific to pretraining"""

import functools
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = get_logger(__name__)


def encode_pretraining(
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    examples: Dict[str, List],
    text_column: str = "text",
    concatenate: bool = True,
) -> Dict[str, List]:
    full_inputs = tokenizer(
        examples[text_column],
        add_special_tokens=True,
    )

    # Convert input_ids and attention_mask to tensors
    full_inputs["input_ids"] = [
        torch.tensor(sample, dtype=torch.long) for sample in full_inputs["input_ids"]
    ]
    full_inputs["attention_mask"] = [
        torch.tensor(sample, dtype=torch.long)
        for sample in full_inputs["attention_mask"]
    ]

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

    inputs_ids, target_ids, attention_mask = [], [], []

    # Concatenate if specified all input_ids and attention masks into one tensor when concatenate is True
    if concatenate:
        full_inputs["input_ids"] = [torch.cat(full_inputs["input_ids"], dim=0)]
        full_inputs["attention_mask"] = [
            torch.cat(full_inputs["attention_mask"], dim=0)
        ]

    # Iterate through each sample and split into chunks of max_tokens
    for sample_index in range(len(full_inputs["input_ids"])):
        for text_index in range(
            0, len(full_inputs["input_ids"][sample_index]), max_tokens
        ):
            # Create partial tensors for inputs, targets, and attention masks with fill values
            partial_inputs_ids = torch.full((max_tokens,), pad_id, dtype=torch.long)
            partial_target_ids = torch.full((max_tokens,), -100, dtype=torch.long)
            partial_attention_mask = torch.zeros((max_tokens,), dtype=torch.long)

            # Determine the length of the text to copy
            text_length = min(
                max_tokens,
                len(full_inputs["input_ids"][sample_index]) - text_index,
            )

            # Copy the text into the partial tensors
            partial_inputs_ids[:text_length] = full_inputs["input_ids"][sample_index][
                text_index : text_index + text_length
            ]
            partial_target_ids[:text_length] = full_inputs["input_ids"][sample_index][
                text_index : text_index + text_length
            ]
            partial_attention_mask[:text_length] = full_inputs["attention_mask"][
                sample_index
            ][text_index : text_index + text_length]

            # Append the partial tensors to the lists
            inputs_ids.append(partial_inputs_ids)
            target_ids.append(partial_target_ids)
            attention_mask.append(partial_attention_mask)

    LOG.debug("Input IDs length: %s", len(inputs_ids))

    return {
        "input_ids": [input_id.tolist() for input_id in inputs_ids],
        "labels": [target_id.tolist() for target_id in target_ids],
        "attention_mask": [mask.tolist() for mask in attention_mask],
    }


def wrap_pretraining_dataset(
    dataset,
    tokenizer,
    cfg,
    ds_wrapper_fn,
    max_tokens=2048,
    batch_size=1,
    seed=42,
    buffer_size=10_000,
):
    if cfg.sample_packing:
        collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=max_tokens,
            multipack_attn=cfg.pretrain_multipack_attn,
        )
        encode = functools.partial(
            encode_packed_pretraining,
            collate_fn,
            ds_wrapper_fn,
            max_seq_length=max_tokens,
            batch_size=batch_size,
            multipack_attn=cfg.pretrain_multipack_attn,
        )
        # set this to 1 so downstream data_loader doesn't try to increase the batch again
        cfg.micro_batch_size = 1
    else:
        encode = functools.partial(
            encode_pretraining,
            tokenizer,
            max_tokens,
            text_column=cfg.pretraining_dataset[0].text_column or "text",
            concatenate=cfg.pretraining_sample_concatenation is True,
        )

    if cfg.shuffle_merged_datasets:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
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
        batch_size=buffer_size,
        # input_columns="text",
        remove_columns=remove_columns,
    )
    return dataset


def encode_packed_pretraining(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
) -> Dict[str, List]:
    # pylint: disable=duplicate-code
    # tokenize all the examples
    # rows get split with stride (overlap)
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]

    train_dataset = process_pretraining_datasets_for_packing(
        train_dataset,
        max_seq_length,
        skip_position_ids=not multipack_attn,
        # FIXME using attention mask unpad/pad with trainer and packed pretraining is broken atm
        # workaround by using the position id logic for now in trainer
        drop_attention_mask=multipack_attn,
    )

    sampler = MultipackBatchSampler(
        sampler=RandomSampler(train_dataset),
        lengths=get_dataset_lengths(train_dataset),
        batch_size=1,
        batch_max_len=batch_size * max_seq_length,
        drop_last=True,
        num_processes=1,
    )

    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            features = train_dataset[data]
            if "num_truncated_tokens" in features:
                del features["num_truncated_tokens"]
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
