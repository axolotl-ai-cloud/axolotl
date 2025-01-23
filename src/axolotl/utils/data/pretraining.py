"""data handling specific to pretraining"""

import functools
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = logging.getLogger("axolotl")


def encode_pretraining(
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    examples: Dict[str, List],
    text_column: str = "text",
    concatenate: bool = True,
) -> Dict[str, List]:
    res = tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_tokens - 2,
        add_special_tokens=True,
    )
    # Convert to PyTorch tensors
    input_ids = [torch.tensor(seq) for seq in res["input_ids"]]
    targets = [torch.tensor(seq) for seq in res["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in res["attention_mask"]]
    if not concatenate:
        return {
            "input_ids": [seq.tolist() for seq in input_ids],
            "labels": [seq.tolist() for seq in targets],
            "attention_mask": [seq.tolist() for seq in attention_mask],
        }

    new_input_ids = []
    new_labels = []
    new_attention_mask = []
    # Append EOS and PAD tokens to input_ids, and correct attention_mask
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

    # Concatenate tokens so that their lengths are less than max_tokens
    buffer_input_ids = torch.tensor([], dtype=torch.long)
    buffer_labels = torch.tensor([], dtype=torch.long)
    buffer_attention_mask = torch.tensor([], dtype=torch.long)

    for ids, labels, mask in zip(input_ids, targets, attention_mask):
        if buffer_input_ids.numel() == max_tokens:
            new_input_ids.append(buffer_input_ids)
            new_labels.append(buffer_labels)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_labels = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_labels = torch.cat((buffer_labels, labels), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        elif buffer_input_ids.numel() + ids.numel() <= max_tokens:
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_labels = torch.cat((buffer_labels, labels), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        else:
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_labels = torch.cat(
                (
                    buffer_labels,
                    torch.full(
                        (max_tokens - buffer_labels.numel(),),
                        -100,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            new_input_ids.append(buffer_input_ids)
            new_labels.append(buffer_labels)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_labels = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)

            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_labels = torch.cat((buffer_labels, labels), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)

    if buffer_input_ids.numel() > 0:  # for any leftover tokens
        while buffer_input_ids.numel() < max_tokens:  # make all sequences equal in size
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_labels = torch.cat(
                (
                    buffer_labels,
                    torch.full(
                        (max_tokens - buffer_labels.numel(),),
                        -100,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
        new_input_ids.append(buffer_input_ids)
        new_labels.append(buffer_labels)
        new_attention_mask.append(buffer_attention_mask)

    ret = {
        "input_ids": [seq.tolist() for seq in new_input_ids],
        "labels": [seq.tolist() for seq in new_labels],
        "attention_mask": [seq.tolist() for seq in new_attention_mask],
    }

    LOG.debug(len(ret["input_ids"]))
    return ret


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
            remove_columns = first_row.keys()
            break
    else:
        remove_columns = dataset.features.keys()

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
    train_dataset = ds_wrapper(Dataset.from_dict(examples))[0]

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
