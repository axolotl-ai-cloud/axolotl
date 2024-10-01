"""data handling specific to pretraining"""

import functools
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = logging.getLogger("axolotl")


def encode_pretraining_multimodal(processor: ProcessorMixin, max_tokens: int, examples: Dict[str, List]) -> Dict[str, list]:
    def format_conversation(messages):
        """
        Concatenate the conversation messages from the 'messages' field in a structured way.
        """
        conversation = []
        for message in messages:
            for content in message['content']:
                if content['type'] == 'text' and content['text'] is not None:
                    conversation.append(content['text'])
                elif content['type'] == 'image':  # Assuming 'image' is the type for images
                    conversation.append(processor.image_token)  # Insert image token
        return "\n".join(conversation)

    texts = []
    for example in examples["messages"]:
        # Step 1: Process text by concatenating messages
        conversation_text = format_conversation(example)
        texts.append(conversation_text)

        # Step 2: Process images
        images = examples['images']  # [0] if len(example['images']) > 0 else self.get_placeholder_image()

    res = processor(
        text=texts,
        images=images,
        truncation=True,
        max_length=max_tokens - 2,
        add_special_tokens=True,
    )
    # Convert to PyTorch tensors
    input_ids = [torch.tensor(seq) for seq in res["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in res["attention_mask"]]
    pixel_values = [torch.tensor(seq) for seq in res["pixel_values"]]
    cross_attention_mask = [torch.tensor(seq) for seq in res["cross_attention_mask"]]
    aspect_ratio_mask = [torch.tensor(seq) for seq in res["aspecti_ratio_mask"]]
    aspect_ratio_ids = [torch.tensor(seq) for seq in res["aspect_ratio_ids"]]

    ret = {
        "input_ids": [seq.tolist() for seq in input_ids],
        "attention_mask": [seq.tolist() for seq in attention_mask],
        "pixel_values": [seq.tolist() for seq in pixel_values],
        "cross_attention_mask": [seq.tolist() for seq in cross_attention_mask],
        "aspect_ratio_mask": [seq.tolist() for seq in aspect_ratio_mask],
        "aspect_ratio_ids": [seq.tolist() for seq in aspect_ratio_ids],
    }
    return ret


def encode_pretraining(
    tokenizer: PreTrainedTokenizerBase, max_tokens: int, examples: Dict[str, List]
) -> Dict[str, List]:
    res = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_tokens - 2,
        add_special_tokens=True,
    )
    # Convert to PyTorch tensors
    input_ids = [torch.tensor(seq) for seq in res["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in res["attention_mask"]]
    new_input_ids = []
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
        attention_mask[i] = torch.cat((attention_mask[i], torch.tensor([1, 0])), dim=0)

    # Concatenate tokens so that their lengths are less than max_tokens
    buffer_input_ids = torch.tensor([], dtype=torch.long)
    buffer_attention_mask = torch.tensor([], dtype=torch.long)

    for ids, mask in zip(input_ids, attention_mask):
        if buffer_input_ids.numel() == max_tokens:
            new_input_ids.append(buffer_input_ids)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        elif buffer_input_ids.numel() + ids.numel() <= max_tokens:
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
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
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)

            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
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
        new_attention_mask.append(buffer_attention_mask)

    ret = {
        "input_ids": [seq.tolist() for seq in new_input_ids],
        "labels": [seq.tolist() for seq in new_input_ids],
        "attention_mask": [seq.tolist() for seq in new_attention_mask],
    }

    LOG.debug(len(ret["input_ids"]))
    return ret


def wrap_pretraining_dataset(
    dataset,
    tokenizer_processor,
    cfg,
    ds_wrapper_fn,
    max_tokens=2048,
    batch_size=1,
    seed=42,
    buffer_size=10_000,
):
    if cfg.sample_packing:
        collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
            tokenizer_processor,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=max_tokens * batch_size,
            multipack_attn=cfg.pretrain_multipack_attn,
        )
        encode = functools.partial(
            encode_packed_pretraining,
            collate_fn,
            ds_wrapper_fn,
            max_seq_length=max_tokens,
            batch_size=batch_size,
            multipack_attn=cfg.pretrain_multipack_attn,
            group_size=cfg.sample_packing_group_size,
            bin_size=cfg.sample_packing_bin_size,
        )
        # set this to 1 so downstream data_loader doesn't try to increase the batch again
        cfg.micro_batch_size = 1
    else:
        if cfg.is_multmodal:
            processor = tokenizer_processor
            encode = functools.partial(encode_pretraining_multimodal, processor, max_tokens)
        else:
            encode = functools.partial(encode_pretraining, tokenizer_processor, max_tokens)

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
    multipack_attn: Optional[bool] = False,
    group_size: int = 100000,
    bin_size: int = 200,
) -> Dict[str, List]:
    # pylint: disable=duplicate-code
    # tokenize all the examples
    # rows get split with stride (overlap)
    train_dataset = ds_wrapper(Dataset.from_dict(examples))[0]

    train_dataset = process_pretraining_datasets_for_packing(
        train_dataset,
        max_seq_length,
        skip_position_ids=not multipack_attn,
    )

    sampler = MultipackBatchSampler(
        sampler=RandomSampler(train_dataset),
        lengths=get_dataset_lengths(train_dataset),
        batch_size=1,
        batch_max_len=batch_size * max_seq_length,
        group_size=group_size,
        bin_size=bin_size,
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
