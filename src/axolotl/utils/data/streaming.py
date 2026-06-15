"""Data handling specific to streaming datasets."""

import functools
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase, ProcessorMixin

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

    for ids, labels, mask in zip(input_ids, targets, attention_mask, strict=False):
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


def encode_streaming_multimodal(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    image_token: str,
    image_token_id: int,
    text_column: str = "text",
    image_column: str = "images",
) -> Dict[str, List]:
    texts: List[str] = examples[text_column]
    imgs_list: List[List[str]] = examples[image_column]

    if len(texts) != len(imgs_list):
        raise ValueError(
            f"encode_streaming_multimodal: text column has {len(texts)} rows "
            f"but image column has {len(imgs_list)}"
        )

    input_ids: List[List[int]] = []
    labels: List[List[int]] = []
    attention_mask: List[List[int]] = []
    keep_images: List[List[str]] = []
    keep_text: List[str] = []

    for text, imgs in zip(texts, imgs_list, strict=True):
        if not isinstance(text, str):
            raise TypeError(
                f"encode_streaming_multimodal: `{text_column}` must be str, "
                f"got {type(text).__name__}."
            )
        if imgs is None:
            imgs = []
        if not isinstance(imgs, (list, tuple)):
            raise ValueError(
                f"encode_streaming_multimodal: row's `{image_column}` must be "
                f"a list; got {type(imgs).__name__}"
            )
        for j, ip in enumerate(imgs):
            if not isinstance(ip, str):
                raise TypeError(
                    f"encode_streaming_multimodal: image {j} in row must be "
                    f"str, got {type(ip).__name__}."
                )
        # No truncation: counting on truncated ids and storing untruncated text
        # (which the collator re-tokenizes without truncation) silently produces
        # oversize batches and confusing placeholder/image-count mismatches.
        enc = tokenizer(text, add_special_tokens=True)
        ids = list(enc["input_ids"]) + [tokenizer.eos_token_id]
        mask = list(enc["attention_mask"]) + [1]
        # Count by id — `text.count` substring-matches `<image>` in `<image_soft_token>`.
        n_placeholders = sum(1 for t in ids if t == image_token_id)
        if n_placeholders != len(imgs):
            raise ValueError(
                f"Multimodal CPT row has {n_placeholders} occurrence(s) of "
                f"{image_token!r} in text but {len(imgs)} image path(s). "
                f"Text and image count must match (one placeholder per image)."
            )
        if len(ids) > max_tokens:
            raise ValueError(
                f"Multimodal CPT row tokenizes to {len(ids)} tokens which "
                f"exceeds sequence_len={max_tokens}. Pre-chunk your text or "
                f"raise sequence_len (image patch expansion at the processor "
                f"may push the final length even higher)."
            )
        # Labels = ids; collator masks image-family ids after re-tokenization.
        input_ids.append(ids)
        labels.append(list(ids))
        attention_mask.append(mask)
        keep_images.append(list(imgs))
        keep_text.append(text)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": keep_images,
        "_mm_text": keep_text,
    }


def wrap_streaming_dataset(
    dataset,
    tokenizer,
    cfg,
    ds_wrapper_fn,
    processor: Optional[ProcessorMixin] = None,
    pretraining_config=None,
    is_eval: bool = False,
):
    # Eval streams honor cfg.eval_sequence_len when set, else cfg.sequence_len.
    effective_seq_len = (
        cfg.eval_sequence_len
        if is_eval and getattr(cfg, "eval_sequence_len", None)
        else cfg.sequence_len
    )
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
        )

        # Set this to 1 so downstream data_loader doesn't try to increase the batch size
        # again
        cfg.micro_batch_size = 1
    else:
        # NOTE: This is not reachable for SFT datasets since we use the pre-existing
        # loading function for non-packed streaming datasets. Refer to
        # _prepare_streaming_datasets in sft.py for that code path.
        # Prefer the resolved per-entry config so eval (test_datasets) doesn't
        # silently inherit the training entry's columns/image_token.
        if pretraining_config is not None:
            ds_first = pretraining_config
        elif cfg.pretraining_dataset:
            ds_first = cfg.pretraining_dataset[0]
        else:
            ds_first = {}
        # Plain dicts need `.get`; pydantic/DictDefault need `getattr`.
        get_ds_value = (
            ds_first.get
            if isinstance(ds_first, dict)
            else lambda key, default=None: getattr(ds_first, key, default)
        )
        text_column = get_ds_value("text_column", "text") or "text"
        ds_type = (get_ds_value("type", None) or "").strip()
        is_mm_cpt = ds_type == "multimodal_pretrain" or bool(
            get_ds_value("multimodal", False)
        )

        if is_mm_cpt:
            if processor is None:
                raise ValueError(
                    "Multimodal CPT (type: multimodal_pretrain) requires a "
                    "processor. Set `processor_type: AutoProcessor` (or the "
                    "concrete processor class) in your config."
                )
            from axolotl.prompt_strategies.multimodal_pretrain import (
                build_image_token_spec,
                check_processor_compatibility,
            )

            check_processor_compatibility(processor)
            spec = build_image_token_spec(
                processor,
                override=get_ds_value("image_token", None),
            )
            image_column = get_ds_value("image_column", None) or "images"
            LOG.info(
                f"multimodal streaming CPT: placeholder={spec.image_token!r} "
                f"(id={spec.image_token_id})"
            )
            encode = functools.partial(
                encode_streaming_multimodal,
                tokenizer=tokenizer,
                max_tokens=effective_seq_len,
                image_token=spec.image_token,
                image_token_id=spec.image_token_id,
                text_column=text_column,
                image_column=image_column,
            )
        else:
            encode = functools.partial(
                encode_streaming,
                tokenizer=tokenizer,
                max_tokens=effective_seq_len,
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


def encode_packed_streaming(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    bin_size: int,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
) -> Dict[str, List]:
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
        bin_size=bin_size,
    )

    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            features = train_dataset[data]
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
