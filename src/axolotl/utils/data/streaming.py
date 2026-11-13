"""Data handling specific to streaming datasets."""

import functools
import hashlib
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.data.packing import process_pretraining_datasets_for_packing
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.samplers.utils import get_dataset_label_counts

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


def wrap_streaming_dataset(
    dataset,
    tokenizer,
    cfg,
    ds_wrapper_fn,
):
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
            balance_labels=bool(cfg.balance_labels),
            seed=cfg.data_seed if cfg.data_seed is not None else (cfg.seed or 0),
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


def encode_packed_streaming(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    bin_size: int,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
    balance_labels: bool = False,
    seed: int = 0,
) -> Dict[str, List]:
    # tokenize all the examples
    # rows get split with stride (overlap)
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]

    rows: Union[_FlatRows, _DatasetRows, None] = _flat_rows(
        train_dataset, collate_fn, max_seq_length, multipack_attn
    )
    if rows is None:
        rows = _DatasetRows(train_dataset, collate_fn, max_seq_length, multipack_attn)
    if len(rows) == 0:
        # every row in this buffer was outside the sequence length range
        return {}

    lengths = rows.lengths()
    label_counts = label_start_counts = None
    if balance_labels:
        if collate_fn.tokenizer.padding_side != "right":
            raise ValueError("Label-balanced packing requires right padding")
        if rows.labeled:
            label_counts, label_start_counts = rows.label_counts()
        else:
            # Pretraining creates labels from input_ids during collation.
            label_counts = lengths
            label_start_counts = np.ones_like(lengths)

    if balance_labels:
        digest = hashlib.sha256(str(seed).encode("ascii"))
        rows.hash_input_ids(digest)
        seed = int.from_bytes(digest.digest()[:8], "little")

    sampler = MultipackBatchSampler(
        sampler=RandomSampler(rows),
        lengths=lengths,
        label_counts=label_counts,
        label_start_counts=label_start_counts,
        seed=seed,
        padding_multiple=getattr(collate_fn, "pad_to_multiple_of", None) or 1,
        batch_size=1,
        batch_max_len=batch_size * max_seq_length,
        drop_last=True,
        num_processes=1,
        bin_size=bin_size,
    )

    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            for feature, row in rows.pack(data).items():
                chunked_data[feature].append(row)

    return chunked_data


class _DatasetRows:
    """The buffer's rows after the datasets length filter and transforms, packed by
    the collator."""

    def __init__(
        self,
        train_dataset: Dataset,
        collate_fn,
        max_seq_length: int,
        multipack_attn: Optional[bool],
    ):
        self.dataset = process_pretraining_datasets_for_packing(
            train_dataset,
            max_seq_length,
            skip_position_ids=not multipack_attn,
            # FIXME using attention mask unpad/pad with trainer and packed pretraining is broken atm
            # workaround by using the position id logic for now in trainer
            drop_attention_mask=multipack_attn,
        )
        self.collate_fn = collate_fn
        self.labeled = bool(
            {"labels", "shift_labels"}.intersection(self.dataset.column_names)
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def lengths(self) -> np.ndarray:
        return get_dataset_lengths(self.dataset)

    def label_counts(self) -> Tuple[np.ndarray, np.ndarray]:
        return get_dataset_label_counts(self.dataset)

    def hash_input_ids(self, digest) -> None:
        for batch in self.dataset.select_columns(["input_ids"]).iter(batch_size=1024):
            for ids in batch["input_ids"]:
                digest.update(len(ids).to_bytes(8, "little"))
                digest.update(np.asarray(ids, dtype="<i8").tobytes())

    def pack(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        features = self.dataset[indices]
        if "num_truncated_tokens" in features:
            del features["num_truncated_tokens"]
        if "overflow_to_sample_mapping" in features:
            del features["overflow_to_sample_mapping"]
        if "labels" not in features:
            features["labels"] = features["input_ids"].copy()
        collated_features = self.collate_fn(features)
        return {
            feature: collated_features[feature].squeeze(0)
            for feature in features.keys()
            if feature != "length"
        }


class _FlatRows:
    """The buffer's rows as flat token arrays and row offsets. Filters, measures,
    hashes and packs them as ``_DatasetRows`` does, without its per-row and
    per-token Python work."""

    def __init__(
        self,
        collate_fn,
        names: List[str],
        flat: Dict[str, np.ndarray],
        row_starts: np.ndarray,
        row_lengths: np.ndarray,
        labeled: bool,
    ):
        tokenizer = collate_fn.tokenizer
        self.names = names
        self.flat = flat
        self.row_starts = row_starts
        self.row_lengths = row_lengths
        self.labeled = labeled
        self.multipack_attn = collate_fn.multipack_attn
        self.multiple = collate_fn.pad_to_multiple_of or 1
        self.pad_ids = {
            "input_ids": tokenizer.pad_token_id,
            "labels": collate_fn.label_pad_token_id,
            "attention_mask": 0,
        }
        self.pad_left = tokenizer.padding_side == "left"

    def __len__(self) -> int:
        return len(self.row_lengths)

    def lengths(self) -> np.ndarray:
        return self.row_lengths

    def label_counts(self) -> Tuple[np.ndarray, np.ndarray]:
        trained = self.flat["labels"] != -100
        cumulative = np.concatenate(([0], np.cumsum(trained)))
        ends = self.row_starts + self.row_lengths
        counts = cumulative[ends] - cumulative[self.row_starts]
        # the length filter leaves no empty rows
        return counts, trained[self.row_starts].astype(np.int64)

    def hash_input_ids(self, digest) -> None:
        input_ids = self.flat["input_ids"]
        for start, length in zip(
            self.row_starts.tolist(), self.row_lengths.tolist(), strict=True
        ):
            digest.update(length.to_bytes(8, "little"))
            digest.update(input_ids[start : start + length].astype("<i8").tobytes())

    def pack(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        lengths = self.row_lengths[indices]
        positions = np.arange(lengths.sum(), dtype=np.int64) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        gather = np.repeat(self.row_starts[indices], lengths) + positions
        pad_len = -len(positions) % self.multiple
        packed = {}
        for name in self.names:
            if name == "position_ids":
                values = positions
                pad = np.arange(pad_len, dtype=np.int64)
            else:
                values = self.flat[name][gather]
                if name == "attention_mask" and self.multipack_attn:
                    values = values * np.repeat(np.arange(1, len(indices) + 1), lengths)
                pad = np.full(pad_len, self.pad_ids[name], dtype=np.int64)
            packed[name] = torch.from_numpy(
                np.concatenate((pad, values) if self.pad_left else (values, pad))
            )
        return packed


# _DatasetRows drops these; its collator concatenates and pads the flat ones
_DROPPED_COLUMNS = ("num_truncated_tokens", "overflow_to_sample_mapping")
_FLAT_COLUMNS = ("input_ids", "labels", "attention_mask")


def _flat_rows(
    train_dataset: Dataset,
    collate_fn,
    max_seq_length: int,
    multipack_attn: Optional[bool],
) -> Optional[_FlatRows]:
    """``_FlatRows`` for the tokenized buffer, or None when only ``_DatasetRows``
    can reproduce its transforms and collator."""
    tokenizer = collate_fn.tokenizer
    column_names = train_dataset.column_names
    # under multipack_attn, process_pretraining_datasets_for_packing swaps the
    # attention mask for position_ids
    names = [
        name
        for name in column_names
        if name not in _DROPPED_COLUMNS
        and not (multipack_attn and name == "attention_mask")
    ]
    if (
        type(collate_fn) is not PretrainingBatchSamplerDataCollatorForSeq2Seq
        or "input_ids" not in names
        or not set(names).issubset(_FLAT_COLUMNS)
        # removing the attention mask raises without one
        or (multipack_attn and "attention_mask" not in column_names)
        or tokenizer.model_input_names[0] != "input_ids"
        or (
            "attention_mask" in names
            and "attention_mask" not in tokenizer.model_input_names
        )
        or tokenizer.pad_token is None
        or tokenizer.pad_token_id is None
        or tokenizer.pad_token_id < 0
        or tokenizer.padding_side not in ("left", "right")
        or collate_fn.padding is not True
        or collate_fn.max_length is not None
        or collate_fn.return_tensors != "pt"
        or collate_fn.model is not None
        or collate_fn.cp_shard is not None
        or collate_fn.emit_fa_varlen_kwargs
    ):
        return None

    # with_format would deep-copy the dataset, and with it the wrapper's tokenizer
    with train_dataset.formatted_as("arrow"):
        columns = {name: train_dataset[name].combine_chunks() for name in names}
    if not all(
        (pa.types.is_list(column.type) or pa.types.is_large_list(column.type))
        and pa.types.is_integer(column.type.value_type)
        and column.null_count == 0
        and column.flatten().null_count == 0
        for column in columns.values()
    ):
        return None
    offsets = _row_offsets(columns["input_ids"])
    # unequal row lengths across columns take the collator's per-column padding
    if not all(
        np.array_equal(_row_offsets(column), offsets) for column in columns.values()
    ):
        return None
    flat = {
        name: column.flatten().to_numpy().astype(np.int64, copy=False)
        for name, column in columns.items()
    }
    labeled = "labels" in flat
    if multipack_attn:
        names.append("position_ids")
    if not labeled:
        names.append("labels")
        flat["labels"] = flat["input_ids"]

    row_lengths = np.diff(offsets)
    # filter_sequences_by_length's default range
    keep = (row_lengths >= 2) & (row_lengths <= max_seq_length)
    return _FlatRows(
        collate_fn, names, flat, offsets[:-1][keep], row_lengths[keep], labeled
    )


def _row_offsets(column: pa.ListArray) -> np.ndarray:
    offsets = column.offsets.to_numpy().astype(np.int64)
    return offsets - offsets[0]
