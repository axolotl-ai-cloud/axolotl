"""Module for testing streaming dataset sequence packing"""

import functools
import hashlib
import json
import random
import string

import multiprocess.pool
import numpy as np
import pytest
import torch
from datasets import (
    Dataset,
    Features,
    IterableDataset,
    List,
    Value,
    concatenate_datasets,
)
from torch.utils.data import DataLoader

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.data import (
    get_dataset_wrapper,
    sft,
    streaming,
    wrap_streaming_dataset,
)
from axolotl.utils.dict import DictDefault


class TestPretrainingPacking:
    """
    Test class for packing streaming dataset sequences
    """

    @pytest.fixture
    def random_text(self):
        # seed with random.seed(0) for reproducibility
        random.seed(0)

        # generate row of random text with "words" of between 2 and 10 characters and
        # between 400 to 1200 characters per line
        def rand_txt():
            return " ".join(
                [
                    "".join(
                        random.choices(string.ascii_lowercase, k=random.randint(2, 10))
                    )
                    for _ in range(random.randint(50, 200))
                ]
            )

        # Create a list of 2000 random texts rather than just using it within the
        # generator so the test runs faster
        data = [rand_txt() for _ in range(500)]

        # Create an IterableDataset
        def generator():
            for row in data:
                yield {"text": row}

        return IterableDataset.from_generator(generator)

    @pytest.mark.flaky(retries=1, delay=5)
    def test_packing_stream_dataset(self, tokenizer_huggyllama, random_text):
        dataset = random_text

        cfg = DictDefault(
            {
                "pretraining_dataset": [
                    {
                        "path": "winglian/tiny-shakespeare",
                        "type": "pretrain",
                    }
                ],
                "sample_packing": True,
                "pretrain_multipack_attn": True,
                "pad_to_sequence_len": True,
                "sequence_len": 2048,
                "micro_batch_size": 2,
                "sample_packing_group_size": 100000,
                "sample_packing_bin_size": 200,
            }
        )

        ds_wrapper_partial = functools.partial(
            get_dataset_wrapper,
            cfg.pretraining_dataset[0],
            tokenizer_huggyllama,
            cfg,
            cfg.pretraining_dataset[0]["type"] or "pretrain",
        )

        original_bsz = cfg.micro_batch_size
        train_dataset = wrap_streaming_dataset(
            dataset,
            tokenizer_huggyllama,
            cfg,
            ds_wrapper_partial,
        )

        trainer_loader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=None,
            drop_last=True,
        )
        idx = 0
        for data in trainer_loader:
            if idx > 3:
                break
            assert data["input_ids"].shape == torch.Size(
                [1, original_bsz * cfg.sequence_len]
            )
            assert data["position_ids"].shape == torch.Size(
                [1, original_bsz * cfg.sequence_len]
            )
            assert data["labels"].shape == torch.Size(
                [1, original_bsz * cfg.sequence_len]
            )
            assert "attention_mask" not in data
            # FIXME add back once we fix packing unpad/pad with attention mask
            # assert data["attention_mask"].shape == torch.Size(
            #     [1, original_bsz * cfg.sequence_len]
            # )
            idx += 1


def test_streaming_packing_tokenizes_in_dataloader_workers(
    tmp_path, tokenizer_huggyllama, monkeypatch
):
    data_file = tmp_path / "data.jsonl"
    row = json.dumps({"text": "lorem ipsum dolor sit amet " * 40})
    data_file.write_text(f"{row}\n" * 32)
    pretraining_config = DictDefault(
        {
            "path": "json",
            "name": None,
            "skip": 0,
            "split": "train",
            "data_files": str(data_file),
            "type": "pretrain",
        }
    )
    cfg = DictDefault(
        {
            "pretraining_dataset": [pretraining_config],
            "sample_packing": True,
            "pretrain_multipack_attn": True,
            "sequence_len": 256,
            "micro_batch_size": 2,
            "sample_packing_bin_size": 200,
            "streaming_multipack_buffer_size": 16,
            "dataset_num_proc": 2,
        }
    )
    dataset = sft._load_streaming_dataset(pretraining_config, cfg, tokenizer_huggyllama)

    def no_pool(*args, **kwargs):
        raise AssertionError("streaming buffers should tokenize in-process")

    # datasets runs map(num_proc=1) in a one-process multiprocess pool
    monkeypatch.setattr(multiprocess.pool.Pool, "__init__", no_pool)

    # the worker is daemonic whatever its start method, so it cannot start a pool
    loader = DataLoader(
        dataset, batch_size=1, num_workers=1, multiprocessing_context="fork"
    )
    batch = next(iter(loader))

    assert batch["input_ids"].shape == torch.Size([1, 2 * 256])


def test_streaming_packing_skips_a_buffer_of_overlong_rows(tokenizer_huggyllama):
    overlong = [
        {"role": "user", "content": "word " * 300},
        {"role": "assistant", "content": "word " * 300},
    ]
    short = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    ds_cfg = DictDefault({"path": "unused", "type": "chat_template"})
    cfg = DictDefault(
        {
            "datasets": [ds_cfg],
            "chat_template": "chatml",
            "sample_packing": True,
            "sequence_len": 256,
            "micro_batch_size": 1,
            "sample_packing_bin_size": 200,
            "streaming_multipack_buffer_size": 4,
        }
    )
    ds_wrapper = functools.partial(
        get_dataset_wrapper,
        dataset_config=ds_cfg,
        tokenizer=tokenizer_huggyllama,
        cfg=cfg,
        dataset_base_type="chat_template",
    )
    dataset = wrap_streaming_dataset(
        Dataset.from_dict(
            {"messages": [overlong] * 4 + [short] * 8}
        ).to_iterable_dataset(),
        tokenizer_huggyllama,
        cfg,
        ds_wrapper,
    )

    packed = list(dataset)

    assert packed
    assert all(len(row["input_ids"]) == 256 for row in packed)


def _assert_same_packed_rows(actual, expected):
    assert type(actual) is type(expected)
    assert list(actual) == list(expected)
    for key, rows in expected.items():
        assert len(actual[key]) == len(rows), key
        for got, want in zip(actual[key], rows, strict=True):
            assert got.dtype == want.dtype, key
            assert got.shape == want.shape, key
            assert got.stride() == want.stride(), key
            assert torch.equal(got, want), key


def _assert_same_array(actual, expected):
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "columns,multipack_attn",
    [
        (("attention_mask",), True),
        (("attention_mask",), False),
        (("labels", "attention_mask"), True),
        (("labels", "attention_mask"), False),
        (("labels",), False),
    ],
)
@pytest.mark.parametrize(
    "padding_side,pad_to_multiple_of", [("right", 256), ("left", 256), ("right", None)]
)
def test_flat_rows_match_dataset_rows_on_random_bins(
    tokenizer_huggyllama, columns, multipack_attn, padding_side, pad_to_multiple_of
):
    tokenizer_huggyllama.padding_side = padding_side
    rng = np.random.default_rng(0)
    # rows on and outside both edges of the [2, 256] length filter
    lengths = np.concatenate(([0, 1, 2, 256, 257], rng.integers(0, 300, size=395)))
    input_ids = [rng.integers(3, 32000, n).tolist() for n in lengths]
    data = {"input_ids": input_ids}
    features = {"input_ids": List(Value("int32"))}
    if "labels" in columns:
        data["labels"] = [
            np.where(rng.random(len(ids)) < 0.3, -100, ids).tolist()
            for ids in input_ids
        ]
        features["labels"] = List(Value("int64"))
    if "attention_mask" in columns:
        data["attention_mask"] = [[1] * len(ids) for ids in input_ids]
        features["attention_mask"] = List(Value("int8"))
    data["overflow_to_sample_mapping"] = list(range(len(input_ids)))
    features["overflow_to_sample_mapping"] = Value("int64")
    dataset = Dataset.from_dict(data, features=Features(features))
    # two arrow chunks, the second sliced, with and without an indices mapping
    dataset = concatenate_datasets(
        [dataset.select(range(150)), dataset.select(range(150, len(dataset)))]
    )
    collate = PretrainingBatchSamplerDataCollatorForSeq2Seq(
        tokenizer_huggyllama,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        multipack_attn=multipack_attn,
    )

    for buffer in (dataset, dataset.select(rng.permutation(len(dataset))[:350])):
        actual = streaming._flat_rows(buffer, collate, 256, multipack_attn)
        assert actual is not None
        expected = streaming._DatasetRows(buffer, collate, 256, multipack_attn)
        assert 0 < len(actual) == len(expected) < len(buffer)
        _assert_same_array(actual.lengths(), expected.lengths())
        assert actual.labeled == expected.labeled == ("labels" in columns)
        if expected.labeled:
            for got, want in zip(
                actual.label_counts(), expected.label_counts(), strict=True
            ):
                _assert_same_array(got, want)
        digests = hashlib.sha256(), hashlib.sha256()
        actual.hash_input_ids(digests[0])
        expected.hash_input_ids(digests[1])
        assert digests[0].digest() == digests[1].digest()
        for _ in range(40):
            indices = rng.choice(
                len(expected), size=rng.integers(1, 9), replace=False
            ).tolist()
            _assert_same_packed_rows(
                {key: [row] for key, row in actual.pack(indices).items()},
                {key: [row] for key, row in expected.pack(indices).items()},
            )


@pytest.mark.parametrize(
    "columns,multipack_attn",
    [
        ({"token_type_ids": lambda ids: [0] * len(ids)}, False),
        ({"labels": lambda ids: ids[:-1]}, False),
        ({"position_ids": lambda ids: list(range(len(ids)))}, False),
        ({"length": len}, False),
        # no attention mask for the datasets transforms to drop
        ({}, True),
    ],
)
def test_flat_rows_defer_to_dataset_rows_when_they_cannot_match_them(
    tokenizer_huggyllama, columns, multipack_attn
):
    input_ids = [[5, 6, 7], [8, 9]]
    data = {"input_ids": input_ids}
    for column, make_row in columns.items():
        data[column] = [make_row(ids) for ids in input_ids]
    collate = PretrainingBatchSamplerDataCollatorForSeq2Seq(
        tokenizer_huggyllama,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=8,
        multipack_attn=multipack_attn,
    )

    assert (
        streaming._flat_rows(Dataset.from_dict(data), collate, 8, multipack_attn)
        is None
    )


@pytest.mark.parametrize("dataset_type", ["pretrain", "chat_template"])
@pytest.mark.parametrize("multipack_attn", [True, False])
@pytest.mark.parametrize("balance_labels", [False, True])
def test_encode_packed_streaming_fast_packing_matches_the_collator(
    tokenizer_huggyllama, monkeypatch, dataset_type, multipack_attn, balance_labels
):
    if balance_labels:
        tokenizer_huggyllama.padding_side = "right"
    rand = random.Random(0)

    def text(low, high):
        return " ".join(
            "".join(rand.choices(string.ascii_lowercase, k=rand.randint(2, 8)))
            for _ in range(rand.randint(low, high))
        )

    if dataset_type == "pretrain":
        examples = {"text": [text(10, 200) for _ in range(120)]}
        ds_cfg = DictDefault({"path": "unused", "type": "pretrain"})
        cfg = DictDefault({"pretraining_dataset": [ds_cfg], "sequence_len": 256})
    else:
        examples = {
            "messages": [
                [
                    {"role": "user", "content": text(2, 60)},
                    {"role": "assistant", "content": text(2, 80)},
                ]
                for _ in range(120)
            ]
        }
        ds_cfg = DictDefault({"path": "unused", "type": "chat_template"})
        cfg = DictDefault(
            {"datasets": [ds_cfg], "chat_template": "chatml", "sequence_len": 256}
        )
    ds_wrapper = functools.partial(
        get_dataset_wrapper,
        dataset_config=ds_cfg,
        tokenizer=tokenizer_huggyllama,
        cfg=cfg,
        dataset_base_type=dataset_type,
    )
    collate = PretrainingBatchSamplerDataCollatorForSeq2Seq(
        tokenizer_huggyllama,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=256,
        multipack_attn=multipack_attn,
    )

    def encode():
        # RandomSampler draws from the global generator
        torch.manual_seed(0)
        return streaming.encode_packed_streaming(
            collate,
            ds_wrapper,
            examples,
            bin_size=200,
            max_seq_length=256,
            batch_size=2,
            multipack_attn=multipack_attn,
            balance_labels=balance_labels,
        )

    flat_rows = streaming._flat_rows
    built = []
    monkeypatch.setattr(
        streaming,
        "_flat_rows",
        lambda *args: built.append(flat_rows(*args)) or built[-1],
    )
    packed = encode()
    assert built and built[0] is not None
    monkeypatch.setattr(streaming, "_flat_rows", lambda *args: None)
    collated = encode()

    assert len(collated["input_ids"]) > 1
    _assert_same_packed_rows(packed, collated)
