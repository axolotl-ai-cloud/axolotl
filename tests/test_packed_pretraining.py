"""Module for testing streaming dataset sequence packing"""

import functools
import json
import random
import string

import multiprocess.pool
import pytest
import torch
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader

from axolotl.utils.data import get_dataset_wrapper, sft, wrap_streaming_dataset
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
