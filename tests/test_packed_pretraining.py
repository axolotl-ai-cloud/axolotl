"""Module for testing streaming dataset sequence packing"""

import functools
import random
import string

import pytest
import torch
from datasets import IterableDataset
from torch.utils.data import DataLoader

from axolotl.utils.data import get_dataset_wrapper, wrap_pretraining_dataset
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

        # pylint: disable=duplicate-code
        original_bsz = cfg.micro_batch_size
        train_dataset = wrap_pretraining_dataset(
            dataset,
            tokenizer_huggyllama,
            cfg,
            ds_wrapper_partial,
            max_tokens=cfg.sequence_len,
            batch_size=cfg.micro_batch_size,
            seed=cfg.seed or 42,
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
