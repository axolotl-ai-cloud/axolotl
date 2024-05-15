"""Module for testing streaming dataset sequence packing"""
import functools
import unittest

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from axolotl.utils.data import get_dataset_wrapper, wrap_pretraining_dataset
from axolotl.utils.dict import DictDefault


class TestPretrainingPacking(unittest.TestCase):
    """
    Test class for packing streaming dataset sequences
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.pad_token = "</s>"

    def test_packing_stream_dataset(self):
        # pylint: disable=duplicate-code
        dataset = load_dataset(
            "c4",
            "en",
            streaming=True,
        )["train"]

        cfg = DictDefault(
            {
                "pretraining_dataset": [
                    {
                        "path": "c4",
                        "name": "en",
                        "type": "pretrain",
                    }
                ],
                "sample_packing": True,
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
            self.tokenizer,
            cfg,
            cfg.pretraining_dataset[0]["type"] or "pretrain",
        )

        original_bsz = cfg.micro_batch_size
        train_dataset = wrap_pretraining_dataset(
            dataset,
            self.tokenizer,
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
            if idx > 10:
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
            assert data["attention_mask"].shape == torch.Size(
                [1, original_bsz * cfg.sequence_len]
            )
            idx += 1


if __name__ == "__main__":
    unittest.main()
