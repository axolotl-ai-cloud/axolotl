"""Module for testing streaming dataset sequence packing"""
import unittest
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.data import encode_packed_pretraining


class TestPretrainingPacking(unittest.TestCase):
    """
    Test class for packing streaming dataset sequences
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.pad_token = "</s>"
        self.max_seq_length = 2048
        self.batch_size = 2

    def test_packing_stream_dataset(self):
        # pylint: disable=duplicate-code
        dataset = load_dataset(
            "c4",
            "en",
            streaming=True,
        )["train"]

        collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
            self.tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.max_seq_length,
        )

        encode = partial(
            encode_packed_pretraining,
            self.tokenizer,
            collate_fn,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
        )

        dataset = dataset.map(
            encode,
            batched=True,
            input_columns="text",
            remove_columns=dataset.features.keys(),
        )

        trainer_loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=None,
            drop_last=True,
        )
        idx = 0
        for data in trainer_loader:
            if idx > 10:
                break
            assert data["input_ids"].shape == torch.Size(
                [1, self.batch_size * self.max_seq_length]
            )
            assert data["position_ids"].shape == torch.Size(
                [1, self.batch_size * self.max_seq_length]
            )
            assert data["labels"].shape == torch.Size(
                [1, self.batch_size * self.max_seq_length]
            )
            assert data["attention_mask"].shape == torch.Size(
                [1, self.batch_size * self.max_seq_length]
            )
            idx += 1


if __name__ == "__main__":
    unittest.main()
