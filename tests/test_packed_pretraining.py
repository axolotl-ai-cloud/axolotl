"""Module for testing streaming dataset sequence packing"""
import math
import unittest
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.data import encode_packed_pretraining


class TestPacking(unittest.TestCase):
    """
    Test class for packing streaming dataset sequences
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "[PAD]",
            }
        )
        self.max_seq_length = 8192
        self.batch_size = 6
        self.sample_packing_efficiency = 1
        self.data_collator_kwargs = {
            "padding": True,
            "pad_to_multiple_of": 64 * math.ceil(self.max_seq_length / 64),
        }

    def test_packing_stream_dataset(self):
        # pylint: disable=duplicate-code
        dataset = load_dataset(
            "c4",
            "en",
            streaming=True,
        )["train"]

        encode = partial(
            encode_packed_pretraining,
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            sample_packing_efficiency=self.sample_packing_efficiency,
        )

        dataset = dataset.map(
            encode,
            batched=True,
            input_columns="text",
            remove_columns=dataset.features.keys(),
        )

        data_collator_fn = DataCollatorForSeq2Seq(
            self.tokenizer,
            return_tensors="pt",
            **self.data_collator_kwargs,
        )

        trainer_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=data_collator_fn,
            drop_last=True,
        )
        idx = 0
        for data in trainer_loader:
            if idx > 10:
                break
            assert data["input_ids"].shape == (self.batch_size, self.max_seq_length)
            assert data["position_ids"].shape == (self.batch_size, self.max_seq_length)
            assert data["labels"].shape == (self.batch_size, self.max_seq_length)
            assert data["attention_mask"].shape == (
                self.batch_size,
                self.max_seq_length,
            )
            idx += 1


if __name__ == "__main__":
    unittest.main()
