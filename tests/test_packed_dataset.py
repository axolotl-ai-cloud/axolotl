"""Module for testing dataset sequence packing"""

import unittest
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from axolotl.datasets import ConstantLengthDataset, TokenizedPromptDataset
from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter


class TestPacking(unittest.TestCase):
    """
    Test class for packing dataset sequences
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        )

    def test_increments_attention(self):
        prompter = AlpacaPrompter("chat")
        strat = AlpacaPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
            False,
            2048,
        )
        dateset = load_dataset(
            "json",
            data_files=str(Path(__file__).parent / "fixtures/alpaca/alpaca.json"),
        )["train"]
        dataset = Dataset.from_list(list(TokenizedPromptDataset(strat, dateset)))

        constant_len_dataset = ConstantLengthDataset(
            self.tokenizer,
            [dataset],
            seq_length=2048,
        )
        packed_dataset = Dataset.from_list(list(constant_len_dataset))
        example = packed_dataset[0]
        next_bos_index = (
            example["input_ids"][1:].index(self.tokenizer.bos_token_id) + 1
        )  # add one since we sliced

        # first example doesn't have mask reset
        assert example["input_ids"][0] == self.tokenizer.bos_token_id
        assert example["attention_mask"][0] == 1
        assert example["position_ids"][0] == 0
        assert example["position_ids"][1] == 1

        # but subsequent one does
        assert example["input_ids"][next_bos_index] == self.tokenizer.bos_token_id
        assert example["attention_mask"][next_bos_index] == 2
        assert example["position_ids"][next_bos_index] == 0
        assert example["position_ids"][next_bos_index + 1] == 1


if __name__ == "__main__":
    unittest.main()
