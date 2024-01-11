"""
Test module for alpacha integration w chatml
"""
import pytest
from datasets import Dataset

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter
from utils import fixture_tokenizer


@pytest.fixture(name="alpacha_dataset")
def fixture_alpacha_dataset():
    return Dataset.from_list(
        [
            {
                "instruction": "Evaluate this sentence for spelling and grammar mistakes",
                "input": "He finnished his meal and left the resturant",
                "output": "He finished his meal and left the restaurant.",
            }
        ]
    )


class TestAlpacha:
    """
    Test class for alpacha prompter
    """

    def test_no_double_im_end(self, alpacha_dataset, tokenizer):
        strategy = AlpacaPromptTokenizingStrategy(
            AlpacaPrompter(prompt_style="chatml"),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, alpacha_dataset, process_count=1
        )

        input_ids = dataset_wrapper[0]["input_ids"]

        assert input_ids == [
            1,
            32001,
            1587,
            13,
            20548,
            336,
            349,
            396,
            13126,
            369,
            13966,
            264,
            3638,
            28725,
            5881,
            1360,
            395,
            396,
            2787,
            369,
            5312,
            3629,
            2758,
            28723,
            12018,
            264,
            2899,
            369,
            6582,
            1999,
            2691,
            274,
            272,
            2159,
            28723,
            32000,
            28705,
            13,
            32001,
            2188,
            13,
            16627,
            11931,
            456,
            12271,
            354,
            668,
            3572,
            304,
            18756,
            3479,
            17179,
            13,
            2428,
            854,
            28711,
            1497,
            516,
            11314,
            304,
            1749,
            272,
            1846,
            324,
            440,
            32000,
            28705,
            13,
            32001,
            13892,
            13,
            650,
            5967,
            516,
            11314,
            304,
            1749,
            272,
            9926,
            28723,
            32000,
        ]

    def test_no_train_on_input(self, alpacha_dataset, tokenizer):
        strategy = AlpacaPromptTokenizingStrategy(
            AlpacaPrompter(prompt_style="chatml"),
            tokenizer,
            False,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, alpacha_dataset, process_count=1
        )

        labels = dataset_wrapper[0]["labels"]

        assert labels == [
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            650,
            5967,
            516,
            11314,
            304,
            1749,
            272,
            9926,
            28723,
            32000,
        ]

    def test_w_train_on_input(self, alpacha_dataset, tokenizer):
        strategy = AlpacaPromptTokenizingStrategy(
            AlpacaPrompter(prompt_style="chatml"),
            tokenizer,
            True,  # train_on_inputs
            2048,  # sequence_len
        )

        dataset_wrapper = TokenizedPromptDataset(
            strategy, alpacha_dataset, process_count=1
        )

        labels = dataset_wrapper[0]["labels"]

        assert labels == [
            1,
            32001,
            1587,
            13,
            20548,
            336,
            349,
            396,
            13126,
            369,
            13966,
            264,
            3638,
            28725,
            5881,
            1360,
            395,
            396,
            2787,
            369,
            5312,
            3629,
            2758,
            28723,
            12018,
            264,
            2899,
            369,
            6582,
            1999,
            2691,
            274,
            272,
            2159,
            28723,
            32000,
            28705,
            13,
            32001,
            2188,
            13,
            16627,
            11931,
            456,
            12271,
            354,
            668,
            3572,
            304,
            18756,
            3479,
            17179,
            13,
            2428,
            854,
            28711,
            1497,
            516,
            11314,
            304,
            1749,
            272,
            1846,
            324,
            440,
            32000,
            28705,
            13,
            32001,
            13892,
            13,
            650,
            5967,
            516,
            11314,
            304,
            1749,
            272,
            9926,
            28723,
            32000,
        ]
