"""Module for testing streaming dataset sequence packing"""

import pytest
from datasets import concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_strategies.completion import load
from axolotl.utils.collators import V2BatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.data.utils import drop_long_seq_in_dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

from tests.hf_offline_utils import enable_hf_offline


@pytest.fixture(name="tokenizer")
def fixture_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = "</s>"
    return tokenizer


class TestBatchedSamplerPacking:
    """
    Test class for packing streaming dataset sequences
    """

    @pytest.mark.parametrize(
        "batch_size, num_workers",
        [
            (1, 0),
            (2, 0),
            (1, 2),
            (2, 2),
        ],
    )
    @pytest.mark.parametrize("max_seq_length", [4096, 512])
    @pytest.mark.parametrize("sequential", [True, False])
    @enable_hf_offline
    def test_packing(
        self,
        dataset_winglian_tiny_shakespeare,
        batch_size,
        num_workers,
        tokenizer,
        max_seq_length,
        sequential,
    ):
        import axolotl.monkeypatch.data.batch_dataset_fetcher  # pylint: disable=unused-import  # noqa: F401

        dataset = dataset_winglian_tiny_shakespeare["train"]

        cfg = DictDefault(
            {
                "train_on_inputs": True,
                "sequence_len": max_seq_length,
            }
        )
        ds_cfg = DictDefault(
            {
                "field": "text",
            }
        )
        completion_strategy = load(tokenizer, cfg, ds_cfg)
        dataset_wrapper = TokenizedPromptDataset(
            completion_strategy,
            dataset,
        )
        train_dataset = concatenate_datasets([dataset_wrapper])

        train_dataset = drop_long_seq_in_dataset(train_dataset, cfg)

        lengths = get_dataset_lengths(train_dataset)
        batch_sampler = MultipackBatchSampler(
            sampler=RandomSampler(train_dataset),
            lengths=lengths,
            batch_size=batch_size,
            batch_max_len=max_seq_length,
            group_size=100000,
            bin_size=200,
            sequential=sequential,
        )

        loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=V2BatchSamplerDataCollatorForSeq2Seq(  # pylint: disable=unexpected-keyword-arg
                tokenizer=tokenizer,
                padding=True,
                pad_to_multiple_of=max_seq_length,
                return_tensors="pt",
            ),
            num_workers=num_workers,
        )

        batch_idxs = []
        for batch in batch_sampler:
            for pack in batch:
                batch_idxs.extend(pack)

        for batch in loader:
            assert batch["input_ids"].numel() <= batch_size * max_seq_length
            assert batch["input_ids"].shape[1] == max_seq_length

        original_idxs = set(range(len(train_dataset)))
        assert original_idxs == set(batch_idxs)
