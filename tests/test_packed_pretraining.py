"""Module for testing streaming dataset sequence packing"""

import functools

import pytest
import torch
from torch.utils.data import DataLoader

from axolotl.utils.data import get_dataset_wrapper, wrap_pretraining_dataset
from axolotl.utils.dict import DictDefault


class TestPretrainingPacking:
    """
    Test class for packing streaming dataset sequences
    """

    @pytest.mark.flaky(retries=1, delay=5)
    def test_packing_stream_dataset(
        self, tokenizer_huggyllama, dataset_tiny_shakespeare_streaming
    ):
        dataset = dataset_tiny_shakespeare_streaming

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
            assert "attention_mask" not in data
            # FIXME add back once we fix packing unpad/pad with attention mask
            # assert data["attention_mask"].shape == torch.Size(
            #     [1, original_bsz * cfg.sequence_len]
            # )
            idx += 1
