"""Module for testing dataset sequence packing"""

import unittest

from transformers import AutoTokenizer

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import setup_model_and_trainer
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import with_temp_dir
from tests.hf_offline_utils import enable_hf_offline


class TestPacking(unittest.TestCase):
    """
    Test class for packing dataset sequences
    """

    @enable_hf_offline
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        )

    @with_temp_dir
    def test_lora_packing(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "sample_packing": True,
                "multipack_real_batches": False,
                "eval_sample_packing": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.2,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "dataset_num_proc": 4,
                "num_epochs": 1,
                "max_steps": 20,
                "save_steps": 10,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "fp16": False,
                "bf16": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        (
            trainer,
            _,
            _,
            _,
            _,
        ) = setup_model_and_trainer(cfg, dataset_meta)

        sampler = trainer._get_eval_sampler(trainer.eval_dataset)
        assert "MultipackBatchSampler" in sampler.__class__.__name__
        assert (
            "V2BatchSamplerDataCollatorForSeq2Seq"
            in trainer.eval_data_collator.__class__.__name__
        )
        dataloader = trainer.get_eval_dataloader(trainer.eval_dataset)
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
        assert batch["input_ids"].shape == (1, 8192)

        sampler = trainer._get_train_sampler(trainer.train_dataset)
        assert "MultipackBatchSampler" in sampler.__class__.__name__
        assert (
            "V2BatchSamplerDataCollatorForSeq2Seq"
            in trainer.train_data_collator.__class__.__name__
        )
        dataloader = trainer.get_train_dataloader()
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
        assert batch["input_ids"].shape == (1, 8192)


if __name__ == "__main__":
    unittest.main()
