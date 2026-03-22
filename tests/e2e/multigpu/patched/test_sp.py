"""E2E tests for sequence parallelism"""

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from ...utils import check_tensorboard


class TestSequenceParallelism:
    """Test case for training with sequence parallelism enabled"""

    def _run_sequence_parallel_test(
        self,
        temp_dir,
        sample_packing=True,
        micro_batch_size=1,
        pad_to_sequence_len=True,
        ring_attn_func=None,
        threshold=2.0,
    ):
        """Helper method to run sequence parallel tests with different configurations"""
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "load_in_8bit": False,
                "load_in_4bit": True,
                "strict": False,
                "sequence_len": 2048,
                "adapter": "qlora",
                "sample_packing": sample_packing,
                "eval_sample_packing": sample_packing,
                "pad_to_sequence_len": pad_to_sequence_len,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_modules_to_save": ["embed_tokens", "lm_head"],
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 8,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "dataset_prepared_path": temp_dir + "/last_run_prepared",
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "loss_watchdog_threshold": 5.0,
                "loss_watchdog_patience": 3,
                "bf16": "auto",
                "warmup_steps": 1,
                "saves_per_epoch": 1,
                "logging_steps": 1,
                "weight_decay": 0.0,
                "use_tensorboard": True,
                "context_parallel_size": 2,
                "ring_attn_func": ring_attn_func,
                "save_first_step": False,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                "2",
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            threshold,
            "Train Loss (%s) is too high",
        )

    @pytest.mark.parametrize(
        "sample_packing, micro_batch_size, pad_to_sequence_len, ring_attn_func, threshold",
        [
            (True, 1, True, None, 2.5),  # defaults to varlen_llama3 ring_attn_func
            (False, 2, True, None, 2.5),  # defaults to batch_ring ring_attn_func
            # (False, 2, True, "batch_zigzag", 2.5),
            # (False, 2, False, None, 2.65),  # defaults to batch_ring ring_attn_func
        ],
        ids=[
            "sample_packing, varlen_llama3 ring_attn_func",
            "no sample_packing, pad_to_sequence_len, batch_ring ring_attn_func",
            # "no sample_packing, no pad_to_sequence_len, batch_zigzag ring_attn_func",
            # "no sample_packing, no pad_to_sequence_len, batch_ring ring_attn_func",
        ],
    )
    def test_sequence_parallel_training(
        self,
        temp_dir,
        sample_packing,
        micro_batch_size,
        pad_to_sequence_len,
        ring_attn_func,
        threshold,
    ):
        """Test sequence parallel training with different configurations"""
        self._run_sequence_parallel_test(
            temp_dir,
            sample_packing=sample_packing,
            micro_batch_size=micro_batch_size,
            pad_to_sequence_len=pad_to_sequence_len,
            ring_attn_func=ring_attn_func,
            threshold=threshold,
        )
