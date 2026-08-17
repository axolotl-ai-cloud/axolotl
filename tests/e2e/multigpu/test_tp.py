"""multigpu e2e test for tensor parallelism."""

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async, get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard_loss_decreased, require_torch_2_7_0


class TestTensorParallel:
    """Test class for Tensor Parallel functionality."""

    @pytest.mark.skip(
        reason="TP doesn't work with models with tied weights (embeddings)"
    )
    @require_torch_2_7_0
    def test_fft_sft(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch",
                "tensor_parallel_size": 2,
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "use_tensorboard": True,
                "bf16": True,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "train",
                str(Path(temp_dir) / "config.yaml"),
                "--num-processes",
                "2",
                "--main-process-port",
                f"{get_torch_dist_unique_port()}",
            ]
        )

        check_tensorboard_loss_decreased(
            temp_dir + "/runs", max_initial=5.0, max_final=4.7
        )
