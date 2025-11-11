"""E2E Test the preprocess cli"""

from pathlib import Path

import yaml
from accelerate.test_utils import execute_subprocess_async

from axolotl.utils.dict import DictDefault

AXOLOTL_ROOT = Path(__file__).parent.parent.parent


class TestPreprocess:
    """test cases for preprocess"""

    def test_w_deepspeed(self, temp_dir):
        """make sure preprocess doesn't choke when using deepspeed in the config"""

        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
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
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "bf16": "auto",
                "deepspeed": str(AXOLOTL_ROOT / "deepspeed_configs/zero1.json"),
                "dataset_prepared_path": temp_dir + "/last_run_prepared",
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "axolotl",
                "preprocess",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        assert (Path(temp_dir) / "last_run_prepared").exists()
