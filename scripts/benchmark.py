import os
import tempfile
from axolotl.train import train
from axolotl.cli import load_datasets
from axolotl.utils.dict import DictDefault
from axolotl.common.cli import TrainerCliArgs
from axolotl.utils.config import normalize_config

os.environ["WANDB_DISABLED"] = "true"

with tempfile.TemporaryDirectory() as output_dir:
    cfg = DictDefault(
    {
        "base_model": "PY007/TinyLlama-1.1B-intermediate-step-480k-1T",
        "base_model_config": "PY007/TinyLlama-1.1B-intermediate-step-480k-1T",
        "sequence_len": 2048,
        "sample_packing": True,
        "special_tokens": {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            },
        ],
        "num_epochs": 1,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "output_dir": output_dir,
        "bf16": True,
        "learning_rate": 0.00001,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "max_steps": 100,
        "val_set_size": 0.001,
        "flash_attention": True,
        "flash_attn_cross_entropy": False,
        "flash_attn_rms_norm": False
    })

    normalize_config(cfg)
    cli_args = TrainerCliArgs()
    dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
