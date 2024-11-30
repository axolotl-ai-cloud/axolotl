"""
E2E tests for lora llama
"""

import logging
import os
from importlib import reload
from pathlib import Path

import pytest
from tbparse import SummaryReader
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from ..utils import most_recent_subdir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


@pytest.fixture(autouse=True)
def reload_transformers():
    import transformers.models.llama.modeling_llama

    yield
    reload(transformers.models.llama.modeling_llama)


class TestFAXentropyLlama:
    """
    Test case for Llama models using LoRA w multipack
    """

    @pytest.mark.parametrize(
        "gradient_accumulation_steps",
        [1, 4],
    )
    def test_lora_packing_fa_cross_entropy(self, temp_dir, gradient_accumulation_steps):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": True,
                "flash_attention": True,
                "flash_attn_cross_entropy": True,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.2,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "mlabonne/FineTome-100k",
                        "field_messages": "conversations",
                        "message_field_content": "value",
                        "message_field_role": "from",
                        "type": "chat_template",
                        "split": "train[:2%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "save_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "use_tensorboard": True,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.bin").exists()

        tb_log_path = most_recent_subdir(temp_dir + "/runs")
        event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
        reader = SummaryReader(event_file)
        df = reader.scalars  # pylint: disable=invalid-name
        df = df[(df.tag == "train/train_loss")]  # pylint: disable=invalid-name
        assert df.value.values[-1] < 1.5, "Loss is too high"
