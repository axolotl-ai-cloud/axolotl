"""
E2E tests for activation checkpointing
"""

import pytest
import transformers
from torch.utils.checkpoint import checkpoint

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists


@pytest.fixture()
def fix_checkpoint_after_test():
    yield
    transformers.modeling_utils.checkpoint = checkpoint


class TestActivationCheckpointing:
    """
    E2E tests for activation checkpointing
    """

    @pytest.mark.parametrize(
        "gradient_checkpointing",
        ["offload", "offload_disk"],
    )
    def test_activation_checkpointing_offload(
        self,
        temp_dir,
        fix_checkpoint_after_test,
        gradient_checkpointing,
    ):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                    "eos_token": "<|im_end|>",
                },
                "datasets": [
                    {
                        "chat_template": "chatml",
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "split": "train[:10%]",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": True,
                "save_safetensors": True,
                "gradient_checkpointing": gradient_checkpointing,
                "save_first_step": False,
                "dataset_num_proc": 4,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
