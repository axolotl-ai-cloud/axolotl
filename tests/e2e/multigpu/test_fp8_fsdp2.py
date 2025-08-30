"""Test module for FP8 mixed precision with FSDP2 multi-GPU functionality."""

import os
from pathlib import Path

import torch
import yaml
from accelerate.test_utils import execute_subprocess_async
from tbparse import SummaryReader
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import most_recent_subdir, require_hopper, require_torch_2_7_0

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


def verify_fp8_training_success(temp_dir):
    """Verify that FP8 training completed successfully by checking artifacts and loss."""
    output_path = Path(temp_dir)

    model_files = list(output_path.glob("*.bin")) + list(
        output_path.glob("*.safetensors")
    )
    assert len(model_files) > 0, "No model files found - training may have failed"

    checkpoint_files = list(output_path.glob("checkpoint-*"))
    assert len(checkpoint_files) > 0, (
        "No checkpoint files found - training may have failed"
    )

    tb_log_path = most_recent_subdir(temp_dir + "/runs")
    if tb_log_path:
        event_files = sorted(os.listdir(tb_log_path))
        if event_files:
            event_file = os.path.join(tb_log_path, event_files[0])
            reader = SummaryReader(event_file)
            df = reader.scalars
            train_loss_df = df[df.tag == "train/train_loss"]
            if len(train_loss_df) > 0:
                final_loss = train_loss_df.value.values[-1]
                assert not torch.isnan(torch.tensor(final_loss)), (
                    f"Training loss is NaN: {final_loss}"
                )


class TestFP8FSDP2:
    """Test class for FP8 mixed precision with FSDP2 functionality."""

    @require_torch_2_7_0
    @require_hopper
    def test_fp8_fsdp2_smoke(self, temp_dir):
        """Smoke test for 2-GPU FP8 + torch.compile + FSDP2 training"""
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "trust_remote_code": True,
                "sequence_len": 512,
                "val_set_size": 0.05,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 3,  # Very short smoke test
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",  # Use standard optimizer for stability
                "lr_scheduler": "cosine",
                "sdp_attention": True,
                "pad_to_seq_len": True,
                "sample_packing": True,
                # FP8 configuration
                "fp8": True,
                "fp8_enable_fsdp_float8_all_gather": True,
                "torch_compile": True,
                # FSDP2 configuration
                "fsdp_version": 2,
                "fsdp_config": {
                    "offload_params": False,
                    "cpu_ram_efficient_loading": False,
                    "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                    "state_dict_type": "FULL_STATE_DICT",
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "reshard_after_forward": True,
                },
                "use_tensorboard": True,
                "save_safetensors": True,
                "save_first_step": False,
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

        verify_fp8_training_success(temp_dir)
