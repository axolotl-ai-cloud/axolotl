"""
E2E regression guard: host-device syncs per micro-step on the default packed
flash-attention LoRA path must not creep up as features land.
"""

import json
import os
import unittest

from transformers.utils import is_torch_bf16_gpu_available

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import requires_flash_attn, with_temp_dir

pytestmark = requires_flash_attn

# transformers' mask preprocessing (masking_utils.find_packed_sequence_indices)
# syncs once per forward and is outside Axolotl's control. Every other sync on
# this path has been removed; a higher number here is a regression.
MAX_SYNCS_PER_MICROSTEP = 1


class TestSyncProfiler(unittest.TestCase):
    """Count syncing CUDA ops over a few non-logging micro-steps."""

    @with_temp_dir
    def test_packed_flash_attention_lora_sync_floor(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "flash_attention": True,
                "gradient_checkpointing": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:5%]",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.0001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "constant",
                "max_steps": 5,
                # keep logging out of the profiler window: it syncs by design
                "logging_steps": 10,
                # HF's NaN/inf loss filter is one sync per micro-step; measure our floor without it
                "logging_nan_inf_filter": False,
                "save_first_step": False,
                "save_strategy": "no",
                "sync_profiler_start_step": 2,
                "sync_profiler_steps": 2,
            }
        )
        if is_torch_bf16_gpu_available():
            cfg.bf16 = True
        else:
            cfg.fp16 = True

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)

        with open(os.path.join(temp_dir, "sync_profile.json"), encoding="utf-8") as f:
            profile = json.load(f)

        sites = profile["sites_per_microstep"]
        assert profile["microsteps"] == 4, profile
        assert profile["syncs_per_microstep"]["max"] <= MAX_SYNCS_PER_MICROSTEP, (
            f"syncs per micro-step grew: {profile['syncs_per_microstep']} from {sites}"
        )
        assert not any("modeling_flash_attention_utils" in site for site in sites), (
            f"attention path is deriving varlen metadata per layer again: {sites}"
        )
        assert profile["other_buckets"]["between_steps"] == 0, profile["buckets"]
        assert profile["other_buckets"]["log"] == 0, profile["buckets"]
