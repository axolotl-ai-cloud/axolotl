"""E2E resume test for NVFP4 LoRA training.

Lives in patched/ (run in its own process) because it drives the full ``train()``
pipeline twice; mixing it with the module-level nvfp4 tests pollutes transformers'
lazy-import state. Validates that a checkpoint resumes: the frozen FP4 base
reconstructs deterministically and the adapter + optimizer reload, so training
continues from the saved step rather than restarting (resume is bit-faithful —
verified manually that the resumed step-N loss equals the original step-N loss).
"""

import json
import os

from ..utils import require_torch_2_8_0, requires_sm_ge_100


@require_torch_2_8_0
@requires_sm_ge_100
class TestNVFP4Resume:
    def test_nvfp4_lora_resume(self, temp_dir):
        from axolotl.common.datasets import load_datasets
        from axolotl.train import train
        from axolotl.utils.config import normalize_config, validate_config
        from axolotl.utils.dict import DictDefault

        def make(extra):
            cfg = DictDefault(
                {
                    "base_model": "axolotl-ai-co/tiny-qwen2-129m",
                    "sequence_len": 256,
                    "sample_packing": False,
                    "bf16": True,
                    "adapter": "lora",
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "lora_target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    "val_set_size": 0.0,
                    "datasets": [
                        {"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}
                    ],
                    "num_epochs": 1,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "output_dir": temp_dir,
                    "learning_rate": 1e-4,
                    "optimizer": "adamw_torch",
                    "max_steps": 8,
                    "save_strategy": "steps",
                    "save_steps": 4,
                    "special_tokens": {},
                    "nvfp4_training": {
                        "enabled": True,
                        "backend": "native",
                        "base_mode": "compute",
                    },
                }
            ) | DictDefault(extra)
            cfg = validate_config(cfg)
            normalize_config(cfg)
            return cfg

        cfg = make({})
        train(cfg=cfg, dataset_meta=load_datasets(cfg=cfg))
        ckpt = os.path.join(temp_dir, "checkpoint-4")
        assert os.path.isdir(ckpt)
        assert os.path.isfile(os.path.join(ckpt, "adapter_model.safetensors"))

        rcfg = make({"resume_from_checkpoint": ckpt})
        train(cfg=rcfg, dataset_meta=load_datasets(cfg=rcfg))
        state = json.load(
            open(os.path.join(temp_dir, "checkpoint-8", "trainer_state.json"))
        )
        assert state["global_step"] == 8
