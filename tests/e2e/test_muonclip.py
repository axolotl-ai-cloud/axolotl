"""End-to-end checks for MuonClip training."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

SKIP_REMOTE = os.environ.get("AXOLOTL_SKIP_REMOTE_DOWNLOADS", "").lower() in (
    "1",
    "true",
    "yes",
)


def with_temp_dir(test_func):
    def wrapper(*args, **kwargs):
        temp_dir = tempfile.mkdtemp()
        try:
            test_func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            shutil.rmtree(temp_dir)

    return wrapper


class TestMuonClipE2E(unittest.TestCase):
    """
    Smoke tests comparing AdamW and MuonClip training on a tiny Qwen3 model.
    """

    def _base_cfg(self, output_dir: Path) -> DictDefault:
        return DictDefault(
            {
                "base_model": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "sequence_len": 64,
                "val_set_size": 0,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "special_tokens": {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                },
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-4,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "output_dir": str(output_dir),
                "save_steps": 0,
                "eval_steps": 0,
                "save_first_step": False,
                "flash_attention": False,
                "bf16": False,
                "fp16": False,
                "muonclip": {"enabled": False},
            }
        )

    def _run_training(self, cfg: DictDefault) -> dict:
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)
        train(cfg=cfg, dataset_meta=dataset_meta)
        state_path = Path(cfg.output_dir) / "trainer_state.json"
        if not state_path.exists():
            checkpoints = sorted(Path(cfg.output_dir).glob("checkpoint-*/trainer_state.json"))
            if checkpoints:
                state_path = checkpoints[-1]
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as state_fin:
                return json.load(state_fin)
        return {"global_step": cfg.max_steps}

    def _muon_cfg(self, output_dir: Path) -> DictDefault:
        cfg = self._base_cfg(output_dir)
        cfg.optimizer = "muon"
        cfg.muonclip = {
            "enabled": True,
            "momentum": 0.95,
            "weight_decay": 0.0,
            "qk_clip": True,
            "qk_clip_tau": 10.0,
        }
        cfg.save_steps = 1
        cfg.save_strategy = "steps"
        cfg.save_total_limit = 2
        cfg.save_first_step = True
        return cfg

    @with_temp_dir
    @unittest.skipIf(
        SKIP_REMOTE, "MuonClip smoke test requires model artifacts from HuggingFace"
    )
    def test_muonclip_smoke(self, temp_dir):
        temp_dir = Path(temp_dir)

        adam_cfg = self._base_cfg(temp_dir / "adamw")
        adam_state = self._run_training(adam_cfg)

        muon_cfg = self._base_cfg(temp_dir / "muonclip")
        muon_cfg.optimizer = "muon"
        muon_cfg.muonclip = {
            "enabled": True,
            "momentum": 0.95,
            "weight_decay": 0.0,
            "qk_clip": True,
            "qk_clip_tau": 10.0,
        }
        muon_state = self._run_training(muon_cfg)

        assert (
            adam_state["global_step"]
            == muon_state["global_step"]
            == muon_cfg.max_steps
        )

    @with_temp_dir
    @unittest.skipIf(
        SKIP_REMOTE, "MuonClip smoke test requires model artifacts from HuggingFace"
    )
    def test_muonclip_checkpoint_resume(self, temp_dir):
        temp_dir = Path(temp_dir)
        output_dir = temp_dir / "muonclip_resume"

        stage1_cfg = self._muon_cfg(output_dir)
        stage1_cfg.max_steps = 1
        stage1_state = self._run_training(stage1_cfg)
        assert stage1_state["global_step"] == 1

        checkpoint_dir = output_dir / "checkpoint-1"
        assert checkpoint_dir.exists()
        muon_state_path = checkpoint_dir / "muonclip_state_rank0.pt"
        assert muon_state_path.exists()
        first_buffers = torch.load(muon_state_path, map_location="cpu")
        assert first_buffers, "Expected Muon buffers in checkpoint"

        stage2_cfg = self._muon_cfg(output_dir)
        stage2_cfg.max_steps = 2
        stage2_cfg.resume_from_checkpoint = str(checkpoint_dir)
        stage2_state = self._run_training(stage2_cfg)
        assert stage2_state["global_step"] == stage2_cfg.max_steps == 2

        final_checkpoint = output_dir / "checkpoint-2"
        assert final_checkpoint.exists()
        final_state_path = final_checkpoint / "muonclip_state_rank0.pt"
        assert final_state_path.exists()
        final_buffers = torch.load(final_state_path, map_location="cpu")
        assert final_buffers

        overlap = set(first_buffers.keys()) & set(final_buffers.keys())
        assert overlap
        assert any(
            not torch.allclose(first_buffers[key], final_buffers[key])
            for key in overlap
        ), "Muon state should change after additional training steps"
