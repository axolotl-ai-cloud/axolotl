"""E2E resume test for NVFP4 LoRA training.

Lives in patched/ (run in its own process) because it drives the full ``train()``
pipeline twice; mixing it with the module-level nvfp4 tests pollutes transformers'
lazy-import state. Validates that a checkpoint resumes: the frozen FP4 base
reconstructs deterministically and the adapter + optimizer reload, so training
continues from the saved step rather than restarting (resume is bit-faithful —
verified manually that the resumed step-N loss equals the original step-N loss).

Beyond ``global_step``, both variants assert the **first post-resume training
loss is sane**. The "H1" failure mode (a save/reload state-dict corruption of the
FP4 embedding / tied lm_head store) is silent in ``global_step`` but produces a
dead forward whose loss instantly jumps to ~ln(vocab) (uniform logits). The
plain compute-base variant does not create the tied FP4 store, so the
``quantize_lm_head`` + ``fused_fp4_cross_entropy`` variant below is what actually
exercises the at-risk path; the loss-sanity guard catches the dead forward there.
"""

import json
import math
import os

from ..utils import require_torch_2_8_0, requires_sm_ge_100


def _post_resume_losses(checkpoint_dir):
    """First few post-resume training losses from a resumed run's trainer state.

    The resume runs into a fresh ``output_dir`` so ``log_history`` starts at the
    first post-resume step — its leading ``loss`` entries are exactly the steps
    that a dead-forward corruption would blow up.
    """
    state = json.load(open(os.path.join(checkpoint_dir, "trainer_state.json")))
    return [e["loss"] for e in state["log_history"] if "loss" in e]


@require_torch_2_8_0
@requires_sm_ge_100
class TestNVFP4Resume:
    def _base_cfg(self, temp_dir, extra):
        from axolotl.utils.config import normalize_config, validate_config
        from axolotl.utils.dict import DictDefault

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
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-4,
                "logging_steps": 1,
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

    def _vocab_ln(self, cfg):
        """ln(effective vocab) — the dead-uniform-logits loss signature."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        return math.log(len(tok))

    def test_nvfp4_lora_resume(self, temp_dir):
        from axolotl.common.datasets import load_datasets
        from axolotl.train import train

        cfg = self._base_cfg(temp_dir, {})
        train(cfg=cfg, dataset_meta=load_datasets(cfg=cfg))
        ckpt = os.path.join(temp_dir, "checkpoint-4")
        assert os.path.isdir(ckpt)
        assert os.path.isfile(os.path.join(ckpt, "adapter_model.safetensors"))

        pre_losses = _post_resume_losses(ckpt)
        assert pre_losses, "no pre-resume loss logged"

        # Resume into a fresh dir so log_history starts at the first post-resume
        # step (clean read of the loss a dead-forward would corrupt).
        resume_dir = os.path.join(temp_dir, "resume")
        rcfg = self._base_cfg(resume_dir, {"resume_from_checkpoint": ckpt})
        train(cfg=rcfg, dataset_meta=load_datasets(cfg=rcfg))
        resume_ckpt = os.path.join(resume_dir, "checkpoint-8")
        state = json.load(open(os.path.join(resume_ckpt, "trainer_state.json")))
        assert state["global_step"] == 8

        self._assert_resume_loss_sane(rcfg, pre_losses, resume_dir)

    def test_nvfp4_lora_resume_quantized_lm_head(self, temp_dir):
        """Guard the H1 at-risk path: tied embedding + FP4 lm_head store.

        Uses a TIED model (``SmolLM2-135M``, ``tie_word_embeddings=True``) so the
        ``quantize_lm_head`` + ``fused_fp4_cross_entropy`` swap takes the tied
        branch and builds the shared ``NVFP4Embedding`` / ``NVFP4TiedLMHead``
        store — the exact save/reload path the H1 corruption hypothesis lives in
        (an untied model swaps a bare lm_head instead and never creates the tied
        store). The loss-sanity assertion below is the guard: a dead forward on
        resume reads back as loss ~ln(vocab) while global_step stays correct.
        """
        from axolotl.common.datasets import load_datasets
        from axolotl.train import train

        extra = {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "nvfp4_training": {
                "enabled": True,
                "backend": "native",
                "base_mode": "compute",
                "quantize_lm_head": True,
                "fused_fp4_cross_entropy": True,
            },
        }
        cfg = self._base_cfg(temp_dir, extra)
        train(cfg=cfg, dataset_meta=load_datasets(cfg=cfg))
        ckpt = os.path.join(temp_dir, "checkpoint-4")
        assert os.path.isdir(ckpt)
        assert os.path.isfile(os.path.join(ckpt, "adapter_model.safetensors"))

        pre_losses = _post_resume_losses(ckpt)
        assert pre_losses, "no pre-resume loss logged"

        resume_dir = os.path.join(temp_dir, "resume")
        rcfg = self._base_cfg(resume_dir, {**extra, "resume_from_checkpoint": ckpt})
        train(cfg=rcfg, dataset_meta=load_datasets(cfg=rcfg))
        resume_ckpt = os.path.join(resume_dir, "checkpoint-8")
        state = json.load(open(os.path.join(resume_ckpt, "trainer_state.json")))
        assert state["global_step"] == 8

        self._assert_resume_loss_sane(rcfg, pre_losses, resume_dir)

    def _assert_resume_loss_sane(self, rcfg, pre_losses, resume_dir):
        """First post-resume loss must look like training, not a dead forward.

        Catches the ~ln(vocab) dead-uniform-logits signature (H1) while
        tolerating normal step-to-step training noise:
          * ``< 0.5 * ln(vocab)`` — far below the ~12.4 dead-forward plateau but
            well above any healthy loss for this tiny model.
          * ``< 5.0`` — absolute backstop independent of vocab size.
          * ``< 3 * pre_resume_loss`` — regression guard tying the resumed step
            to the pre-interruption scale (3x slack absorbs the step's own noise
            and the LR/optimizer-state differences across the boundary).
        """
        resume_ckpt = os.path.join(resume_dir, "checkpoint-8")
        post_losses = _post_resume_losses(resume_ckpt)
        assert post_losses, "no post-resume loss logged"
        first_post = post_losses[0]
        ref = min(pre_losses)
        vocab_ln = self._vocab_ln(rcfg)

        assert first_post < 0.5 * vocab_ln, (
            f"post-resume loss {first_post:.4f} is near ln(vocab)={vocab_ln:.4f} "
            f"(dead/uniform logits — H1 checkpoint-resume corruption)"
        )
        assert first_post < 5.0, (
            f"post-resume loss {first_post:.4f} >= 5.0 — forward looks dead, "
            f"not training (pre-resume losses {pre_losses})"
        )
        assert first_post < 3 * ref, (
            f"post-resume loss {first_post:.4f} is >3x the pre-resume loss "
            f"{ref:.4f} — resume did not continue cleanly from the checkpoint"
        )
