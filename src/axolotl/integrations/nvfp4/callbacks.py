"""NVFP4 TrainerCallbacks — resume integrity guard + FP4-packed save sidecar."""

import os

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class NVFP4ResumeIntegrityCallback(TrainerCallback):
    """Fail loud if a resumed checkpoint loaded non-finite trainable weights (HF's non-strict load otherwise silently trains a dead forward)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **_kwargs,
    ) -> TrainerControl:
        if not self.cfg.resume_from_checkpoint or model is None:
            return control
        bad = []
        for name, param in model.named_parameters():
            if not param.requires_grad or param.is_meta:
                continue
            if not torch.isfinite(param).all():
                bad.append(name)
                if len(bad) >= 5:
                    break
        if bad:
            raise RuntimeError(
                f"NVFP4 resume integrity: trainable weights contain NaN/Inf after "
                f"loading checkpoint '{self.cfg.resume_from_checkpoint}' (e.g. {bad}). "
                "The checkpoint is likely corrupt — a common cause is an auto-save "
                "that captured the model after a GPU Xid NaN'd training. Resume from "
                "an earlier good checkpoint."
            )
        return control


class NVFP4SaveCallback(TrainerCallback):
    """Write the FP4-packed sidecar (nvfp4_packed.pt) alongside each checkpoint."""

    def __init__(self, cfg, trainer):
        self.cfg = cfg
        self.trainer = trainer

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **_kwargs,
    ) -> TrainerControl:
        from .nvfp4_training import save_nvfp4_packed

        if not args.should_save:  # runs on every rank; only the saving process writes
            return control
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model = self.trainer.accelerator.unwrap_model(
            self.trainer.model, keep_torch_compile=False
        )
        if os.path.isdir(ckpt):
            save_nvfp4_packed(model, ckpt)
        return control
