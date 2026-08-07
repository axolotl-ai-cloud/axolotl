# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
"""Callbacks for Arctic SFT (remote sample generation)."""

from __future__ import annotations

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from axolotl.utils.generation.sft import format_generation_for_logging
from axolotl.utils.logging import get_logger

from .generation import generate_samples_remote

LOG = get_logger(__name__)


class ArcticSFTGenerationCallback(TrainerCallback):
    """``on_evaluate`` sample generation via AP sampling job (not local stub)."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        cfg = getattr(self.trainer, "axolotl_cfg", None)
        if cfg is None or not getattr(cfg, "generate_samples", False):
            return

        client = getattr(self.trainer, "_client", None)
        if client is None:
            # Lazy-create via trainer helper if train() hasn't run yet.
            get_client = getattr(self.trainer, "_get_client", None)
            if get_client is None:
                LOG.warning("Arctic SFT generate_samples: no client; skipping")
                return
            client = get_client()

        acfg = getattr(self.trainer, "_arctic_client_config", None)
        if acfg is None or int(getattr(acfg, "sampling_gpus", 0) or 0) <= 0:
            LOG.warning(
                "Arctic SFT generate_samples requires arctic_sft.sampling_gpus > 0; skipping"
            )
            return

        dataloader = None
        try:
            if getattr(self.trainer, "eval_dataset", None) is not None:
                dataloader = self.trainer.get_eval_dataloader()
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"Could not get eval dataloader: {e}")
        if dataloader is None:
            dataloader = self.trainer.get_train_dataloader()

        samples = generate_samples_remote(
            client,
            self.trainer.processing_class,
            dataloader,
            num_generation_samples=getattr(cfg, "num_generation_samples", 3),
            max_new_tokens=getattr(cfg, "generation_max_new_tokens", 50),
            temperature=getattr(cfg, "generation_temperature", 0.7),
            top_p=getattr(cfg, "generation_top_p", None),
            top_k=getattr(cfg, "generation_top_k", None),
            do_sample=getattr(cfg, "generation_do_sample", True),
            prompt_ratio=getattr(cfg, "generation_prompt_ratio", 0.5),
            colocate=bool(getattr(acfg, "colocate", False)),
        )
        self._log_samples(samples, state.global_step)

    def _log_samples(self, samples: list[dict], step: int) -> None:
        for i, sample in enumerate(samples):
            console_text, wandb_text = format_generation_for_logging(sample, i, step)
            LOG.info(console_text)
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({f"samples/sample_{i}": wandb.Html(f"<pre>{wandb_text}</pre>")}, step=step)
            except Exception:  # noqa: BLE001
                pass
