"""Callbacks for diffusion training."""

import logging
import sys

import wandb
from colorama import Fore, Style
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .generation import generate_samples

# Simpler logger for more readable sample generation
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
logger.setLevel(logging.INFO)


class DiffusionGenerationCallback(TrainerCallback):
    """Callback for generating samples during diffusion training."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Generate samples at specified intervals."""
        if (
            state.global_step > 0
            and state.global_step % self.trainer.cfg.diffusion.generation_interval == 0
        ):
            if not self.trainer.state.is_world_process_zero:
                return

            # Use eval dataloader if available, otherwise use train dataloader
            dataloader = None
            try:
                if getattr(self.trainer, "eval_dataset", None) is not None:
                    dataloader = self.trainer.get_eval_dataloader()
            except Exception:
                dataloader = None
            if dataloader is None:
                dataloader = self.trainer.get_train_dataloader()

            # Generate samples
            diffusion_cfg = self.trainer.cfg.diffusion
            samples = generate_samples(
                model=self.trainer.model,
                tokenizer=self.trainer.processing_class,
                dataloader=dataloader,
                num_generation_samples=diffusion_cfg.num_generation_samples,
                max_length=diffusion_cfg.generation_max_length,
                num_diffusion_steps=diffusion_cfg.generation_steps,
                temperature=diffusion_cfg.generation_temperature,
                mask_token_id=diffusion_cfg.mask_token_id,
            )

            # Log samples
            self._log_samples(samples, state.global_step)

    def _log_samples(self, samples: list, step: int):
        """Log generated samples."""
        if not samples:
            return

        logger.info("=" * 60)
        logger.info("GENERATED SAMPLES")
        logger.info("=" * 60)

        for i, sample_data in enumerate(samples, 1):
            original = sample_data["original"]
            masked = sample_data["masked"]
            generated = sample_data["generated"]
            mask_ratio = sample_data["mask_ratio"]
            masked_tokens = sample_data["masked_tokens"]
            total_tokens = sample_data["total_tokens"]

            logger.info(f"\nSample {i}:")
            logger.info(f"\tOriginal ({total_tokens} tokens): {original}")
            logger.info(
                f"\tMasked ({masked_tokens}/{total_tokens} tokens, "
                f"{mask_ratio:.1%}): {masked}"
            )

            try:
                gen_ids = sample_data.get("generated_ids")
                orig_ids = sample_data.get("orig_ids")
                masked_positions = set(sample_data.get("masked_positions") or [])
                if isinstance(gen_ids, list) and isinstance(orig_ids, list):
                    styles: list[str] = []
                    for i, tid in enumerate(gen_ids):
                        if i in masked_positions:
                            if i < len(orig_ids) and tid == orig_ids[i]:
                                styles.append("green")
                            elif i < len(orig_ids):
                                styles.append("red")
                            else:
                                styles.append("normal")
                        else:
                            same = i < len(orig_ids) and tid == orig_ids[i]
                            styles.append("dim" if same else "normal")

                    spans: list[tuple[str, int, int]] = []
                    if gen_ids:
                        cur = styles[0]
                        start = 0
                        for i in range(1, len(gen_ids)):
                            s = styles[i]
                            if s != cur:
                                spans.append((cur, start, i))
                                cur, start = s, i
                        spans.append((cur, start, len(gen_ids)))

                    parts = []
                    for style_name, a, b in spans:
                        chunk_text = self.trainer.processing_class.decode(
                            gen_ids[a:b], skip_special_tokens=False
                        )
                        if style_name == "green":
                            parts.append(Fore.GREEN + chunk_text + Style.RESET_ALL)
                        elif style_name == "red":
                            parts.append(Fore.RED + chunk_text + Style.RESET_ALL)
                        else:
                            if style_name == "dim":
                                parts.append(Style.DIM + chunk_text + Style.RESET_ALL)
                            else:
                                parts.append(chunk_text)
                    logger.info("\tGenerated:\n%s", "".join(parts))
                else:
                    logger.info(f"\tGenerated: {generated}")
            except Exception:
                logger.info(f"\tGenerated: {generated}")

        logger.info("=" * 60)

        if self.trainer.cfg.use_wandb:
            if wandb.run is not None:
                wandb.log(
                    {
                        "generated_samples": wandb.Table(
                            columns=[
                                "step",
                                "original",
                                "masked",
                                "generated",
                                "mask_ratio",
                                "masked_tokens",
                                "total_tokens",
                            ],
                            data=[
                                [
                                    step,
                                    sample["original"],
                                    sample["masked"],
                                    sample["generated"],
                                    f"{sample['mask_ratio']:.1%}",
                                    sample["masked_tokens"],
                                    sample["total_tokens"],
                                ]
                                for sample in samples
                            ],
                        )
                    },
                    step=step,
                )
