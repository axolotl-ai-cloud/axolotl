"""Callbacks for diffusion training."""

import wandb
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from axolotl.utils.logging import get_logger

from .generation import generate_samples

LOG = get_logger(__name__)


class DiffusionGenerationCallback(TrainerCallback):
    """Callback for generating samples during diffusion training."""

    def __init__(self, trainer):
        self.trainer = trainer

    # pylint: disable=unused-argument
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Generate samples at specified intervals."""
        config = getattr(self.trainer, 'diffusion_config', self.trainer.args)
        
        if (
            state.global_step > 0
            and state.global_step % config.get('generation_interval', 100) == 0
        ):
            # Use eval dataloader if available, otherwise use train dataloader
            if (
                hasattr(self.trainer, "eval_dataset")
                and self.trainer.eval_dataset is not None
            ):
                dataloader = self.trainer.get_eval_dataloader()
            else:
                dataloader = self.trainer.get_train_dataloader()

            # Generate samples
            samples = generate_samples(
                model=self.trainer.model,
                tokenizer=self.trainer.tokenizer,
                dataloader=dataloader,
                num_generation_samples=config.get('num_generation_samples', 3),
                max_length=config.get('generation_max_length', 256),
                num_diffusion_steps=config.get('generation_steps', 10),
                temperature=config.get('generation_temperature', 1.0),
                mask_token_id=config.get('mask_token_id', 32000),
            )

            # Log samples
            self._log_samples(samples, state.global_step)

    def _log_samples(self, samples: list, step: int):
        """Log generated samples."""
        if not samples:
            return

        LOG.info("=" * 60)
        LOG.info("GENERATED SAMPLES")
        LOG.info("=" * 60)

        for i, sample_data in enumerate(samples, 1):
            original = sample_data["original"]
            masked = sample_data["masked"]
            generated = sample_data["generated"]
            mask_ratio = sample_data["mask_ratio"]
            masked_tokens = sample_data["masked_tokens"]
            total_tokens = sample_data["total_tokens"]

            LOG.info(f"\nSample {i}:")
            LOG.info(f"\tOriginal ({total_tokens} tokens): {original}")
            LOG.info(
                f"\tMasked ({masked_tokens}/{total_tokens} tokens, "
                f"{mask_ratio:.1%}): {masked}"
            )
            LOG.info(f"\tGenerated: {generated}")

        LOG.info("=" * 60)

        config = getattr(self.trainer, 'diffusion_config', self.trainer.args)
        if config.get('use_wandb', False) and self.trainer.state.is_world_process_zero:
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
