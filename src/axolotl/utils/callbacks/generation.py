"""Callback for generating samples during SFT/Pretrain training."""

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from axolotl.utils.generation.sft import generate_samples
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class SFTGenerationCallback(TrainerCallback):
    """Callback for generating samples during SFT/Pretrain training."""

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
        cfg = self.trainer.axolotl_cfg

        if not getattr(cfg, "generate_samples", False):
            return

        generation_interval = getattr(cfg, "generation_interval", 100)

        if state.global_step > 0 and state.global_step % generation_interval == 0:
            if not self.trainer.state.is_world_process_zero:
                return

            dataloader = None
            try:
                if getattr(self.trainer, "eval_dataset", None) is not None:
                    dataloader = self.trainer.get_eval_dataloader()
                    LOG.info(
                        f"Using eval dataloader for generation at step {state.global_step}"
                    )
            except Exception as e:
                LOG.warning(f"Could not get eval dataloader: {e}")
                dataloader = None

            if dataloader is None:
                dataloader = self.trainer.get_train_dataloader()
                LOG.info(
                    f"Using train dataloader for generation at step {state.global_step}"
                )

            samples = generate_samples(
                model=self.trainer.model,
                tokenizer=self.trainer.processing_class,
                dataloader=dataloader,
                num_generation_samples=getattr(cfg, "num_generation_samples", 3),
                max_new_tokens=getattr(cfg, "generation_max_new_tokens", 50),
                temperature=getattr(cfg, "generation_temperature", 0.7),
                top_p=getattr(cfg, "generation_top_p", None),
                top_k=getattr(cfg, "generation_top_k", None),
                do_sample=getattr(cfg, "generation_do_sample", True),
                prompt_ratio=getattr(cfg, "generation_prompt_ratio", 0.5),
            )
            self._log_samples(samples, state.global_step)

    def _log_samples(self, samples: list, step: int):
        """Log generated samples to console and W&B."""
        from axolotl.utils.generation.sft import format_generation_for_logging

        for i, sample in enumerate(samples):
            console_text, wandb_text = format_generation_for_logging(sample, i, step)

            LOG.info(console_text)

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            f"samples/sample_{i + 1}": wandb.Html(
                                f"<pre>{wandb_text}</pre>"
                            )
                        },
                        step=step,
                    )
            except (ImportError, Exception):
                pass
