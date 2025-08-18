"""Diffusion LM training plugin for Axolotl."""

from peft import PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .args import DiffusionArgs
from .callbacks import DiffusionGenerationCallback
from .loss import register_diffusion_loss
from .model_patch import patch_model_for_bidirectional_attention

LOG = get_logger(__name__)


class DiffusionPlugin(BasePlugin):
    """
    Plugin for diffusion language model training.

    This plugin enables diffusion-based training using the LLaDA approach, which uses
    random masking and bidirectional attention to train language models.
    """

    def __init__(self):
        super().__init__()
        self.cfg = None

        if register_diffusion_loss():
            LOG.info("Registered ForDiffusionLM loss function")
        else:
            LOG.warning(
                "Failed to register diffusion loss - older transformers version"
            )

    def get_input_args(self) -> str:
        """Returns the pydantic model for LLaDA plugin arguments."""
        return "axolotl.integrations.diffusion.DiffusionArgs"

    def post_model_load(
        self, cfg: DictDefault, model: PreTrainedModel | PeftModel
    ):
        """Configure model for diffusion training after loading."""
        self.cfg = cfg

        # Set loss type for diffusion training
        if hasattr(model, "config"):
            model.config.loss_type = "ForDiffusionLM"

            # Store diffusion config in model config
            model.config.diffusion_config = {
                "eps": getattr(cfg, "eps", 1e-3),
                "importance_weighting": getattr(cfg, "importance_weighting", True),
                "mask_token_id": getattr(cfg, "mask_token_id", 128002),
            }

            LOG.info("Configured model for diffusion training with ForDiffusionLM loss")

        # Patch model for bidirectional attention during training
        patch_model_for_bidirectional_attention(model)
        LOG.info("Applied bidirectional attention patch to model")

        return model

    def post_trainer_create(self, cfg: DictDefault, trainer):
        """Configure trainer after creation."""
        # Create diffusion config from cfg
        diffusion_config = DiffusionArgs(
            noise_schedule=getattr(cfg, "noise_schedule", "linear"),
            min_mask_ratio=getattr(cfg, "min_mask_ratio", 0.1),
            max_mask_ratio=getattr(cfg, "max_mask_ratio", 0.9),
            num_diffusion_steps=getattr(cfg, "num_diffusion_steps", 128),
            eps=getattr(cfg, "eps", 1e-3),
            importance_weighting=getattr(cfg, "importance_weighting", True),
            mask_token_id=getattr(cfg, "mask_token_id", 128002),
            generate_samples=getattr(cfg, "generate_samples", True),
            generation_interval=getattr(cfg, "generation_interval", 100),
            num_generation_samples=getattr(cfg, "num_generation_samples", 3),
            generation_steps=getattr(cfg, "generation_steps", 128),
            generation_temperature=getattr(cfg, "generation_temperature", 0.0),
            generation_max_length=getattr(cfg, "generation_max_length", 100),
        )

        # Store diffusion config on trainer for callbacks to access
        trainer.diffusion_config = diffusion_config
        LOG.info("Stored diffusion config on trainer")

    def add_callbacks_post_trainer(self, cfg: DictDefault, trainer):
        """Add diffusion generation callback if enabled."""
        if hasattr(trainer, 'diffusion_config') and trainer.diffusion_config.generate_samples:
            generation_callback = DiffusionGenerationCallback(trainer)
            LOG.info("Added diffusion generation callback")
            return [generation_callback]
        return []
