"""Diffusion LM training plugin for Axolotl."""

from typing import TYPE_CHECKING

from peft import PeftModel
from transformers import AutoConfig, AutoModel, PreTrainedModel

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .callbacks import DiffusionGenerationCallback
from .configuration import LlamaForDiffusionConfig, MistralForDiffusionConfig
from .models import LlamaForDiffusionLM, MistralForDiffusionLM

if TYPE_CHECKING:
    from transformers import Trainer

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

    def get_input_args(self) -> str:
        """Returns the pydantic model for LLaDA plugin arguments."""
        return "axolotl.integrations.diffusion.DiffusionArgs"

    def pre_model_load(self, cfg: DictDefault):
        """Configure model loading to use diffusion model classes."""
        # Map base model types to diffusion equivalents
        base_model_type = cfg.get("model_type")
        
        if base_model_type == "llama":
            # Create diffusion config from base config
            diffusion_config = LlamaForDiffusionConfig(
                mask_token_id=getattr(cfg, "mask_token_id", 32000),
                eps=getattr(cfg, "eps", 1e-3),
                importance_weighting=getattr(cfg, "importance_weighting", False),
                sample_packing=getattr(cfg, "sample_packing", False),
                min_mask_ratio=getattr(cfg, "min_mask_ratio", 0.0),
                max_mask_ratio=getattr(cfg, "max_mask_ratio", 1.0),
                noise_schedule=getattr(cfg, "noise_schedule", "linear"),
            )
            
            # Override model type for loading
            cfg.model_type = "llama_diffusion"
            
        elif base_model_type == "mistral":
            # Create diffusion config from base config
            diffusion_config = MistralForDiffusionConfig(
                mask_token_id=getattr(cfg, "mask_token_id", 32000),
                eps=getattr(cfg, "eps", 1e-3),
                importance_weighting=getattr(cfg, "importance_weighting", False),
                sample_packing=getattr(cfg, "sample_packing", False),
                min_mask_ratio=getattr(cfg, "min_mask_ratio", 0.0),
                max_mask_ratio=getattr(cfg, "max_mask_ratio", 1.0),
                noise_schedule=getattr(cfg, "noise_schedule", "linear"),
            )
            
            # Override model type for loading
            cfg.model_type = "mistral_diffusion"
        else:
            LOG.warning(f"Diffusion plugin not implemented for model type: {base_model_type}")

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Configure model after loading."""
        self.cfg = cfg
        
        # Set tokenizer on diffusion models for special token handling
        if hasattr(model, "set_tokenizer"):
            # Get tokenizer from cfg if available
            tokenizer = getattr(cfg, "tokenizer", None)
            if tokenizer is not None:
                model.set_tokenizer(tokenizer)

    def add_callbacks_post_trainer(self, cfg: DictDefault, trainer: "Trainer"):
        """Add diffusion-specific callbacks after trainer creation."""
        callbacks = []
        
        # Store diffusion config on trainer for callbacks
        trainer.diffusion_config = cfg
        
        # Add generation callback if enabled
        if cfg.get("generate_samples", False):
            generation_callback = DiffusionGenerationCallback(trainer)
            callbacks.append(generation_callback)
            
        return callbacks
