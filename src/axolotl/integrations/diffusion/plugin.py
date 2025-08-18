"""Diffusion LM training plugin for Axolotl."""

from peft import PeftModel
from transformers import PreTrainedModel

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .trainer import DiffusionTrainer

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

    def post_model_load(self, cfg: DictDefault, model: PreTrainedModel | PeftModel):
        """Perform actions after model is loaded."""
        self.cfg = cfg

    def get_trainer_cls(self, cfg: DictDefault) -> type[DiffusionTrainer] | None:
        """Return custom trainer class for diffusion training."""
        return DiffusionTrainer

    def post_trainer_create(self, cfg: DictDefault, trainer: DiffusionTrainer):
        """Configure trainer after creation."""
        trainer.set_config(cfg)
