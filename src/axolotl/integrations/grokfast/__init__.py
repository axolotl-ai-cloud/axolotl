"""
Grokfast plugin for Axolotl
"""

import logging

from transformers.trainer_callback import TrainerCallback

from ..base import BasePlugin
from .args import GrokfastArgs  # pylint: disable=unused-import. # noqa: F401
from .optimizer import gradfilter_ema

LOG = logging.getLogger("axolotl.integrations.grokfast")


class GrokfastCallbackHandler(TrainerCallback):
    """
    Transformer trainer callbacks for Grokfast
    """

    def __init__(self, *args_, alpha=0.98, lamb=2.0, **kwargs):
        super().__init__(*args_, **kwargs)
        self.grads = None
        self.alpha = alpha
        self.lamb = lamb

    def on_train_begin(self, *args_, **kwargs):  # pylint: disable=unused-argument
        self.grads = None

    def on_pre_optimizer_step(
        self, args_, state, control, **kwargs
    ):  # pylint: disable=unused-argument
        model = kwargs.pop("model")
        self.grads = gradfilter_ema(model, self.grads, alpha=self.alpha, lamb=self.lamb)
        return control


class GrokfastPlugin(BasePlugin):
    """
    Plugin for Grokfast optimizer integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.grokfast.GrokfastArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        LOG.info("Adding Grokfast callback to the trainer")
        callback = GrokfastCallbackHandler(
            alpha=cfg.grokfast_alpha, lamb=cfg.grokfast_lamb
        )
        return [callback]
