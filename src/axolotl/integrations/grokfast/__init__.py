"""
Grokfast plugin for Axolotl
"""
from transformers.trainer_callback import CallbackHandler

from ..base import BasePlugin
from .args import GrokfastArgs  # pylint: disable=unused-import. # noqa: F401
from .optimizer import gradfilter_ema


class GrokfastCallbackHandler(CallbackHandler):
    """
    Transformer trainer callbacks for Grokfast
    """

    def __init__(self, *args, alpha=0.98, lamb=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.grads = None
        self.alpha = alpha
        self.lamb = lamb

    def on_train_begin(self, args, state):  # pylint: disable=unused-argument
        self.grads = None

    def on_pre_optimizer_step(
        self, args, state, control, model
    ):  # pylint: disable=unused-argument
        self.grads = gradfilter_ema(model, self.grads, alpha=self.alpha, lamb=self.lamb)
        return control


class GrokfastPlugin(BasePlugin):
    """
    Plugin for Grokfast optimizer integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.grokfast.GrokfastArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        callback = GrokfastCallbackHandler(
            alpha=cfg.grokfast_alpha, lamb=cfg.grokfast_lamb
        )
        return [callback]
