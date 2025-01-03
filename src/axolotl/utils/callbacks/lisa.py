"""
module for LISA

Adapted from https://github.com/OptimalScale/LMFlow/pull/701 for HF transformers & Axolotl
Arxiv: https://arxiv.org/abs/2403.17919
License: Apache 2.0
"""

import logging
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from transformers import TrainerCallback

if TYPE_CHECKING:
    from axolotl.core.trainer_builder import AxolotlTrainer

LOG = logging.getLogger("axolotl.callbacks.lisa")


def lisa_callback_factory(trainer: "AxolotlTrainer"):
    class LISACallback(TrainerCallback):
        """trainer callback for lisa layer switching"""

        def __init__(
            self, n_layers, step_interval, trainer, layers_attribute="model.layers"
        ):
            super().__init__()
            self.n_layers = n_layers
            self.step_interval = step_interval
            self.layers_attribute = layers_attribute
            self.trainer = trainer

            reduce(getattr, self.layers_attribute.split("."), self.trainer.model)

            self.total_layers = len(
                reduce(getattr, self.layers_attribute.split("."), self.trainer.model)
            )
            self.active_layers_indices = []

            layers = reduce(
                getattr, self.layers_attribute.split("."), self.trainer.model
            )
            LOG.info(
                f"LISA will activate {self.n_layers}/{len(layers)} layers ({self.n_layers * 100 / len(layers)}%) every {self.step_interval} steps"
            )

        def freeze_all_layers(self):
            layers = reduce(
                getattr, self.layers_attribute.split("."), self.trainer.model
            )
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        def on_step_begin(
            self, args, state, control, **kwargs
        ):  # pylint: disable=unused-argument
            # Check if it's time to switch active layers, including at step 0
            if state.global_step % self.step_interval == 0 or state.global_step == 1:
                self.switch_active_layers()

        def switch_active_layers(self):
            # First, disable gradients for all layers
            self.freeze_all_layers()

            # Randomly select n_layers to activate
            layers = reduce(
                getattr, self.layers_attribute.split("."), self.trainer.model
            )
            self.active_layers_indices = np.random.choice(
                range(self.total_layers), self.n_layers, replace=False
            )
            LOG.info(
                f"Activating layers at indices: {self.active_layers_indices} for the next steps."
            )

            # Enable gradients only for the selected layers
            for idx in self.active_layers_indices:
                for param in layers[idx].parameters():
                    param.requires_grad = True

    lisa_callback = LISACallback(
        n_layers=trainer.args.lisa_n_layers,
        step_interval=trainer.args.lisa_step_interval,
        trainer=trainer,
        layers_attribute=trainer.args.lisa_layers_attribute,
    )

    return lisa_callback
