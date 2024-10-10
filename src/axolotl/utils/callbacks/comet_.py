"""Comet module for trainer callbacks"""

import logging
from typing import TYPE_CHECKING

import comet_ml
from transformers import TrainerCallback, TrainerControl, TrainerState

from axolotl.utils.distributed import is_main_process

if TYPE_CHECKING:
    from axolotl.core.trainer_builder import AxolotlTrainingArguments

LOG = logging.getLogger("axolotl.callbacks")


class SaveAxolotlConfigtoCometCallback(TrainerCallback):
    """Callback to save axolotl config to comet"""

    def __init__(self, axolotl_config_path):
        self.axolotl_config_path = axolotl_config_path

    def on_train_begin(
        self,
        args: "AxolotlTrainingArguments",  # pylint: disable=unused-argument
        state: TrainerState,  # pylint: disable=unused-argument
        control: TrainerControl,
        **kwargs,  # pylint: disable=unused-argument
    ):
        if is_main_process():
            try:
                comet_experiment = comet_ml.start(source="axolotl")
                comet_experiment.log_other("Created from", "axolotl")
                comet_experiment.log_asset(
                    self.axolotl_config_path,
                    file_name="axolotl-config",
                )
                LOG.info(
                    "The Axolotl config has been saved to the Comet Experiment under assets."
                )
            except (FileNotFoundError, ConnectionError) as err:
                LOG.warning(f"Error while saving Axolotl config to Comet: {err}")
        return control
