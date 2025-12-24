"""Trackio module for trainer callbacks"""

from typing import TYPE_CHECKING

import trackio
from transformers import TrainerCallback, TrainerControl, TrainerState

from axolotl.utils.distributed import is_main_process
from axolotl.utils.environment import is_package_version_ge
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.core.training_args import AxolotlTrainingArguments

LOG = get_logger(__name__)


class SaveAxolotlConfigtoTrackioCallback(TrainerCallback):
    """Callback for trackio integration"""

    def __init__(self, axolotl_config_path):
        self.axolotl_config_path = axolotl_config_path

    def on_train_begin(
        self,
        args: "AxolotlTrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if is_main_process():
            try:
                if not is_package_version_ge("trackio", "0.11.0"):
                    LOG.warning(
                        "Trackio version 0.11.0 or higher is required to save config files. "
                        "Please upgrade trackio: pip install --upgrade trackio"
                    )
                    return control

                trackio.save(self.axolotl_config_path)
                LOG.info("The Axolotl config has been saved to Trackio.")
            except (FileNotFoundError, ConnectionError, AttributeError) as err:
                LOG.warning(f"Error while saving Axolotl config to Trackio: {err}")
        return control
