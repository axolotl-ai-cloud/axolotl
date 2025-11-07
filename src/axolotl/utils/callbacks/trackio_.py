"""Trackio module for trainer callbacks"""

from typing import TYPE_CHECKING

from transformers import TrainerCallback, TrainerControl, TrainerState

from axolotl.utils.distributed import is_main_process
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
            LOG.info("Trackio experiment tracking is enabled.")
        return control

