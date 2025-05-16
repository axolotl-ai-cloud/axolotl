"""MLFlow module for trainer callbacks"""

import logging
import os
from shutil import copyfile
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import mlflow
from transformers import TrainerCallback, TrainerControl, TrainerState

from axolotl.utils.distributed import is_main_process

if TYPE_CHECKING:
    from axolotl.core.trainer_builder import AxolotlTrainingArguments

LOG = logging.getLogger("axolotl.callbacks")


def should_log_artifacts() -> bool:
    truths = ["TRUE", "1", "YES"]
    return os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper() in truths


class SaveAxolotlConfigtoMlflowCallback(TrainerCallback):
    # pylint: disable=duplicate-code
    """Callback to save axolotl config to mlflow"""

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
                if should_log_artifacts():
                    with NamedTemporaryFile(
                        mode="w", delete=False, suffix=".yml", prefix="axolotl_config_"
                    ) as temp_file:
                        copyfile(self.axolotl_config_path, temp_file.name)
                        mlflow.log_artifact(temp_file.name, artifact_path="")
                        LOG.info(
                            "The Axolotl config has been saved to the MLflow artifacts."
                        )
                else:
                    LOG.info(
                        "Skipping logging artifacts to MLflow (hf_mlflow_log_artifacts is false)"
                    )
            except (FileNotFoundError, ConnectionError) as err:
                LOG.warning(f"Error while saving Axolotl config to MLflow: {err}")
        return control
