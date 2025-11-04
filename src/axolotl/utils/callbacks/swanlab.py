"""Callbacks for SwanLab integration"""

from __future__ import annotations

import json
import os
from shutil import copyfile
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from transformers import TrainerCallback, TrainerControl, TrainerState

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.core.training_args import AxolotlTrainingArguments

LOG = get_logger(__name__)


class SaveAxolotlConfigtoSwanLabCallback(TrainerCallback):
    """Callback to save axolotl config to SwanLab"""

    def __init__(self, axolotl_config_path):
        self.axolotl_config_path = axolotl_config_path

    def on_train_begin(
        self,
        args: AxolotlTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            try:
                import swanlab

                # Check if SwanLab is initialized
                if swanlab.get_run() is None:
                    LOG.warning(
                        "SwanLab run is not initialized. Please initialize SwanLab before training."
                    )
                    return control

                # Log Axolotl config as artifact
                with NamedTemporaryFile(
                    mode="w", delete=False, suffix=".yml", prefix="axolotl_config_"
                ) as temp_file:
                    copyfile(self.axolotl_config_path, temp_file.name)

                    # Log config file to SwanLab
                    swanlab.log(
                        {
                            "axolotl_config": swanlab.Text(
                                open(temp_file.name).read(), caption="Axolotl Config"
                            )
                        }
                    )

                    LOG.info(
                        "The Axolotl config has been saved to the SwanLab run under logs."
                    )

                    # Clean up temp file
                    os.unlink(temp_file.name)

            except ImportError:
                LOG.warning(
                    "SwanLab is not installed. Install it with: pip install swanlab"
                )
            except (FileNotFoundError, ConnectionError) as err:
                LOG.warning(f"Error while saving Axolotl config to SwanLab: {err}")

            # Log DeepSpeed config if available
            if args.deepspeed:
                try:
                    import swanlab

                    with NamedTemporaryFile(
                        mode="w",
                        delete=False,
                        suffix=".json",
                        prefix="deepspeed_config_",
                    ) as temp_file:
                        skip_upload = False
                        if isinstance(args.deepspeed, dict):
                            json.dump(args.deepspeed, temp_file, indent=4)
                        elif isinstance(args.deepspeed, str) and os.path.exists(
                            args.deepspeed
                        ):
                            copyfile(args.deepspeed, temp_file.name)
                        else:
                            skip_upload = True

                        if not skip_upload:
                            temp_file.flush()
                            swanlab.log(
                                {
                                    "deepspeed_config": swanlab.Text(
                                        open(temp_file.name).read(),
                                        caption="DeepSpeed Config",
                                    )
                                }
                            )
                            LOG.info(
                                "The DeepSpeed config has been saved to the SwanLab run under logs."
                            )

                        # Clean up temp file
                        os.unlink(temp_file.name)

                except (FileNotFoundError, ConnectionError) as err:
                    LOG.warning(f"Error while saving DeepSpeed config to SwanLab: {err}")
                except ImportError:
                    pass

        return control






