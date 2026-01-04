"""Callbacks for SwanLab integration"""

from __future__ import annotations

import json
import os
from shutil import copyfile
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.core.training_args import AxolotlTrainingArguments

LOG = get_logger(__name__)


class CustomSwanLabCallback(TrainerCallback):
    """
    Lightweight SwanLab callback that directly logs metrics without using
    SwanLab's transformers integration (which requires omegaconf).

    This avoids the antlr4 version conflict between omegaconf and axolotl.
    """

    def __init__(self):
        self._initialized = False
        self.swanlab = None

    def setup(self):
        """Lazy initialization of SwanLab"""
        if self._initialized:
            return

        try:
            import swanlab

            self.swanlab = swanlab

            # Check if SwanLab run is initialized
            if swanlab.get_run() is None:
                LOG.warning("SwanLab run is not initialized")
                return

            self._initialized = True
            LOG.info("CustomSwanLabCallback initialized successfully")
        except ImportError:
            LOG.error("SwanLab is not installed")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the beginning of training"""
        if not state.is_world_process_zero:
            return control

        self.setup()

        if not self._initialized:
            return control

        # Log training configuration
        try:
            self.swanlab.config.update(
                {
                    "train_batch_size": args.per_device_train_batch_size,
                    "eval_batch_size": args.per_device_eval_batch_size,
                    "learning_rate": args.learning_rate,
                    "num_train_epochs": args.num_train_epochs,
                    "max_steps": args.max_steps,
                    "warmup_steps": args.warmup_steps,
                    "logging_steps": args.logging_steps,
                    "save_steps": args.save_steps,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                }
            )
            LOG.debug("Training configuration logged to SwanLab")
        except Exception as err:
            LOG.warning(f"Failed to log training config: {err}")

        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """Called when logging metrics"""
        if not state.is_world_process_zero:
            return control

        if not self._initialized:
            self.setup()

        if not self._initialized or logs is None:
            return control

        # Log metrics to SwanLab
        try:
            # Filter out non-numeric values and prepare for logging
            metrics = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    # Use step from state
                    metrics[key] = value

            if metrics and state.global_step is not None:
                self.swanlab.log(metrics, step=state.global_step)
        except Exception as err:
            LOG.warning(f"Failed to log metrics to SwanLab: {err}")

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training"""
        if not state.is_world_process_zero:
            return control

        if self._initialized:
            LOG.info("Training completed. SwanLab logs are available.")

        return control


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
                    with open(temp_file.name, "r", encoding="utf-8") as config_file:
                        swanlab.log(
                            {
                                "axolotl_config": swanlab.Text(
                                    config_file.read(), caption="Axolotl Config"
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
                            with open(
                                temp_file.name, "r", encoding="utf-8"
                            ) as ds_config_file:
                                swanlab.log(
                                    {
                                        "deepspeed_config": swanlab.Text(
                                            ds_config_file.read(),
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
                    LOG.warning(
                        f"Error while saving DeepSpeed config to SwanLab: {err}"
                    )
                except ImportError:
                    pass

        return control
