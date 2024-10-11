"""Module for wandb utilities"""

import logging
import os

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.utils.comet_")

COMET_ENV_MAPPING_OVERRIDE = {
    "comet_mode": "COMET_START_MODE",
    "comet_online": "COMET_START_ONLINE",
}
COMET_EXPERIMENT_CONFIG_ENV_MAPPING_OVERRIDE = {
    "auto_histogram_activation_logging": "COMET_AUTO_LOG_HISTOGRAM_ACTIVATIONS",
    "auto_histogram_epoch_rate": "COMET_AUTO_LOG_HISTOGRAM_EPOCH_RATE",
    "auto_histogram_gradient_logging": "COMET_AUTO_LOG_HISTOGRAM_GRADIENTS",
    "auto_histogram_tensorboard_logging": "COMET_AUTO_LOG_HISTOGRAM_TENSORBOARD",
    "auto_histogram_weight_logging": "COMET_AUTO_LOG_HISTOGRAM_WEIGHTS",
    "auto_log_co2": "COMET_AUTO_LOG_CO2",
    "auto_metric_logging": "COMET_AUTO_LOG_METRICS",
    "auto_metric_step_rate": "COMET_AUTO_LOG_METRIC_STEP_RATE",
    "auto_output_logging": "COMET_AUTO_LOG_OUTPUT_LOGGER",
    "auto_param_logging": "COMET_AUTO_LOG_PARAMETERS",
    "comet_disabled": "COMET_AUTO_LOG_DISABLE",
    "display_summary_level": "COMET_DISPLAY_SUMMARY_LEVEL",
    "distributed_node_identifier": "COMET_DISTRIBUTED_NODE_IDENTIFIER",
    "log_code": "COMET_AUTO_LOG_CODE",
    "log_env_cpu": "COMET_AUTO_LOG_ENV_CPU",
    "log_env_details": "COMET_AUTO_LOG_ENV_DETAILS",
    "log_env_disk": "COMET_AUTO_LOG_ENV_DISK",
    "log_env_gpu": "COMET_AUTO_LOG_ENV_GPU",
    "log_env_host": "COMET_AUTO_LOG_ENV_HOST",
    "log_env_network": "COMET_AUTO_LOG_ENV_NETWORK",
    "log_git_metadata": "COMET_AUTO_LOG_GIT_METADATA",
    "log_git_patch": "COMET_AUTO_LOG_GIT_PATCH",
    "log_graph": "COMET_AUTO_LOG_GRAPH",
    "name": "COMET_START_EXPERIMENT_NAME",
    "offline_directory": "COMET_OFFLINE_DIRECTORY",
    "parse_args": "COMET_AUTO_LOG_CLI_ARGUMENTS",
    "tags": "COMET_START_EXPERIMENT_TAGS",
}


def python_value_to_environ_value(python_value):
    if isinstance(python_value, bool):
        if python_value is True:
            return "true"

        return "false"

    if isinstance(python_value, int):
        return str(python_value)

    if isinstance(python_value, list):  # Comet only have one list of string parameter
        return ",".join(map(str, python_value))

    return python_value


def setup_comet_env_vars(cfg: DictDefault):
    # TODO, we need to convert Axolotl configuration to environment variables
    # as Transformers integration are call first and would create an
    # Experiment first

    for key in cfg.keys():
        if key.startswith("comet_") and key != "comet_experiment_config":
            value = cfg.get(key, "")

            if value is not None and value != "":
                env_variable_name = COMET_ENV_MAPPING_OVERRIDE.get(key, key.upper())
                final_value = python_value_to_environ_value(value)
                os.environ[env_variable_name] = final_value

    if cfg.comet_experiment_config:
        for key, value in cfg.comet_experiment_config.items():
            if value is not None and value != "":
                config_env_variable_name = (
                    COMET_EXPERIMENT_CONFIG_ENV_MAPPING_OVERRIDE.get(key)
                )

                if config_env_variable_name is None:
                    LOG.warning(
                        f"Unknown Comet Experiment Config name {key}, ignoring it"
                    )
                    continue

                final_value = python_value_to_environ_value(value)
                os.environ[config_env_variable_name] = final_value

    # Enable comet if project name is present
    if cfg.comet_project_name and len(cfg.comet_project_name) > 0:
        cfg.use_comet = True
