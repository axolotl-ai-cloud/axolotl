"""Module for mlflow utilities"""

import os

from axolotl.utils.dict import DictDefault


def setup_mlflow_env_vars(cfg: DictDefault):
    for key in cfg.keys():
        if key.startswith("mlflow_") or key.startswith("hf_mlflow_"):
            value = cfg.get(key, "")

            if value and isinstance(value, str) and len(value) > 0:
                os.environ[key.upper()] = value

    # Enable mlflow if experiment name is present
    if cfg.mlflow_experiment_name and len(cfg.mlflow_experiment_name) > 0:
        cfg.use_mlflow = True

    # Enable logging hf artifacts in mlflow if value is truthy
    if cfg.hf_mlflow_log_artifacts is True:
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "true"
