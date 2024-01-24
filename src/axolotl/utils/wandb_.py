"""Module for wandb utilities"""

import os

from axolotl.utils.dict import DictDefault


def setup_wandb_env_vars(cfg: DictDefault):
    for key in cfg.keys():
        if key.startswith("wandb_"):
            value = cfg.get(key, "")

            if value and isinstance(value, str) and len(value) > 0:
                os.environ[key.upper()] = value

    # Enable wandb if project name is present
    if cfg.wandb_project and len(cfg.wandb_project) > 0:
        cfg.use_wandb = True
        os.environ.pop("WANDB_DISABLED", None)  # Remove if present
    else:
        os.environ["WANDB_DISABLED"] = "true"
