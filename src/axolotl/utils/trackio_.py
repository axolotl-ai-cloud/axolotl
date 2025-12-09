"""Module for trackio utilities"""

import os

from axolotl.utils.dict import DictDefault


def setup_trackio_env_vars(cfg: DictDefault):
    for key in cfg.keys():
        if key.startswith("trackio_"):
            value = cfg.get(key, "")

            if value and isinstance(value, str) and len(value) > 0:
                os.environ[key.upper()] = value

    if cfg.trackio_project_name and len(cfg.trackio_project_name) > 0:
        cfg.use_trackio = True
