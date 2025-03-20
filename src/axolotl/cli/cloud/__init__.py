"""
launch axolotl in supported cloud platforms
"""

from pathlib import Path
from typing import Union

import yaml

from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.cloud.modal_ import ModalCloud
from axolotl.utils.dict import DictDefault


def load_cloud_cfg(cloud_config: Union[Path, str]) -> DictDefault:
    """Load and validate cloud configuration."""
    # Load cloud configuration.
    with open(cloud_config, encoding="utf-8") as file:
        cloud_cfg: DictDefault = DictDefault(yaml.safe_load(file))
    return cloud_cfg


def do_cli_preprocess(
    cloud_config: Union[Path, str],
    config: Union[Path, str],
) -> None:
    print_axolotl_text_art()
    cloud_cfg = load_cloud_cfg(cloud_config)
    cloud = ModalCloud(cloud_cfg)
    with open(config, "r", encoding="utf-8") as file:
        config_yaml = file.read()
    cloud.preprocess(config_yaml)


def do_cli_train(
    cloud_config: Union[Path, str],
    config: Union[Path, str],
    accelerate: bool = True,
    cwd=None,
    **kwargs,
) -> None:
    print_axolotl_text_art()
    cloud_cfg = load_cloud_cfg(cloud_config)
    cloud = ModalCloud(cloud_cfg)
    with open(config, "r", encoding="utf-8") as file:
        config_yaml = file.read()
    local_dirs = {}
    if cwd and not Path(cwd).joinpath("src", "axolotl").exists():
        local_dirs = {"/workspace/mounts": cwd}
    cloud.train(config_yaml, accelerate=accelerate, local_dirs=local_dirs, **kwargs)


def do_cli_lm_eval(
    cloud_config: Union[Path, str],
    config: Union[Path, str],
) -> None:
    print_axolotl_text_art()
    cloud_cfg = load_cloud_cfg(cloud_config)
    cloud = ModalCloud(cloud_cfg)
    with open(config, "r", encoding="utf-8") as file:
        config_yaml = file.read()
    cloud.lm_eval(config_yaml)
