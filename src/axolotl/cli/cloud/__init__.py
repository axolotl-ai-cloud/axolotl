"""
launch axolotl in supported cloud platforms
"""

from pathlib import Path
from typing import Literal

import yaml

from axolotl.cli.cloud.base import Cloud
from axolotl.cli.cloud.baseten import BasetenCloud
from axolotl.cli.cloud.modal_ import ModalCloud
from axolotl.utils.dict import DictDefault


def load_cloud_cfg(cloud_config: Path | str) -> DictDefault:
    """Load and validate cloud configuration."""
    # Load cloud configuration.
    with open(cloud_config, encoding="utf-8") as file:
        cloud_cfg: DictDefault = DictDefault(yaml.safe_load(file))
    return cloud_cfg


def do_cli_preprocess(
    cloud_config: Path | str,
    config: Path | str,
) -> None:
    cloud_cfg = load_cloud_cfg(cloud_config)
    cloud = ModalCloud(cloud_cfg)
    with open(config, "r", encoding="utf-8") as file:
        config_yaml = file.read()
    cloud.preprocess(config_yaml)


def do_cli_train(
    cloud_config: Path | str,
    config: Path | str,
    launcher: Literal["accelerate", "torchrun", "python"] = "accelerate",
    launcher_args: list[str] | None = None,
    cwd=None,
    **kwargs,
) -> None:
    cloud_cfg: DictDefault = load_cloud_cfg(cloud_config)
    provider = cloud_cfg.provider or "modal"
    cloud: Cloud | None
    if provider == "modal":
        cloud = ModalCloud(cloud_cfg)
    elif provider == "baseten":
        cloud = BasetenCloud(cloud_cfg.to_dict())
    else:
        raise ValueError(f"Unsupported cloud provider: {provider}")
    with open(config, "r", encoding="utf-8") as file:
        config_yaml = file.read()
    local_dirs = {}
    if cwd and not Path(cwd).joinpath("src", "axolotl").exists():
        local_dirs = {"/workspace/mounts": cwd}
    cloud.train(
        config_yaml,
        launcher=launcher,
        launcher_args=launcher_args,
        local_dirs=local_dirs,
        **kwargs,
    )


def do_cli_lm_eval(
    cloud_config: Path | str,
    config: Path | str,
) -> None:
    cloud_cfg = load_cloud_cfg(cloud_config)
    cloud = ModalCloud(cloud_cfg)
    with open(config, "r", encoding="utf-8") as file:
        config_yaml = file.read()
    cloud.lm_eval(config_yaml)
