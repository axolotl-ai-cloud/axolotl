"""
launch axolotl in supported cloud platforms
"""

from pathlib import Path
from typing import Literal

import yaml

from axolotl.cli.cloud.base import Cloud
from axolotl.utils.dict import DictDefault


def load_cloud_provider(cloud_cfg: DictDefault) -> Cloud:
    """Import only the selected provider's dependencies."""
    provider = cloud_cfg.provider or "modal"
    if provider == "modal":
        from axolotl.cli.cloud.modal_ import ModalCloud

        return ModalCloud(cloud_cfg)
    if provider == "baseten":
        from axolotl.cli.cloud.baseten import BasetenCloud

        return BasetenCloud(cloud_cfg.to_dict())
    if provider == "nebius":
        from axolotl.cli.cloud.nebius import NebiusCloud

        return NebiusCloud(cloud_cfg.to_dict())
    raise ValueError(f"Unsupported cloud provider: {provider}")


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
    cloud = load_cloud_provider(cloud_cfg)
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
    cloud = load_cloud_provider(cloud_cfg)
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
    cloud = load_cloud_provider(cloud_cfg)
    with open(config, "r", encoding="utf-8") as file:
        config_yaml = file.read()
    if not hasattr(cloud, "lm_eval"):
        raise NotImplementedError(
            f"{cloud_cfg.provider} does not support cloud lm-eval"
        )
    cloud.lm_eval(config_yaml)
