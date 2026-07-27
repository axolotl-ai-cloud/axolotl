"""Launcher backends for axolotl training (accelerate, torchrun, ray, python)."""

import os
from typing import TYPE_CHECKING, Literal

from axolotl.cli.launchers.accelerate import _launch_accelerate_training
from axolotl.cli.launchers.base import build_command, run_command
from axolotl.cli.launchers.torchrun import (
    _add_default_rdzv_args,
    _launch_torchrun_training,
)

if TYPE_CHECKING:
    from axolotl.utils.schemas.runtime import RuntimeConfig

LauncherChoice = Literal["accelerate", "torchrun", "ray", "python"]


def launch_training(
    cfg_file: str,
    launcher: LauncherChoice | None,
    cloud: str | None,
    kwargs: dict,
    launcher_args: list[str] | None = None,
    use_exec: bool = False,
    env: dict[str, str] | None = None,
    runtime: "RuntimeConfig | None" = None,
) -> None:
    """Execute training with the given configuration."""
    launcher_args = launcher_args or []

    if cloud:
        _launch_cloud_training(cloud, cfg_file, launcher, kwargs, launcher_args)
    elif launcher:
        if launcher == "accelerate":
            _launch_accelerate_training(cfg_file, kwargs, launcher_args, use_exec, env)
        elif launcher == "torchrun":
            _launch_torchrun_training(cfg_file, kwargs, launcher_args, use_exec, env)
        elif launcher == "ray":
            from axolotl.cli.launchers.ray_ import launch_ray_training

            launch_ray_training(cfg_file, kwargs, runtime, env=env)
        elif launcher == "python":
            _launch_python_training(cfg_file, kwargs)
    elif launcher is None:
        # legacy programmatic path (pre-runtime-config callers passed None for ray)
        _launch_python_training(cfg_file, kwargs)


def _launch_cloud_training(
    cloud: str,
    cfg_file: str,
    launcher: LauncherChoice | None,
    kwargs: dict,
    launcher_args: list[str] | None = None,
) -> None:
    """Execute training via cloud launcher."""
    from axolotl.cli.cloud import do_cli_train

    launcher_args = launcher_args or []
    cwd = os.getcwd() if launcher else None

    do_cli_train(
        cloud_config=cloud,
        config=cfg_file,
        launcher=launcher or "accelerate",
        launcher_args=launcher_args,
        cwd=cwd,
        **kwargs,
    )


def _launch_python_training(cfg_file: str, kwargs: dict) -> None:
    """Execute training via python launcher."""
    from axolotl.cli.train import do_cli

    do_cli(config=cfg_file, **kwargs)


__all__ = [
    "LauncherChoice",
    "build_command",
    "launch_training",
    "run_command",
    "_add_default_rdzv_args",
    "_launch_accelerate_training",
    "_launch_cloud_training",
    "_launch_python_training",
    "_launch_torchrun_training",
]
