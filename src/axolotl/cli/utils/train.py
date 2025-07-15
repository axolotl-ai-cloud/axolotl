"""Utilities for axolotl train CLI command."""

import os
import subprocess  # nosec
from typing import Any


def build_command(base_cmd: list[str], options: dict[str, Any]) -> list[str]:
    """
    Build command list from base command and options.

    Args:
        base_cmd: Command without options.
        options: Options to parse and append to base command.

    Returns:
        List of strings giving shell command.
    """
    cmd = base_cmd.copy()

    for key, value in options.items():
        if value is None:
            continue

        key = key.replace("_", "-")
        cmd.append(f"--{key}={value}")

    return cmd


def execute_training(
    cfg_file: str, launcher: str | None, cloud: str | None, kwargs: dict
) -> None:
    """Execute training with the given configuration."""
    if cloud:
        _execute_cloud_training(cloud, cfg_file, launcher, kwargs)
    elif launcher:
        if launcher == "accelerate":
            _execute_accelerate_training(cfg_file, kwargs)
        elif launcher == "torchrun":
            _execute_torchrun_training(cfg_file, kwargs)
        elif launcher == "python":
            _execute_python_training(cfg_file, kwargs)


def _execute_cloud_training(
    cloud: str, cfg_file: str, launcher: str | None, kwargs: dict
) -> None:
    """Execute training via cloud launcher."""
    from axolotl.cli.cloud import do_cli_train

    accelerate = launcher == "accelerate" if launcher else False
    cwd = os.getcwd() if launcher else None

    do_cli_train(
        cloud_config=cloud,
        config=cfg_file,
        accelerate=accelerate,
        cwd=cwd,
        **kwargs,
    )


def _execute_accelerate_training(cfg_file: str, kwargs: dict) -> None:
    """Execute training via accelerate launcher."""
    launcher_args = []

    # Extract launcher-specific arguments
    if "main_process_port" in kwargs:
        main_process_port = kwargs.pop("main_process_port")
        launcher_args.extend(["--main_process_port", str(main_process_port)])

    if "num_processes" in kwargs:
        num_processes = kwargs.pop("num_processes")
        launcher_args.extend(["--num_processes", str(num_processes)])

    base_cmd = ["accelerate", "launch"] + launcher_args + ["-m", "axolotl.cli.train"]
    if cfg_file:
        base_cmd.append(cfg_file)

    cmd = build_command(base_cmd, kwargs)
    subprocess.run(cmd, check=True)  # nosec B603


def _execute_torchrun_training(cfg_file: str, kwargs: dict) -> None:
    """Execute training via torchrun launcher."""
    base_cmd = ["torchrun", "-m", "axolotl.cli.train"]
    if cfg_file:
        base_cmd.append(cfg_file)

    cmd = build_command(base_cmd, kwargs)
    subprocess.run(cmd, check=True)  # nosec B603


def _execute_python_training(cfg_file: str, kwargs: dict) -> None:
    """Execute training via python launcher."""
    from axolotl.cli.train import do_cli

    do_cli(config=cfg_file, **kwargs)
