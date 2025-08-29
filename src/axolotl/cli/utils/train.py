"""Utilities for axolotl train CLI command."""

import os
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Literal

import yaml

from axolotl.cli.utils.sweeps import generate_sweep_configs


def _add_default_rdzv_args(launcher_args: list[str]) -> list[str]:
    """
    Add default RDZV arguments if rdzv_endpoint is set but rdzv_backend/rdzv_id are missing.

    Args:
        launcher_args: List of launcher arguments

    Returns:
        Updated launcher args with defaults added if needed
    """
    args = launcher_args.copy()

    # Check if rdzv_endpoint is present
    has_rdzv_endpoint = any("--rdzv_endpoint" in arg for arg in args)

    if has_rdzv_endpoint:
        # Check if rdzv_backend is already provided
        has_rdzv_backend = any("--rdzv_backend" in arg for arg in args)
        if not has_rdzv_backend:
            args.extend(["--rdzv_backend", "c10d"])

        # Check if rdzv_id is already provided
        has_rdzv_id = any("--rdzv_id" in arg for arg in args)
        if not has_rdzv_id:
            import uuid

            args.extend(["--rdzv_id", str(uuid.uuid4())[:8]])

    return args


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


def generate_config_files(config: str, sweep: str | None) -> Iterator[tuple[str, bool]]:
    """
    Generate list of configuration files to process. Yields a tuple of the configuration file name and a boolean indicating
    whether this is a group of configurations (i.e., a sweep).

    Args:
        config: Base configuration file
        sweep: Sweep configuration file
    """

    if not sweep:
        yield config, False
        return

    # Load sweep and base configurations
    with open(sweep, "r", encoding="utf-8") as fin:
        sweep_config: dict[str, list] = yaml.safe_load(fin)
    with open(config, "r", encoding="utf-8") as fin:
        base_config: dict[str, list] = yaml.safe_load(fin)

    # Generate all possible configurations
    permutations = generate_sweep_configs(base_config, sweep_config)
    is_group = len(permutations) > 1
    base_output_dir = base_config.get("output_dir", "./model-out")
    for idx, permutation in enumerate(permutations, start=1):
        permutation_dir = Path(permutation.get("output_dir", base_output_dir))
        permutation_id = f"sweep{idx:04d}"
        permutation["output_dir"] = str(permutation_dir / permutation_id)

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        )
        yaml.dump(permutation, temp_file)
        temp_file.close()
        yield temp_file.name, is_group


def launch_training(
    cfg_file: str,
    launcher: Literal["accelerate", "torchrun", "python"] | None,
    cloud: str | None,
    kwargs: dict,
    launcher_args: list[str] | None = None,
    use_exec: bool = False,
) -> None:
    """Execute training with the given configuration."""
    launcher_args = launcher_args or []

    if cloud:
        _launch_cloud_training(cloud, cfg_file, launcher, kwargs, launcher_args)
    elif launcher:
        if launcher == "accelerate":
            _launch_accelerate_training(cfg_file, kwargs, launcher_args, use_exec)
        elif launcher == "torchrun":
            _launch_torchrun_training(cfg_file, kwargs, launcher_args, use_exec)
        elif launcher == "python":
            _launch_python_training(cfg_file, kwargs)
    elif launcher is None:
        # handle ray train launch
        _launch_python_training(cfg_file, kwargs)


def _launch_cloud_training(
    cloud: str,
    cfg_file: str,
    launcher: Literal["accelerate", "torchrun", "python"] | None,
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


def _launch_accelerate_training(
    cfg_file: str,
    kwargs: dict,
    launcher_args: list[str] | None = None,
    use_exec: bool = False,
) -> None:
    """Execute training via accelerate launcher."""
    launcher_args = launcher_args or []
    internal_launcher_args = []

    # Extract launcher-specific arguments from kwargs (legacy support)
    if "main_process_port" in kwargs:
        main_process_port = kwargs.pop("main_process_port")
        internal_launcher_args.extend(["--main_process_port", str(main_process_port)])

    if "num_processes" in kwargs:
        num_processes = kwargs.pop("num_processes")
        internal_launcher_args.extend(["--num_processes", str(num_processes)])

    # Combine internal args with user-provided launcher args
    all_launcher_args = internal_launcher_args + launcher_args

    base_cmd = (
        ["accelerate", "launch"] + all_launcher_args + ["-m", "axolotl.cli.train"]
    )
    if cfg_file:
        base_cmd.append(cfg_file)

    cmd = build_command(base_cmd, kwargs)
    if use_exec:
        # make sure to flush stdout and stderr before replacing the process
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe(cmd[0], cmd, os.environ)  # nosec B606
    else:
        subprocess.run(cmd, check=True)  # nosec B603


def _launch_torchrun_training(
    cfg_file: str,
    kwargs: dict,
    launcher_args: list[str] | None = None,
    use_exec: bool = False,
) -> None:
    """Execute training via torchrun launcher."""
    launcher_args = launcher_args or []

    # Add default RDZV arguments if rdzv_endpoint is set
    launcher_args = _add_default_rdzv_args(launcher_args)

    base_cmd = ["torchrun"] + launcher_args + ["-m", "axolotl.cli.train"]
    if cfg_file:
        base_cmd.append(cfg_file)

    cmd = build_command(base_cmd, kwargs)
    if use_exec:
        # make sure to flush stdout and stderr before replacing the process
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe(cmd[0], cmd, os.environ)  # nosec B606
    else:
        subprocess.run(cmd, check=True)  # nosec B603


def _launch_python_training(cfg_file: str, kwargs: dict) -> None:
    """Execute training via python launcher."""
    from axolotl.cli.train import do_cli

    do_cli(config=cfg_file, **kwargs)
