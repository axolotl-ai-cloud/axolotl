"""Utilities for axolotl train CLI command."""

import tempfile
from pathlib import Path
from typing import Iterator

import yaml

# Launcher internals live in axolotl.cli.launchers; re-exported here for back-compat.
from axolotl.cli.launchers import (  # noqa: F401
    build_command,
    launch_training,
    _launch_accelerate_training,
    _launch_cloud_training,
    _launch_python_training,
    _launch_torchrun_training,
)
from axolotl.cli.launchers.torchrun import _add_default_rdzv_args  # noqa: F401
from axolotl.cli.utils.sweeps import generate_sweep_configs


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
