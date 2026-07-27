"""Shared helpers for axolotl launcher backends."""

import os
import subprocess  # nosec
import sys
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


def run_command(
    cmd: list[str],
    use_exec: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """
    Run a launcher command, either replacing the current process or as a subprocess.

    Args:
        cmd: Full command to run.
        use_exec: Replace the current process via `os.execvpe` (single-run path);
            otherwise run as a subprocess (sweeps need the loop to continue).
        env: Extra environment variables layered over `os.environ`.
    """
    full_env = {**os.environ, **env} if env else None
    if use_exec:
        # make sure to flush stdout and stderr before replacing the process
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe(cmd[0], cmd, full_env if full_env is not None else os.environ)  # nosec B606
    else:
        subprocess.run(cmd, check=True, env=full_env)  # nosec B603
