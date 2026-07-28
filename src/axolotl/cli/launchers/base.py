"""Shared helpers for axolotl launcher backends."""

import os
import subprocess  # nosec
import sys
from typing import Any

import yaml

LEGACY_LAUNCHER_KWARGS = ("num_processes", "main_process_port")


def _peek_yaml_key(config_path: str, key: str) -> Any:
    """Cheap top-level YAML lookup; None on missing file, bad YAML, or non-mapping."""
    try:
        with open(config_path, encoding="utf-8") as fin:
            data = yaml.safe_load(fin)
        return data.get(key) if isinstance(data, dict) else None
    except (OSError, yaml.YAMLError):
        return None


def pop_legacy_launcher_kwargs(kwargs: dict) -> dict:
    """Remove legacy launcher kwargs so they never leak into config overrides."""
    return {key: kwargs.pop(key) for key in LEGACY_LAUNCHER_KWARGS if key in kwargs}


def _flag_name(arg: str) -> str | None:
    if not arg.startswith("--"):
        return None
    return arg[2:].split("=", 1)[0].replace("-", "_")


def merge_launcher_args(derived: list[str], passthrough: list[str]) -> list[str]:
    """
    Merge runtime-file-derived launcher args with explicit `--` passthrough args.

    Passthrough wins: any derived flag whose normalized name also appears in the
    passthrough list is dropped, and passthrough args come last. `derived` must
    use single-token `--name=value` form so flags can be dropped independently.
    """
    overridden = {name for arg in passthrough if (name := _flag_name(arg))}
    kept = [
        arg for arg in derived if (name := _flag_name(arg)) and name not in overridden
    ]
    return kept + list(passthrough)


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
    if use_exec:
        # make sure to flush stdout and stderr before replacing the process
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe(cmd[0], cmd, {**os.environ, **env} if env else os.environ)  # nosec B606
    else:
        run_kwargs: dict = {"check": True}
        if env:
            run_kwargs["env"] = {**os.environ, **env}
        subprocess.run(cmd, **run_kwargs)  # nosec B603
