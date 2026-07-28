"""torchrun launcher backend."""

from axolotl.cli.launchers.base import _flag_name, build_command, run_command


def _add_default_rdzv_args(launcher_args: list[str]) -> list[str]:
    """
    Add default RDZV arguments if rdzv_endpoint is set but rdzv_backend/rdzv_id are missing.

    Args:
        launcher_args: List of launcher arguments

    Returns:
        Updated launcher args with defaults added if needed
    """
    args = launcher_args.copy()
    flags = {name for arg in args if (name := _flag_name(arg)) is not None}

    if "rdzv_endpoint" in flags:
        if "rdzv_backend" not in flags:
            args.extend(["--rdzv_backend", "c10d"])

        if "rdzv_id" not in flags:
            import uuid

            args.extend(["--rdzv_id", str(uuid.uuid4())[:8]])

    return args


def _launch_torchrun_training(
    cfg_file: str,
    kwargs: dict,
    launcher_args: list[str] | None = None,
    use_exec: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """Execute training via torchrun launcher."""
    launcher_args = launcher_args or []

    # Add default RDZV arguments if rdzv_endpoint is set
    launcher_args = _add_default_rdzv_args(launcher_args)

    base_cmd = ["torchrun"] + launcher_args + ["-m", "axolotl.cli.train"]
    if cfg_file:
        base_cmd.append(cfg_file)

    cmd = build_command(base_cmd, kwargs)
    run_command(cmd, use_exec=use_exec, env=env)
