"""accelerate launcher backend."""

from axolotl.cli.launchers.base import build_command, run_command


def _launch_accelerate_training(
    cfg_file: str,
    kwargs: dict,
    launcher_args: list[str] | None = None,
    use_exec: bool = False,
    env: dict[str, str] | None = None,
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
    run_command(cmd, use_exec=use_exec, env=env)
