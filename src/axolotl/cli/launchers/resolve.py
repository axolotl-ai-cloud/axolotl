"""Launcher resolution: CLI flags > runtime file > legacy training-YAML keys > defaults."""

import dataclasses

import click
import yaml

from axolotl.cli.launchers.base import merge_launcher_args, pop_legacy_launcher_kwargs
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.runtime import RuntimeConfig

LOG = get_logger(__name__)

_TORCHRUN_FIELDS = (
    "nnodes",
    "nproc_per_node",
    "rdzv_backend",
    "rdzv_endpoint",
    "rdzv_id",
    "master_addr",
    "master_port",
)
_ACCELERATE_FIELDS = (
    "num_processes",
    "num_machines",
    "main_process_ip",
    "main_process_port",
    "config_file",
    "mixed_precision",
)
_LEGACY_KWARG_TRANSLATION = {
    "accelerate": {"num_processes": "num_processes", "main_process_port": "main_process_port"},
    "torchrun": {"num_processes": "nproc_per_node", "main_process_port": "master_port"},
}


@dataclasses.dataclass
class ResolvedLaunch:
    """Outcome of launcher resolution for a single train/evaluate invocation."""

    launcher: str
    launcher_args: list[str]
    env: dict[str, str]
    runtime: RuntimeConfig | None


def peek_legacy_use_ray(config_path: str) -> bool:
    """Cheap top-level peek at the training YAML; no load_cfg/pydantic."""
    try:
        with open(config_path, encoding="utf-8") as fin:
            data = yaml.safe_load(fin)
        return isinstance(data, dict) and bool(data.get("use_ray"))
    except (OSError, yaml.YAMLError):
        return False


def _runtime_launcher_args(runtime: RuntimeConfig | None, launcher: str) -> list[str]:
    if runtime is None or launcher not in ("torchrun", "accelerate"):
        return []
    block = getattr(runtime, launcher)
    if block is None:
        return []
    fields = _TORCHRUN_FIELDS if launcher == "torchrun" else _ACCELERATE_FIELDS
    return [
        f"--{name}={getattr(block, name)}"
        for name in fields
        if getattr(block, name) is not None
    ]


def _translate_legacy_kwargs(legacy: dict, launcher: str) -> list[str]:
    """Translate popped legacy kwargs (num_processes/main_process_port) into launcher flags."""
    if not legacy:
        return []
    translation = _LEGACY_KWARG_TRANSLATION.get(launcher)
    if translation is None:
        LOG.warning(
            "%s are accelerate/torchrun launcher options; ignored for launcher=%s"
            " (use a --runtime file instead)",
            "/".join(f"--{key}" for key in legacy),
            launcher,
        )
        return []
    args = []
    for key, value in legacy.items():
        flag = translation[key]
        if flag != key:
            LOG.info("translating --%s to --%s for launcher=%s", key, flag, launcher)
        args.append(f"--{flag}={value}")
    return args


def _coerce_resources_per_worker(kwargs: dict) -> None:
    """`--resources-per-worker` arrives as a raw string from the generated CLI options."""
    value = kwargs.get("resources_per_worker")
    if isinstance(value, str):
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = None
        if not isinstance(parsed, dict):
            raise click.BadParameter(
                "resources_per_worker must be a mapping, e.g. '{GPU: 1}'",
                param_hint="--resources-per-worker",
            )
        kwargs["resources_per_worker"] = parsed


def _apply_ray_runtime_kwargs(runtime: RuntimeConfig | None, kwargs: dict) -> None:
    """Populate the legacy flat ray cfg fields so the worker-side code paths just work.

    `setdefault` keeps explicit CLI flags (already present in kwargs) ahead of the
    runtime file.
    """
    kwargs["use_ray"] = True
    ray_cfg = runtime.ray if runtime else None
    if ray_cfg is not None:
        if isinstance(ray_cfg.num_workers, int):
            kwargs.setdefault("ray_num_workers", ray_cfg.num_workers)
        # num_workers == "auto" is resolved against the live cluster at launch time
        kwargs.setdefault("resources_per_worker", dict(ray_cfg.resources_per_worker))
        if ray_cfg.run_name:
            kwargs.setdefault("ray_run_name", ray_cfg.run_name)
    _coerce_resources_per_worker(kwargs)


def resolve_launch(
    config: str,
    runtime_path: str | None,
    cli_launcher: str | None,
    kwargs: dict,
    passthrough_args: list[str],
    supports_ray: bool = True,
) -> ResolvedLaunch:
    """
    Resolve the effective launcher and its arguments for a CLI invocation.

    Mutates `kwargs`: pops legacy launcher keys and, on the ray path, injects the
    flat ray cfg fields. Precedence: explicit CLI flags > runtime file > legacy
    training-YAML `use_ray` > default accelerate.
    """
    runtime = RuntimeConfig.from_file(runtime_path) if runtime_path else None
    # tri-state: absent (flag not passed; filter_none_kwargs strips it) / True / False
    use_ray_flag = kwargs.get("use_ray")

    launcher = cli_launcher
    if use_ray_flag:
        if launcher and launcher != "ray":
            raise click.UsageError(
                f"--use-ray conflicts with --launcher {launcher}; use --launcher ray"
            )
        LOG.warning(
            "--use-ray is deprecated; use `--launcher ray` or `launcher: ray` in a --runtime file"
        )
        launcher = "ray"

    if launcher is None and runtime is not None:
        try:
            launcher = runtime.resolve_launcher_choice()
        except ValueError as err:
            raise click.UsageError(str(err)) from err

    if (
        supports_ray
        and launcher is None
        and use_ray_flag is not False
        and peek_legacy_use_ray(config)
    ):
        LOG.warning(
            "`use_ray: true` found in the training config; prefer `launcher: ray`"
            " in a --runtime file"
        )
        launcher = "ray"

    launcher = launcher or "accelerate"

    if launcher == "ray" and not supports_ray:
        raise click.UsageError("the ray launcher is only supported for `axolotl train`")

    legacy = pop_legacy_launcher_kwargs(kwargs)
    derived = _runtime_launcher_args(runtime, launcher)
    derived += _translate_legacy_kwargs(legacy, launcher)
    launcher_args = merge_launcher_args(derived, list(passthrough_args))

    if launcher == "ray":
        if launcher_args:
            raise click.UsageError(
                "the ray launcher takes no `--` passthrough args; use the ray: block"
                " of a --runtime file"
            )
        _apply_ray_runtime_kwargs(runtime, kwargs)
    elif peek_legacy_use_ray(config) and supports_ray:
        # neutralize the YAML key so per-rank processes don't each start a ray driver
        LOG.warning(
            "`use_ray: true` in the training config is overridden by launcher=%s",
            launcher,
        )
        kwargs["use_ray"] = False

    return ResolvedLaunch(
        launcher=launcher,
        launcher_args=launcher_args,
        env=dict(runtime.env) if runtime else {},
        runtime=runtime,
    )
