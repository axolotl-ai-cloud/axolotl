"""Ray launcher backend: in-cluster Ray Train driver and Ray Jobs submission.

Named ray_ to avoid shadowing the ray package.
"""

import asyncio
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml

from axolotl.cli.launchers.base import build_command
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.utils.schemas.runtime import RayLauncherConfig, RuntimeConfig

LOG = get_logger(__name__)

RAY_SUBMITTED_ENV = "AXOLOTL_RAY_SUBMITTED"
_STAGED_CONFIG_NAME = "axolotl_job_config.yaml"
_STAGED_RUNTIME_NAME = "axolotl_job_runtime.yaml"
# flat cfg kwargs that the staged runtime file carries instead of the entrypoint
_RAY_CFG_KWARGS = ("use_ray", "ray_num_workers", "resources_per_worker", "ray_run_name")


def _require_ray() -> None:
    try:
        import ray  # noqa: F401  # pylint: disable=unused-import
    except ImportError as err:
        raise click.UsageError(
            "the ray launcher requires ray; install with `pip install axolotl[ray]`"
        ) from err


def launch_ray_training(
    cfg_file: str,
    kwargs: dict,
    runtime: "RuntimeConfig | None" = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run training through Ray Train (in-process driver) or the Ray Jobs API."""
    ray_rt = runtime.ray if runtime is not None else None

    if (
        runtime is not None
        and runtime.ray is not None
        and runtime.ray.is_job_submission
    ):
        if os.environ.get(RAY_SUBMITTED_ENV) == "1":
            raise RuntimeError(
                "refusing to submit a Ray job from inside a Ray job; the staged"
                " runtime config must not contain a Jobs API address"
            )
        submit_ray_job(cfg_file, kwargs, runtime, env=env)
        return

    _require_ray()
    if env:
        os.environ.update(env)
    kwargs.setdefault("use_ray", True)
    ensure_ray_initialized(ray_rt, extra_env=env)

    if (
        ray_rt is not None
        and ray_rt.num_workers == "auto"
        and "ray_num_workers" not in kwargs
        and _peek_yaml_key(cfg_file, "ray_num_workers") is None
    ):
        resources = kwargs.get("resources_per_worker") or dict(
            ray_rt.resources_per_worker
        )
        kwargs["ray_num_workers"] = resolve_num_workers("auto", resources)

    from axolotl.cli.config import load_cfg

    parsed_cfg = load_cfg(cfg_file, **kwargs)
    parsed_cli_args = _parse_trainer_cli_args()
    fit_torchtrainer(parsed_cfg, parsed_cli_args, ray_rt)


def ensure_ray_initialized(
    ray_rt: "RayLauncherConfig | None" = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Attach to a running Ray cluster, or start a local instance as a fallback."""
    import ray

    if ray.is_initialized():
        return

    runtime_env: dict[str, Any] = {}
    env_vars: dict[str, str] = dict(extra_env or {})
    if ray_rt is not None and ray_rt.runtime_env is not None:
        dumped = ray_rt.runtime_env.model_dump(exclude_none=True)
        env_vars.update(dumped.pop("env_vars", {}))
        runtime_env.update(dumped)
    if env_vars:
        runtime_env["env_vars"] = env_vars

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if runtime_env:
        init_kwargs["runtime_env"] = runtime_env

    address = ray_rt.address if ray_rt is not None else None
    if address and address != "auto":
        ray.init(address=address, **init_kwargs)
        return

    try:
        ray.init(address="auto", **init_kwargs)
        if ray.cluster_resources().get("GPU", 0) == 0 and _local_gpu_count() > 0:
            LOG.warning(
                "attached Ray cluster reports no GPUs but this node has %d;"
                " starting a private local Ray instance instead",
                _local_gpu_count(),
            )
            ray.shutdown()
            raise ConnectionError("attached cluster has no GPUs")
    except ConnectionError:
        LOG.info("no running Ray cluster found; starting a local single-node instance")
        ray.init(**init_kwargs)


def resolve_num_workers(
    requested: "int | str", resources_per_worker: dict[str, float]
) -> int:
    """Resolve num_workers, deriving 'auto' from total cluster resources."""
    if requested != "auto":
        return int(requested)

    import ray

    cluster = ray.cluster_resources()
    counts = {
        name: int(cluster.get(name, 0) // amount)
        for name, amount in resources_per_worker.items()
        if amount
    }
    if not counts or min(counts.values()) < 1:
        raise click.UsageError(
            f"cluster resources {cluster} cannot satisfy one worker of"
            f" {resources_per_worker}; is the Ray cluster up? (`axolotl ray status`)"
        )
    binding, num = min(counts.items(), key=lambda item: item[1])
    LOG.info(
        "resolved num_workers=%d from cluster resources (binding resource %r:"
        " %s available, %s per worker)",
        num,
        binding,
        cluster.get(binding),
        resources_per_worker[binding],
    )
    return num


def fit_torchtrainer(cfg, cli_args, ray_rt: "RayLauncherConfig | None" = None) -> None:
    """Build and fit the Ray Train TorchTrainer for a parsed axolotl config."""
    _require_ray()
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from axolotl.cli.train import ray_train_func

    ensure_ray_initialized(ray_rt)

    resources = cfg.resources_per_worker
    if hasattr(resources, "to_dict"):
        resources = resources.to_dict()
    resources = dict(resources or {"GPU": 1})

    trainer = TorchTrainer(
        ray_train_func,
        train_loop_config={"cfg": cfg.to_dict(), "cli_args": cli_args},
        scaling_config=ScalingConfig(
            num_workers=cfg.ray_num_workers,
            resources_per_worker=resources,
            use_gpu=resources.get("GPU", 0) > 0,
        ),
        run_config=RunConfig(
            name=cfg.ray_run_name,
            storage_path=Path(cfg.output_dir).absolute().as_posix(),
        ),
    )
    trainer.fit()


def submit_ray_job(
    cfg_file: str,
    kwargs: dict,
    runtime: "RuntimeConfig",
    env: dict[str, str] | None = None,
) -> str:
    """Submit training to a remote Ray cluster via the Jobs API and stream logs."""
    _require_ray()
    from ray.job_submission import JobStatus, JobSubmissionClient

    import axolotl

    ray_rt = runtime.ray
    assert ray_rt is not None and ray_rt.address
    staging_dir, entrypoint, runtime_env = _prepare_job(cfg_file, kwargs, runtime, env)
    try:
        client = JobSubmissionClient(ray_rt.address)
        submission_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            metadata={"axolotl_version": axolotl.__version__},
        )
        LOG.info("submitted Ray job %s to %s", submission_id, ray_rt.address)
        if ray_rt.detach:
            LOG.info(
                "detached; follow logs with: ray job logs --follow --address %s %s",
                ray_rt.address,
                submission_id,
            )
            return submission_id
        try:
            asyncio.run(_tail_job_logs(client, submission_id))
            status = client.get_job_status(submission_id)
        except KeyboardInterrupt:
            LOG.warning(
                "stopping Ray job %s (set `ray.detach: true` to leave jobs running"
                " after Ctrl-C)",
                submission_id,
            )
            client.stop_job(submission_id)
            raise SystemExit(130) from None
        if status != JobStatus.SUCCEEDED:
            LOG.error("Ray job %s finished with status %s", submission_id, status)
            raise SystemExit(1)
        return submission_id
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


async def _tail_job_logs(client, submission_id: str) -> None:
    async for line in client.tail_job_logs(submission_id):
        print(line, end="", flush=True)


def _prepare_job(
    cfg_file: str,
    kwargs: dict,
    runtime: "RuntimeConfig",
    env: dict[str, str] | None = None,
) -> tuple[Path, str, dict]:
    """Stage config + rewritten runtime file, and assemble the job runtime_env."""
    ray_rt = runtime.ray
    assert ray_rt is not None
    staging_dir = Path(tempfile.mkdtemp(prefix="axolotl-ray-job-"))

    user_runtime_env = (
        ray_rt.runtime_env.model_dump(exclude_none=True) if ray_rt.runtime_env else {}
    )
    working_dir = user_runtime_env.pop("working_dir", None)
    excludes = user_runtime_env.pop("excludes", None)
    env_vars = user_runtime_env.pop("env_vars", {})
    if env:
        env_vars = {**env, **env_vars}

    if working_dir:
        ignore = shutil.ignore_patterns(".git", *(excludes or []))
        shutil.copytree(working_dir, staging_dir, ignore=ignore, dirs_exist_ok=True)
        _warn_large_staging(staging_dir)
    else:
        LOG.warning(
            "no ray.runtime_env.working_dir set; only the config and runtime files"
            " are uploaded — relative paths in the config (local datasets, deepspeed"
            " JSONs) must already exist on the cluster or shared storage"
        )

    shutil.copy2(cfg_file, staging_dir / _STAGED_CONFIG_NAME)
    (staging_dir / _STAGED_RUNTIME_NAME).write_text(
        yaml.safe_dump(_rewrite_runtime_for_cluster(runtime, kwargs)),
        encoding="utf-8",
    )

    job_runtime_env: dict[str, Any] = {
        **user_runtime_env,  # pip, py_modules, extra passthrough keys
        "working_dir": str(staging_dir),
        "env_vars": {**env_vars, RAY_SUBMITTED_ENV: "1"},
    }
    return staging_dir, _build_entrypoint(kwargs), job_runtime_env


def _rewrite_runtime_for_cluster(runtime: "RuntimeConfig", kwargs: dict) -> dict:
    """Runtime file as staged on the cluster: in-cluster driver mode, CLI overrides folded in."""
    data = runtime.model_dump(exclude_none=True)
    data["launcher"] = "ray"
    data.pop("torchrun", None)
    data.pop("accelerate", None)
    # runtime.env and runtime_env are applied at the job level during submission
    data.pop("env", None)

    ray_block = data.get("ray") or {}
    ray_block.pop("address", None)
    ray_block.pop("detach", None)
    ray_block.pop("runtime_env", None)
    if "ray_num_workers" in kwargs:
        ray_block["num_workers"] = kwargs["ray_num_workers"]
    if "resources_per_worker" in kwargs:
        ray_block["resources_per_worker"] = kwargs["resources_per_worker"]
    if "ray_run_name" in kwargs:
        ray_block["run_name"] = kwargs["ray_run_name"]
    data["ray"] = ray_block
    return data


def _build_entrypoint(kwargs: dict) -> str:
    forwarded = {
        key: value for key, value in kwargs.items() if key not in _RAY_CFG_KWARGS
    }
    cmd = build_command(
        ["axolotl", "train", _STAGED_CONFIG_NAME, "--runtime", _STAGED_RUNTIME_NAME],
        forwarded,
    )
    return shlex.join(cmd)


def _warn_large_staging(staging_dir: Path) -> None:
    total = sum(f.stat().st_size for f in staging_dir.rglob("*") if f.is_file())
    if total > 200 * 1024 * 1024:
        LOG.warning(
            "staged working_dir is %.0f MB; Ray's default upload limit is ~500 MB —"
            " trim it with ray.runtime_env.excludes",
            total / (1024 * 1024),
        )


def _peek_yaml_key(config_path: str, key: str):
    try:
        with open(config_path, encoding="utf-8") as fin:
            data = yaml.safe_load(fin)
        return data.get(key) if isinstance(data, dict) else None
    except (OSError, yaml.YAMLError):
        return None


def _local_gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def _parse_trainer_cli_args():
    from transformers.hf_argparser import HfArgumentParser

    from axolotl.cli.args import TrainerCliArgs

    parser = HfArgumentParser(TrainerCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    return parsed_cli_args
