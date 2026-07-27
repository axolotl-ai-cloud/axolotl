"""Pydantic schema for the axolotl runtime (environment/cluster) config file.

The runtime file is passed via `axolotl train recipe.yaml --runtime cluster.yaml`.
It declares cluster-global resources and launcher settings; topology (worker
placement, ranks) is derived, so the same file can be shipped to every node
unchanged. This module must stay importable without torch or the training
config schema — it is loaded on the CLI fast path.
"""

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

LauncherChoice = Literal["accelerate", "torchrun", "ray", "python"]
LAUNCHER_CHOICES: tuple[str, ...] = ("accelerate", "torchrun", "ray", "python")


def _stringify_dict(value):
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items()}
    return value


class RayRuntimeEnv(BaseModel):
    """Subset of Ray's ``runtime_env``; unknown keys pass through to Ray verbatim."""

    model_config = ConfigDict(extra="allow")

    env_vars: dict[str, str] | None = Field(
        default=None,
        json_schema_extra={"help": "Environment variables set on every Ray worker."},
    )
    working_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "help": "Directory uploaded to the cluster and set as each worker's cwd."
        },
    )
    pip: list[str] | str | None = Field(
        default=None,
        json_schema_extra={
            "help": "Pip requirements (list or requirements.txt path) installed on workers."
        },
    )
    py_modules: list[str] | None = None
    excludes: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "help": "Glob patterns excluded from the working_dir upload."
        },
    )

    _stringify_env = field_validator("env_vars", mode="before")(_stringify_dict)


class RayClusterConfig(BaseModel):
    """Cluster bring-up settings consumed by `axolotl ray up/down/status`."""

    model_config = ConfigDict(extra="forbid")

    hostfile: str | None = Field(
        default=None,
        json_schema_extra={
            "help": "Newline-delimited hostfile (`hostname [slots=N]`); the head node is the first entry."
        },
    )
    ssh_user: str | None = None
    ssh_key: str | None = None
    head_port: int = 6379
    dashboard_port: int = 8265


class RayLauncherConfig(BaseModel):
    """Ray launcher settings."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = Field(
        default=None,
        json_schema_extra={
            "help": "Ray address. None/'auto'/'ray://…' attaches (or starts a local"
            " cluster) and runs the driver in-process; 'http(s)://host:8265' submits"
            " the run to a remote cluster via the Ray Jobs API."
        },
    )
    num_workers: int | Literal["auto"] = Field(
        default="auto",
        json_schema_extra={
            "help": "Ray Train worker count; 'auto' derives it from total cluster resources."
        },
    )
    resources_per_worker: dict[str, float] = Field(
        default_factory=lambda: {"GPU": 1.0},
        json_schema_extra={
            "help": "Ray resources per worker, e.g. {GPU: 1, 'accelerator_type:H100': 0.001}."
        },
    )
    run_name: str | None = None
    detach: bool = Field(
        default=False,
        json_schema_extra={
            "help": "Jobs-API submissions only: return after submitting instead of tailing logs."
        },
    )
    runtime_env: RayRuntimeEnv | None = None
    cluster: RayClusterConfig | None = None

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, value):
        if isinstance(value, int) and value < 1:
            raise ValueError("`ray.num_workers` must be >= 1 or 'auto'")
        return value

    @property
    def is_job_submission(self) -> bool:
        return bool(self.address and self.address.startswith(("http://", "https://")))


class TorchrunLauncherConfig(BaseModel):
    """torchrun launcher settings.

    ``node_rank`` is intentionally absent: the runtime file is identical on every
    node. Per-node identity comes from rdzv (c10d) or `-- --node_rank N` passthrough.
    """

    model_config = ConfigDict(extra="forbid")

    # torchrun accepts elastic ranges like "1:4" for nnodes
    nnodes: int | str | None = None
    nproc_per_node: int | Literal["auto", "gpu", "cpu"] | None = None
    rdzv_backend: str | None = None
    rdzv_endpoint: str | None = None
    rdzv_id: str | None = None
    master_addr: str | None = None
    master_port: int | None = None

    @model_validator(mode="after")
    def warn_conflicting_rendezvous(self):
        if self.rdzv_endpoint and self.master_addr:
            LOG.warning(
                "Both `rdzv_endpoint` and `master_addr` are set; torchrun prefers the rendezvous endpoint"
            )
        return self


class AccelerateLauncherConfig(BaseModel):
    """accelerate launch settings.

    ``machine_rank`` is intentionally absent (same-file-on-every-node); pass
    `-- --machine_rank N` if needed.
    """

    model_config = ConfigDict(extra="forbid")

    num_processes: int | None = None
    num_machines: int | None = None
    main_process_ip: str | None = None
    main_process_port: int | None = None
    config_file: str | None = None
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] | None = None


class RuntimeConfig(BaseModel):
    """Top-level runtime file (`--runtime cluster.yaml`)."""

    model_config = ConfigDict(extra="forbid")

    launcher: LauncherChoice | None = Field(
        default=None,
        json_schema_extra={
            "help": "Launcher to use; may be omitted when exactly one launcher block is present."
        },
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        json_schema_extra={
            "help": "Environment variables applied to the launcher process (any launcher);"
            " merged into ray runtime_env.env_vars for the ray launcher."
        },
    )
    ray: RayLauncherConfig | None = None
    torchrun: TorchrunLauncherConfig | None = None
    accelerate: AccelerateLauncherConfig | None = None

    _stringify_env = field_validator("env", mode="before")(_stringify_dict)

    @model_validator(mode="after")
    def warn_ignored_blocks(self):
        if self.launcher:
            ignored = [
                name
                for name in ("ray", "torchrun", "accelerate")
                if name != self.launcher and getattr(self, name) is not None
            ]
            if ignored:
                LOG.warning(
                    "runtime file declares launcher=%s; ignoring blocks: %s",
                    self.launcher,
                    ignored,
                )
        return self

    def resolve_launcher_choice(self) -> LauncherChoice | None:
        """Explicit `launcher:`, else derived from the single block present."""
        if self.launcher:
            return self.launcher
        present = [
            name
            for name in ("ray", "torchrun", "accelerate")
            if getattr(self, name) is not None
        ]
        if len(present) == 1:
            return present[0]  # type: ignore[return-value]
        if len(present) > 1:
            raise ValueError(
                f"runtime file has multiple launcher blocks ({present}) and no"
                " `launcher:` key; set `launcher:` or pass --launcher"
            )
        return None

    @classmethod
    def from_file(cls, path: str) -> "RuntimeConfig":
        with open(path, encoding="utf-8") as fin:
            data = yaml.safe_load(fin) or {}
        if not isinstance(data, dict):
            raise ValueError(f"runtime config {path} must be a YAML mapping")
        return cls.model_validate(data)
