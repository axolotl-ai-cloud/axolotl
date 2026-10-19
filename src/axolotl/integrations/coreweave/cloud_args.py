"""CoreWeave job submission configuration (SUNK or CKS), independent of training imports."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from axolotl.utils.schemas.cloud import CloudImageConfig

AbsolutePath = Annotated[str, StringConstraints(pattern=r"^/[^\s,:]*$")]


class ObjectStorageIdentity(BaseModel):
    """CKS workload identity for AI Object Storage: the pod exchanges a projected
    service-account token for temporary S3 credentials, no static keys."""

    model_config = ConfigDict(extra="forbid")

    org_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    region: str = Field(pattern=r"^[A-Za-z0-9-]+$")
    endpoint: str = "http://cwlota.com"
    audience: str = "https://coreweave.com/iam"


class CoreWeaveCloudConfig(CloudImageConfig):
    """Submit an Axolotl container to a SUNK Slurm cluster or a CKS Kubernetes cluster."""

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(default=None, pattern=r"^[A-Za-z0-9/][^\s\x00]*$")

    provider: str = "coreweave"
    backend: Literal["sunk", "cks"] = "sunk"
    ssh_host: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.@-]*$"
    )
    shared_dir: str | None = Field(default=None, pattern=r"^/[^\s,:]*$")
    partition: str | None = Field(default=None, min_length=1)
    account: str | None = None
    qos: str | None = None
    nodes: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=8, ge=1)
    cpus_per_task: int = Field(default=64, ge=1)
    memory: str = "0"
    time_limit: str = "24:00:00"
    exclusive: bool = True
    master_port: int = Field(default=29500, ge=1024, le=65535)
    mounts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    env_file: str | None = None
    dry_run: bool = False

    # --- CKS (Kubernetes) backend ---------------------------------------------------------
    kubeconfig: str | None = None
    context: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.@:/-]+$")
    namespace: str | None = Field(
        default=None, pattern=r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$"
    )
    job_name: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$",
        max_length=52,
        description=(
            "Name for the Job, Service and ConfigMap; must not already exist. Capped "
            "at 52 characters because Job pod names (<job_name>-<index>-<suffix>) "
            "and Service names are limited to 63."
        ),
    )
    # ReadWriteMany volume for datasets, caches and outputs, mounted at pvc_mount
    pvc: str | None = Field(default=None, pattern=r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
    pvc_mount: str = Field(default="/data", pattern=r"^/[^\s,:]*$")
    cpu: str = Field(default="32", pattern=r"^[0-9]+m?$")
    memory_request: str = Field(default="256Gi", pattern=r"^[0-9]+[EPTGMK]i?$")
    shm_size: str = Field(default="64Gi", pattern=r"^[0-9]+[EPTGMK]i?$")
    scratch_size: str | None = Field(default=None, pattern=r"^[0-9]+[EPTGMK]i?$")
    # node-local directories mounted read-only, host path -> container path (e.g. model weights staged on NVMe)
    host_paths: dict[AbsolutePath, AbsolutePath] = Field(default_factory=dict)
    # request rdma/ib and set CoreWeave's InfiniBand NCCL recipe
    rdma: bool = True
    node_selector: dict[str, str] = Field(default_factory=dict)
    # env var -> "secret-name/key" (e.g. HF_TOKEN: hf-token/token)
    secret_env: dict[str, str] = Field(default_factory=dict)
    object_storage: ObjectStorageIdentity | None = None
    # shell lines run in the container before the launcher (extra pip installs, ...)
    setup_commands: list[str] = Field(default_factory=list)
    rdzv_port: int = Field(default=29400, ge=1024, le=65535)

    @model_validator(mode="after")
    def validate_job(self):
        import re
        from pathlib import PurePosixPath

        if self.backend == "cks":
            return self._validate_cks()
        if self.image is None and self.image_build is None:
            raise ValueError("SUNK requires image or image_build")
        if self.image_build and not self.image_build.tag:
            raise ValueError("SUNK image_build requires a registry tag")
        if self.shared_dir is None or self.partition is None:
            raise ValueError("SUNK requires shared_dir and partition")
        if ".." in PurePosixPath(self.shared_dir).parts:
            raise ValueError("shared_dir cannot contain '..'")
        for name in ("partition", "account", "qos", "memory", "time_limit"):
            value = getattr(self, name)
            if value is not None and not re.fullmatch(r"[A-Za-z0-9_:.+-]+", value):
                raise ValueError(f"Invalid Slurm {name}")
        for path in self.mounts:
            if (
                not re.fullmatch(r"/[^\s,:]+", path)
                or ".." in PurePosixPath(path).parts
            ):
                raise ValueError(
                    "mounts must be absolute shared paths without commas or colons"
                )
        for key in self.env:
            if (
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                or key.startswith("SLURM_")
                or key in {"MASTER_ADDR", "MASTER_PORT"}
            ):
                raise ValueError(
                    f"Invalid or launcher-owned environment variable: {key}"
                )
        if self.env_file is not None:
            if not re.fullmatch(r"/[^\s,:]+", self.env_file):
                raise ValueError(
                    "env_file must be an absolute path visible on allocated SUNK nodes"
                )
        return self

    def _validate_cks(self):
        import re
        from pathlib import PurePosixPath

        if self.image is None and self.image_build is None:
            raise ValueError("CKS requires image or image_build")
        if self.image_build and not self.image_build.tag:
            raise ValueError("CKS image_build requires a registry tag")
        if not self.namespace:
            raise ValueError("CKS requires namespace")
        if not self.pvc:
            raise ValueError(
                "CKS requires pvc: a ReadWriteMany volume for datasets, caches and outputs"
            )
        if ".." in PurePosixPath(self.pvc_mount).parts:
            raise ValueError("pvc_mount cannot contain '..'")
        for key in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in {
                "MASTER_ADDR",
                "MASTER_PORT",
                "AXOLOTL_NNODES",
                "AXOLOTL_NODE_RANK",
            }:
                raise ValueError(
                    f"Invalid or launcher-owned environment variable: {key}"
                )
        for key, ref in self.secret_env.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not re.fullmatch(
                r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?/[A-Za-z0-9._-]+", ref
            ):
                raise ValueError(
                    f"secret_env entries are ENV: secret-name/key, got {key}: {ref}"
                )
        if self.env_file is not None:
            raise ValueError("env_file is a SUNK option; CKS uses secret_env")
        return self
