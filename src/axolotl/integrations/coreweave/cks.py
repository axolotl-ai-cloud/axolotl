"""Submit an Axolotl job to a CoreWeave Kubernetes Service (CKS) cluster.

One indexed ``Job`` with ``nodes`` pods (one per node, ``gpus_per_node`` GPUs each), a headless
``Service`` that gives pod 0 a stable rendezvous name, and a ``ConfigMap`` carrying the training
YAML, the launch spec and the runner. Pods request ``rdma/ib`` and carry CoreWeave's InfiniBand
NCCL recipe; a ReadWriteMany PVC holds datasets, caches and outputs; AI Object Storage is reached
through workload identity (a projected service-account token), never static keys.
"""

import json
import shlex
import subprocess  # nosec B404
import uuid
from pathlib import Path, PurePosixPath

import yaml

from axolotl.utils.logging import get_logger

from .cloud_args import CoreWeaveCloudConfig

LOG = get_logger(__name__)

JOB_LABEL = "axolotl.ai/job"
INFINIBAND_ENV = {
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_IB_HCA": "ibp",
    "UCX_NET_DEVICES": "eth0",
    "UCX_TLS": "tcp",
}
CONFIG_MOUNT = "/axolotl-job"
TOKEN_MOUNT = "/var/run/caios"  # nosec B105


def _kubectl(cfg: CoreWeaveCloudConfig) -> list[str]:
    command = ["kubectl"]
    if cfg.kubeconfig:
        command += ["--kubeconfig", str(Path(cfg.kubeconfig).expanduser())]
    if cfg.context:
        command += ["--context", cfg.context]
    return command + ["-n", str(cfg.namespace)]


def _entrypoint(cfg: CoreWeaveCloudConfig, job_name: str) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    rdzv = f"{job_name}-0.{job_name}.{cfg.namespace}.svc.cluster.local"
    lines += [
        f"export MASTER_ADDR={rdzv}",
        f"export MASTER_PORT={cfg.rdzv_port}",
        f"export AXOLOTL_NNODES={cfg.nodes}",
        'export AXOLOTL_NODE_RANK="${JOB_COMPLETION_INDEX:-0}"',
        f"export HF_HOME={shlex.quote(cfg.pvc_mount + '/hf')} "
        f"HF_HUB_CACHE={shlex.quote(cfg.pvc_mount + '/hf/hub')} "
        f"HF_DATASETS_CACHE={shlex.quote(cfg.pvc_mount + '/hf/datasets')}",
        "export TRITON_CACHE_DIR=/scratch/triton TORCHINDUCTOR_CACHE_DIR=/scratch/inductor",
        "mkdir -p /scratch/triton /scratch/inductor",
    ]
    if cfg.rdma:
        lines += [f"export {k}={v}" for k, v in INFINIBAND_ENV.items()]
    if cfg.object_storage:
        lines += [
            "export AWS_CONTAINER_CREDENTIALS_FULL_URI="
            f"https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/{cfg.object_storage.org_id}",
            f"export AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE={TOKEN_MOUNT}/token",
            f"export AWS_ENDPOINT_URL_S3={cfg.object_storage.endpoint}",
            f"export AWS_REGION={cfg.object_storage.region}",
            "export AWS_REQUEST_CHECKSUM_CALCULATION=when_required",
        ]
    lines += list(cfg.setup_commands)
    lines.append(f"exec python3 {CONFIG_MOUNT}/runner.py")
    return "\n".join(lines) + "\n"


def render_manifests(
    cfg: CoreWeaveCloudConfig,
    training: dict,
    launcher: str,
    launcher_args: list[str],
    image: str,
    job_name: str,
) -> list[dict]:
    """Kubernetes objects for one training job, in apply order."""
    labels = {JOB_LABEL: job_name}
    files = {
        "train.yaml": yaml.safe_dump(training),
        "launch.json": json.dumps(
            {
                "launcher": launcher,
                "launcher_args": launcher_args,
                "gpus_per_node": cfg.gpus_per_node,
            }
        ),
        "runner.py": Path(__file__)
        .with_name("cloud_runner.py")
        .read_text(encoding="utf-8"),
        "entrypoint.sh": _entrypoint(cfg, job_name),
    }
    env: list[dict] = [{"name": key, "value": value} for key, value in cfg.env.items()]
    for key, ref in cfg.secret_env.items():
        name, secret_key = ref.split("/", 1)
        env.append(
            {
                "name": key,
                "valueFrom": {"secretKeyRef": {"name": name, "key": secret_key}},
            }
        )
    resources = {
        "cpu": cfg.cpu,
        "memory": cfg.memory_request,
        "nvidia.com/gpu": str(cfg.gpus_per_node),
    }
    if cfg.rdma:
        resources["rdma/ib"] = "1"
    volumes = [
        {"name": "data", "persistentVolumeClaim": {"claimName": cfg.pvc}},
        {
            "name": "job",
            "configMap": {"name": job_name, "defaultMode": 0o755},
        },
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": cfg.shm_size}},
        {
            "name": "scratch",
            "emptyDir": {"sizeLimit": cfg.scratch_size} if cfg.scratch_size else {},
        },
    ]
    mounts = [
        {"name": "data", "mountPath": cfg.pvc_mount},
        {"name": "job", "mountPath": CONFIG_MOUNT, "readOnly": True},
        {"name": "shm", "mountPath": "/dev/shm"},  # nosec B108
        {"name": "scratch", "mountPath": "/scratch"},
    ]
    for i, (host_path, mount_path) in enumerate(sorted(cfg.host_paths.items())):
        volumes.append(
            {"name": f"host-{i}", "hostPath": {"path": host_path, "type": "Directory"}}
        )
        mounts.append({"name": f"host-{i}", "mountPath": mount_path, "readOnly": True})
    if cfg.object_storage:
        volumes.append(
            {
                "name": "caios-token",
                "projected": {
                    "sources": [
                        {
                            "serviceAccountToken": {
                                "path": "token",
                                "audience": cfg.object_storage.audience,
                                "expirationSeconds": 3600,
                            }
                        }
                    ]
                },
            }
        )
        mounts.append(
            {"name": "caios-token", "mountPath": TOKEN_MOUNT, "readOnly": True}
        )
    pod_spec: dict = {
        "restartPolicy": "Never",
        "subdomain": job_name,
        "containers": [
            {
                "name": "axolotl",
                "image": image,
                "command": ["bash", f"{CONFIG_MOUNT}/entrypoint.sh"],
                "env": env,
                "resources": {"requests": dict(resources), "limits": dict(resources)},
                "volumeMounts": mounts,
            }
        ],
        "volumes": volumes,
    }
    if cfg.node_selector:
        pod_spec["nodeSelector"] = dict(cfg.node_selector)
    if cfg.nodes > 1:
        pod_spec["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "topologyKey": "kubernetes.io/hostname",
                        "labelSelector": {"matchLabels": labels},
                    }
                ]
            }
        }
    return [
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": job_name,
                "namespace": cfg.namespace,
                "labels": labels,
            },
            "spec": {
                "clusterIP": "None",
                "selector": labels,
                "ports": [{"name": "rdzv", "port": cfg.rdzv_port}],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": job_name,
                "namespace": cfg.namespace,
                "labels": labels,
            },
            "data": files,
        },
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": cfg.namespace,
                "labels": labels,
            },
            "spec": {
                "completionMode": "Indexed",
                "completions": cfg.nodes,
                "parallelism": cfg.nodes,
                "backoffLimit": 0,
                "template": {"metadata": {"labels": labels}, "spec": pod_spec},
            },
        },
    ]


def place_output_dir(cfg: CoreWeaveCloudConfig, training: dict, job_name: str) -> dict:
    """Keep outputs on the shared volume: relative paths land under the job's own directory."""
    output = PurePosixPath(training.get("output_dir") or "outputs")
    if ".." in output.parts:
        raise ValueError("output_dir cannot contain '..'")
    if output.is_absolute():
        if not output.is_relative_to(cfg.pvc_mount):
            raise ValueError(
                f"An absolute output_dir must be under pvc_mount ({cfg.pvc_mount})"
            )
        training["output_dir"] = str(output)
    else:
        training["output_dir"] = str(
            PurePosixPath(cfg.pvc_mount) / "axolotl-jobs" / job_name / output
        )
    return training


def _run_kubectl(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(  # nosec B603
            command, text=True, capture_output=True, check=False, **kwargs
        )
    except OSError as exc:
        raise RuntimeError(
            f"kubectl not found or not executable ({exc}); install kubectl and make "
            "sure it is on PATH"
        ) from exc


def _require_unused_job_name(cfg: CoreWeaveCloudConfig, job_name: str) -> None:
    # kubectl apply would update the existing Job in place instead of starting a run.
    command = _kubectl(cfg) + [
        "get",
        "job",
        job_name,
        "--ignore-not-found",
        "-o",
        "name",
    ]
    result = _run_kubectl(command)
    if result.returncode != 0:
        raise RuntimeError(
            f"kubectl get job failed ({result.returncode}): {result.stderr.strip()}"
        )
    if result.stdout.strip():
        raise RuntimeError(
            f"CKS job {job_name} already exists in {cfg.namespace}. Choose another "
            "job_name, omit it for a generated one, or delete the old job: "
            + shlex.join(_kubectl(cfg) + ["delete", "job,service,configmap", job_name])
        )


def submit(cfg: CoreWeaveCloudConfig, manifests: list[dict], job_name: str) -> dict:
    documents = yaml.safe_dump_all(manifests, sort_keys=False)
    if cfg.dry_run:
        print(documents)
        return {
            "job_name": job_name,
            "namespace": cfg.namespace,
            "manifests": manifests,
        }
    if cfg.job_name:
        _require_unused_job_name(cfg, job_name)
    cleanup = shlex.join(_kubectl(cfg) + ["delete", "job,service,configmap", job_name])
    result = _run_kubectl(_kubectl(cfg) + ["apply", "-f", "-"], input=documents)
    if result.returncode != 0:
        raise RuntimeError(
            f"kubectl apply failed ({result.returncode}): {result.stderr.strip()}. "
            f"Some objects may already exist; remove them with: {cleanup}"
        )
    follow = shlex.join(_kubectl(cfg) + ["logs", "-f", f"job/{job_name}"])
    LOG.info(
        "CKS job %s submitted to %s (%d node(s) x %d GPU(s)). Logs: %s. Delete: %s",
        job_name,
        cfg.namespace,
        cfg.nodes,
        cfg.gpus_per_node,
        follow,
        cleanup,
    )
    return {
        "job_name": job_name,
        "namespace": cfg.namespace,
        "manifests": manifests,
        "logs_command": follow,
    }


def new_job_name(cfg: CoreWeaveCloudConfig) -> str:
    return cfg.job_name or f"axolotl-{uuid.uuid4().hex[:10]}"
