"""Launch complete Axolotl training jobs on CoreWeave SUNK."""

import io
import json
import logging
import re
import shlex
import subprocess  # nosec B404
import tarfile
import uuid
from pathlib import Path, PurePosixPath

import yaml

from axolotl.cli.cloud.base import CloudLauncher
from axolotl.cli.cloud.images import build_and_push_image

from .cloud_args import CoreWeaveCloudConfig

LOG = logging.getLogger(__name__)


def container_image(image):
    """Translate Docker registry tags to Enroot's registry separator."""
    if image.startswith(("/", "docker://")) or "#" in image:
        return image
    registry, separator, name = image.partition("/")
    if separator and ("." in registry or ":" in registry or registry == "localhost"):
        return f"docker://{registry}#{name}"
    return image


class CoreWeaveCloud(CloudLauncher):
    """Stage a job on shared storage and submit it once with sbatch."""

    def __init__(self, config, *, config_dir=None):
        self.image_config = CoreWeaveCloudConfig.from_config(
            config, config_dir=config_dir
        )
        self.config = self.image_config.model_dump()

    @classmethod
    def from_config(cls, config, *, config_dir=None):
        return cls(config, config_dir=config_dir)

    def _run(self, command, **kwargs):
        if self.image_config.ssh_host:
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                self.image_config.ssh_host,
                shlex.join(command),
            ]
        return subprocess.run(command, check=True, capture_output=True, **kwargs)  # nosec B603

    def _script(self, directory, image):
        cfg = self.image_config
        mounts = list(dict.fromkeys([cfg.shared_dir, *cfg.mounts]))
        lines = ["#!/bin/bash", "set -euo pipefail"]
        lines += [
            f"export {key}={shlex.quote(value)}" for key, value in cfg.env.items()
        ]
        lines += [
            'mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")',
            'export MASTER_ADDR="${nodes[0]}"',
            f"export MASTER_PORT={cfg.master_port}",
        ]
        container_mounts = [f"{path}:{path}" for path in mounts]
        if cfg.env_file:
            container_mounts.append(f"{cfg.env_file}:{cfg.env_file}:ro")
        bootstrap = ["set -euo pipefail", "readonly MASTER_ADDR MASTER_PORT"]
        if cfg.env_file:
            bootstrap += ["set -a", f"source {shlex.quote(cfg.env_file)}", "set +a"]
        bootstrap += [
            f"export {key}={shlex.quote(value)}" for key, value in cfg.env.items()
        ]
        bootstrap.append("exec python3 " + shlex.quote(str(directory / "runner.py")))
        command = [
            "srun",
            "--kill-on-bad-exit=1",
            "--ntasks-per-node=1",
            f"--ntasks={cfg.nodes}",
            f"--nodes={cfg.nodes}",
            f"--container-image={container_image(image)}",
            "--container-mounts=" + ",".join(container_mounts),
            "--container-env="
            + ",".join(sorted({"MASTER_ADDR", "MASTER_PORT", *cfg.env})),
            f"--container-workdir={directory}",
            "--no-container-mount-home",
            "/bin/bash",
            "-c",
            "\n".join(bootstrap),
        ]
        lines.append("exec " + shlex.join(command))
        return "\n".join(lines) + "\n"

    def train(
        self,
        config_yaml,
        launcher="accelerate",
        launcher_args=None,
        local_dirs=None,
        **kwargs,
    ):
        cfg = self.image_config
        training = yaml.safe_load(config_yaml)
        if not isinstance(training, dict):
            raise ValueError("The training config must be a YAML mapping")
        training.update(kwargs)
        backend = cfg.backend.upper()
        if local_dirs:
            raise ValueError(
                f"CoreWeave {backend} needs shared storage; local directory uploads "
                "are not supported"
            )
        if training.get("use_ray"):
            raise ValueError(
                f"CoreWeave {backend} launches Accelerate/torchrun directly; set "
                "use_ray: false"
            )
        if launcher not in {"accelerate", "torchrun", "python"}:
            raise ValueError(f"Unsupported launcher: {launcher}")
        if launcher == "python" and (cfg.nodes != 1 or cfg.gpus_per_node != 1):
            raise ValueError("The python launcher requires one node and one GPU")
        launcher_args = launcher_args or []
        if cfg.backend == "cks":
            return self._train_cks(training, launcher, launcher_args)
        reserved = {
            "nnodes",
            "nproc_per_node",
            "node_rank",
            "master_addr",
            "master_port",
            "num_machines",
            "num_processes",
            "machine_rank",
            "main_process_ip",
            "main_process_port",
            "rdzv_backend",
            "rdzv_endpoint",
            "rdzv_id",
            "standalone",
        }
        if any(
            not isinstance(arg, str)
            or arg.split("=", 1)[0].lstrip("-").replace("-", "_") in reserved
            for arg in launcher_args
        ):
            raise ValueError(
                "SUNK controls distributed topology; remove conflicting launcher arguments"
            )
        directory = PurePosixPath(cfg.shared_dir) / f"axolotl-{uuid.uuid4().hex}"
        output = training.get("output_dir") or "outputs"
        output_path = PurePosixPath(output)
        if ".." in output_path.parts:
            raise ValueError("output_dir cannot contain '..'")
        if output_path.is_absolute() and not any(
            output_path.is_relative_to(path) for path in [cfg.shared_dir, *cfg.mounts]
        ):
            raise ValueError("An absolute output_dir must be on a shared mount")
        training["output_dir"] = str(
            output_path if output_path.is_absolute() else directory / output_path
        )
        image = cfg.image or cfg.image_build.tag
        if cfg.image_build and not cfg.dry_run:
            image = build_and_push_image(cfg.image_build)
        files = {
            "train.yaml": yaml.safe_dump(training).encode(),
            "launch.json": json.dumps(
                {
                    "launcher": launcher,
                    "launcher_args": launcher_args,
                    "gpus_per_node": cfg.gpus_per_node,
                }
            ).encode(),
            "runner.py": Path(__file__).with_name("cloud_runner.py").read_bytes(),
            "job.sh": self._script(directory, image).encode(),
        }
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={directory.name}",
            f"--chdir={directory}",
            f"--output={directory}/slurm-%j.out",
            f"--partition={cfg.partition}",
            f"--nodes={cfg.nodes}",
            "--ntasks-per-node=1",
            f"--gpus-per-node={cfg.gpus_per_node}",
            f"--cpus-per-task={cfg.cpus_per_task}",
            f"--mem={cfg.memory}",
            f"--time={cfg.time_limit}",
        ]
        for option in ("account", "qos"):
            if value := getattr(cfg, option):
                command.append(f"--{option}={value}")
        if cfg.exclusive:
            command.append("--exclusive")
        command.append(str(directory / "job.sh"))
        if cfg.dry_run:
            LOG.info(
                "SUNK preview (no files staged or job submitted): %s",
                shlex.join(command),
            )
            return {"directory": str(directory), "command": command, "files": files}
        if cfg.ssh_host:
            self._run(["mkdir", "-p", "--", cfg.shared_dir])
            self._run(["mkdir", "-m", "700", "--", str(directory)])
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as tar:
                for name, contents in files.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(contents), 0o600
                    tar.addfile(info, io.BytesIO(contents))
            self._run(
                ["tar", "-xf", "-", "-C", str(directory)], input=archive.getvalue()
            )
        else:
            root = Path(directory)
            root.mkdir(parents=True, mode=0o700)
            for name, contents in files.items():
                (root / name).write_bytes(contents)
                (root / name).chmod(0o600)
        try:
            result = self._run(command, text=True)
        except (subprocess.CalledProcessError, OSError, KeyboardInterrupt) as exc:
            raise RuntimeError(
                f"SUNK submission did not confirm a job ID. Check squeue --name={directory.name} before resubmitting. Staged files: {directory}"
            ) from exc
        job_id = result.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise RuntimeError(
                f"Unrecognized sbatch response; check squeue --name={directory.name} before resubmitting"
            )
        LOG.info(
            "SUNK job %s submitted. Logs: %s/slurm-%s.out. Status: squeue -j %s. Cancel: scancel %s",
            job_id,
            directory,
            job_id,
            job_id,
            job_id,
        )
        return {"job_id": job_id, "directory": str(directory)}

    def _train_cks(self, training, launcher, launcher_args):
        from . import cks

        cfg = self.image_config
        if launcher not in {"accelerate", "torchrun"}:
            raise ValueError("CKS supports the accelerate and torchrun launchers")
        reserved = {
            "nnodes",
            "nproc_per_node",
            "node_rank",
            "master_addr",
            "master_port",
            "num_machines",
            "num_processes",
            "machine_rank",
            "main_process_ip",
            "main_process_port",
            "rdzv_backend",
            "rdzv_endpoint",
            "rdzv_id",
            "standalone",
        }
        if any(
            not isinstance(arg, str)
            or arg.split("=", 1)[0].lstrip("-").replace("-", "_") in reserved
            for arg in launcher_args
        ):
            raise ValueError(
                "CKS controls distributed topology; remove conflicting launcher arguments"
            )
        job_name = cks.new_job_name(cfg)
        training = cks.place_output_dir(cfg, training, job_name)
        image = cfg.image or cfg.image_build.tag
        if cfg.image_build and not cfg.dry_run:
            image = build_and_push_image(cfg.image_build)
        manifests = cks.render_manifests(
            cfg, training, launcher, launcher_args, image, job_name
        )
        return cks.submit(cfg, manifests, job_name)
