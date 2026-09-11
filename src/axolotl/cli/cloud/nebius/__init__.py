"""Run Axolotl training through the Nebius Serverless Jobs CLI."""

import json
import logging
import re
import shlex
import shutil
import subprocess  # nosec B404
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml

from axolotl.cli.cloud.base import Cloud

LOG = logging.getLogger(__name__)


class NebiusCloud(Cloud):
    """Submit one training job using an installed, authenticated Nebius CLI."""

    OPTIONS = {
        "provider",
        "image",
        "platform",
        "preset",
        "profile",
        "parent_id",
        "subnet_id",
        "disk_size",
        "timeout",
        "output",
        "volumes",
        "env",
        "env_secret",
        "show_context",
        "dry_run",
    }

    def __init__(self, config):
        self.config = dict(config)
        unknown = self.config.keys() - self.OPTIONS
        if unknown:
            raise ValueError(
                f"Unknown Nebius cloud options: {', '.join(sorted(unknown))}"
            )
        for key in ("image", "platform", "preset"):
            self._string(key, required=True)
        for key in ("profile", "parent_id", "subnet_id", "disk_size", "output"):
            self._string(key)
        timeout = self.config.get("timeout", 86400)
        if type(timeout) is not int or not 3600 <= timeout <= 604800:
            raise ValueError(
                "Nebius timeout must be an integer from 3600 to 604800 seconds"
            )
        for key in ("show_context", "dry_run"):
            if key in self.config and type(self.config[key]) is not bool:
                raise ValueError(f"Nebius {key} must be a boolean")
        if self.config.get("show_context") and self.config.get("dry_run"):
            raise ValueError("Choose either show_context or dry_run")
        self.volumes = self._volumes()
        self.env = self._environment("env")
        self.env_secret = self._environment("env_secret")
        if self.env.keys() & self.env_secret.keys():
            raise ValueError("A variable cannot appear in both env and env_secret")

    def _string(self, key, required=False):
        value = self.config.get(key)
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"Nebius {key} must be a nonempty string")
        return value

    def _environment(self, key):
        values = self.config.get(key, {})
        if not isinstance(values, dict):
            raise ValueError(f"Nebius {key} must map variable names to strings")
        for name, value in values.items():
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", name
            ):
                raise ValueError(f"Invalid environment variable name in {key}")
            if name in {
                "NEBIUS_OUTPUT_DIR",
                "AXOLOTL_NEBIUS_COMPLETION_FILE",
                "AXOLOTL_NEBIUS_EXPORT_DIR",
            }:
                raise ValueError(f"{name} is managed by the Nebius launcher")
            if (
                not isinstance(value, str)
                or "\x00" in value
                or (key == "env_secret" and not value)
            ):
                raise ValueError(f"Nebius {key} values must be strings")
        return values

    def _volumes(self):
        volumes = self.config.get("volumes", [])
        if not isinstance(volumes, list):
            raise ValueError("Nebius volumes must be a list")
        mounts = [PurePosixPath("/outputs")]
        result = []
        for volume in volumes:
            if not isinstance(volume, dict) or volume.keys() - {
                "source",
                "mount",
                "mode",
            }:
                raise ValueError("Each volume needs source, mount, and optional mode")
            source, mount = volume.get("source"), volume.get("mount")
            mode = volume.get("mode", "ro")
            if (
                not isinstance(source, str)
                or not source
                or ":" in source
                or "\x00" in source
            ):
                raise ValueError("Volume source must be a Nebius resource ID or name")
            if (
                not isinstance(mount, str)
                or not mount.startswith("/")
                or ":" in mount
                or "\x00" in mount
            ):
                raise ValueError("Volume mount must be an absolute container path")
            path = PurePosixPath("/" + mount.lstrip("/"))
            if ".." in path.parts or any(
                path.is_relative_to(p) or p.is_relative_to(path) for p in mounts
            ):
                raise ValueError(
                    "Volume mounts must not overlap each other or /outputs"
                )
            if mode not in {"ro", "rw"}:
                raise ValueError("Volume mode must be ro or rw")
            mounts.append(path)
            result.append({"source": source, "mount": str(path), "mode": mode})
        return result

    def preprocess(self, config_yaml: str, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Nebius currently supports train only; dataset preparation runs during training"
        )

    def train(
        self,
        config_yaml: str,
        launcher: Literal["accelerate", "torchrun", "python"] = "accelerate",
        launcher_args: list[str] | None = None,
        local_dirs: dict[str, str] | None = None,
        **kwargs,
    ):
        from axolotl.cli.cloud.nebius.runner import validate_config

        cfg = yaml.safe_load(config_yaml)
        if not isinstance(cfg, dict):
            raise ValueError("The training config must be a YAML mapping")
        cfg.update(kwargs)
        validate_config(cfg, [volume["mount"] for volume in self.volumes])
        if launcher not in {"accelerate", "torchrun", "python"}:
            raise ValueError(f"Unsupported launcher: {launcher}")
        if launcher_args and not all(isinstance(arg, str) for arg in launcher_args):
            raise ValueError("Launcher arguments must be strings")
        if local_dirs:
            LOG.warning(
                "Nebius uploads the training config only, not your working directory. "
                "Use Hub datasets or container paths on mounted volumes."
            )
        executable = shutil.which("nebius")
        if not executable:
            raise RuntimeError(
                "Install the Nebius CLI (with 'ai job run') and run 'nebius auth login'"
            )
        name = f"axolotl-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="axolotl-nebius-") as directory:
            root = Path(directory)
            (root / "train.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            (root / "launch.json").write_text(
                json.dumps(
                    {
                        "launcher": launcher,
                        "launcher_args": launcher_args or [],
                        "mounts": [volume["mount"] for volume in self.volumes],
                    }
                ),
                encoding="utf-8",
            )
            for source, target in (
                ("runner.py", "run.py"),
                ("completion.py", "nebius_completion.py"),
                ("storage.py", "nebius_storage.py"),
            ):
                shutil.copyfile(Path(__file__).with_name(source), root / target)
            command = [executable, "ai", "job", "run", "run.py", "--name", name]
            for key in (
                "image",
                "platform",
                "preset",
                "profile",
                "parent_id",
                "subnet_id",
                "disk_size",
            ):
                if self.config.get(key) is not None:
                    command.extend([f"--{key.replace('_', '-')}", self.config[key]])
            command.extend(
                [
                    "--timeout",
                    f"{self.config.get('timeout', 86400)}s",
                    "--output",
                    self.config.get("output") or "auto",
                ]
            )
            for volume in self.volumes:
                command.extend(
                    [
                        "--volume",
                        f"{volume['source']}:{volume['mount']}:{volume['mode']}",
                    ]
                )
            for flag, values in (
                ("--env", self.env),
                ("--env-secret", self.env_secret),
            ):
                for key, value in values.items():
                    command.extend([flag, f"{key}={value}"])
            for key in ("show_context", "dry_run"):
                if self.config.get(key):
                    command.append(f"--{key.replace('_', '-')}")
            LOG.info(
                "Nebius job name: %s. The CLI reports the job ID and output location.",
                name,
            )
            LOG.warning(
                "Ctrl+C stops following logs; the job keeps running. Cancel it with 'nebius ai job cancel JOB_ID'."
            )
            try:
                result = subprocess.run(command, cwd=directory, check=False)  # nosec B603
            except KeyboardInterrupt:
                LOG.warning(
                    "Reattach or cancel using the job ID, or resolve the job by name: %s",
                    name,
                )
                raise
            if result.returncode:
                # CalledProcessError would include environment values in its command.
                raise RuntimeError(
                    self._failure_message(executable, name, result.returncode)
                )

    def _job_command(self, executable, action, *args):
        command = [executable, "ai", "job", action, *args]
        if self.config.get("profile"):
            command.extend(["--profile", self.config["profile"]])
        if action == "get-by-name" and self.config.get("parent_id"):
            command.extend(["--parent-id", self.config["parent_id"]])
        return command

    def _failure_message(self, executable, name, returncode):
        prefix = f"Nebius CLI exited with code {returncode} for {name}. "
        if self.config.get("show_context") or self.config.get("dry_run"):
            return (
                prefix
                + "Preview/validation failed; no training job was requested. See the CLI error above."
            )
        lookup = self._job_command(executable, "get-by-name", "--name", name)
        fallback = (
            prefix
            + "Remote state could not be confirmed. Do not resubmit until you check: "
            + shlex.join([*lookup, "--format", "json"])
            + ". Stopping log following does not cancel the job."
        )
        try:
            result = subprocess.run(  # nosec B603
                [*lookup, "--format", "json"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if result.returncode:
                return fallback
            job = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return fallback
        if not isinstance(job, dict):
            return fallback
        metadata, status = job.get("metadata"), job.get("status")
        if not isinstance(metadata, dict) or not isinstance(status, dict):
            return fallback
        job_id, state = metadata.get("id"), status.get("state")
        if (
            metadata.get("name") != name
            or not isinstance(job_id, str)
            or not re.fullmatch(r"aijob-[a-zA-Z0-9]+", job_id)
            or not isinstance(state, str)
            or not re.fullmatch(r"[A-Z_]{1,64}", state)
        ):
            return fallback
        details = status.get("state_details")
        code = details.get("code") if isinstance(details, dict) else None
        if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,80}", code):
            code = None
        summary = prefix + f"Job {job_id}: {state}" + (f" ({code}). " if code else ". ")
        logs = shlex.join(self._job_command(executable, "logs", job_id))
        if state in {"ERROR", "FAILED", "CANCELLED"}:
            if code == "NotEnoughResources":
                return (
                    summary
                    + f"Nebius could not allocate the requested {self.config['platform']}/{self.config['preset']} capacity. "
                    "The job is terminal; you may rerun the same command later to submit a new job. "
                    "Capacity may still be unavailable. No automatic retry was made."
                )
            return (
                summary
                + "The job is terminal. Inspect the cause before starting a new job: "
                + logs
                + ". No automatic retry was made."
            )
        if state == "COMPLETED":
            return (
                summary
                + "The job completed despite the CLI error. Verify its outputs before considering another submission."
            )
        get = shlex.join(self._job_command(executable, "get", job_id))
        cancel = shlex.join(self._job_command(executable, "cancel", job_id))
        return (
            summary
            + "A terminal outcome has not been confirmed; do not submit a duplicate. "
            + f"Check: {get}. To cancel: {cancel}."
        )
