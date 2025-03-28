"""
Modal Cloud support from CLI
"""

import copy
import json
import os
import subprocess  # nosec B404
from pathlib import Path
from random import randint
from typing import Optional

import modal

from axolotl.cli.cloud.base import Cloud


def run_cmd(cmd: str, run_folder: str, volumes=None):
    """Run a command inside a folder, with Modal Volume reloading before and commit on success."""
    # Ensure volumes contain latest files.
    if volumes:
        for _, vol in volumes.items():
            vol.reload()

    # modal workaround so it doesn't use the automounted axolotl
    new_env = copy.deepcopy(os.environ)

    if "PYTHONPATH" in new_env:
        paths = ["/workspace/mounts"]
        for sub_python_path_str in new_env["PYTHONPATH"].split(":"):
            sub_python_path = Path(sub_python_path_str)
            if not sub_python_path.joinpath("src", "axolotl").exists():
                # we don't want to use the automounted axolotl or unexpected behavior happens
                paths.append(str(sub_python_path))
        if paths:
            new_env["PYTHONPATH"] = ":".join(paths)
        else:
            del new_env["PYTHONPATH"]

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(  # nosec B603
        cmd.split(), cwd=run_folder, env=new_env
    ):
        exit(exit_code)  # pylint: disable=consider-using-sys-exit

    # Commit writes to volume.
    if volumes:
        for _, vol in volumes.items():
            vol.commit()


class ModalCloud(Cloud):
    """
    Modal Cloud implementation.
    """

    def __init__(self, config, app=None):
        self.config = config
        if not app:
            app = modal.App()
        self.app = app

        self.volumes = {}
        if config.volumes:
            for volume_config in config.volumes:
                _, mount, vol = self.create_volume(volume_config)
                self.volumes[mount] = (vol, volume_config)

    def get_env(self):
        res = {
            "HF_DATASETS_CACHE": "/workspace/data/huggingface-cache/datasets",
            "HF_HUB_CACHE": "/workspace/data/huggingface-cache/hub",
        }

        for key in self.config.get("env", []):
            if isinstance(key, str):
                if val := os.environ.get(key, ""):
                    res[key] = val
            elif isinstance(key, dict):
                (key_, val) = list(key.items())[0]
                res[key_] = val
        return res

    def get_image(self):
        docker_tag = "main-py3.11-cu124-2.5.1"
        if self.config.docker_tag:
            docker_tag = self.config.docker_tag
        docker_image = f"axolotlai/axolotl:{docker_tag}"

        # grab the sha256 hash from docker hub for this image+tag
        # this ensures that we always get the latest image for this tag, even if it's already cached
        try:
            manifest = subprocess.check_output(  # nosec B602
                f"docker manifest inspect {docker_image}",
                shell=True,
            ).decode("utf-8")
            sha256_hash = json.loads(manifest)["manifests"][0]["digest"]
        except subprocess.CalledProcessError:
            sha256_hash = None

        # create the image
        if sha256_hash:
            image = modal.Image.from_registry(f"axolotlai/axolotl@{sha256_hash}")
        else:
            image = modal.Image.from_registry(docker_image)

        dockerfile_commands = []
        if self.config.dockerfile_commands:
            dockerfile_commands.extend(self.config.dockerfile_commands)

        # branch
        if self.config.branch:
            dockerfile_commands.extend(
                [
                    # Random id for cache busting of branch commits
                    f"RUN echo '{str(randint(0, 1000000))}'",  # nosec B311
                    f"RUN cd /workspace/axolotl && git fetch && git checkout {self.config.branch} && git pull",
                ]
            )

        if dockerfile_commands:
            image = image.dockerfile_commands(dockerfile_commands)

        if env := self.get_env():
            image = image.env(env)

        return image

    def get_secrets(self):
        res = []
        if self.config.secrets:
            for key in self.config.get("secrets", []):
                # pylint: disable=duplicate-code
                if isinstance(key, str):
                    if val := os.environ.get(key, ""):
                        res.append(modal.Secret.from_dict({key: val}))
                elif isinstance(key, dict):
                    (key_, val) = list(key.items())[0]
                    res.append(modal.Secret.from_dict({key_: val}))
        return res

    def create_volume(self, volume_config):
        name = volume_config.name
        mount = volume_config.mount
        return name, mount, modal.Volume.from_name(name, create_if_missing=True)

    def get_ephemeral_disk_size(self):
        return 1000 * 525  # 1 TiB

    def get_preprocess_timeout(self):
        if self.config.timeout_preprocess:
            return int(self.config.timeout_preprocess)
        return 60 * 60 * 3  # 3 hours

    def get_preprocess_memory(self):
        memory = 128  # default to 128GiB
        if self.config.memory:
            memory = int(self.config.memory)
        if self.config.memory_preprocess:
            memory = int(self.config.memory_preprocess)
        return 1024 * memory

    def get_preprocess_env(self):
        return self.app.function(
            image=self.get_image(),
            volumes={k: v[0] for k, v in self.volumes.items()},
            cpu=8.0,
            ephemeral_disk=self.get_ephemeral_disk_size(),
            memory=self.get_preprocess_memory(),
            timeout=self.get_preprocess_timeout(),
            secrets=self.get_secrets(),
        )

    def preprocess(self, config_yaml: str, *args, **kwargs):
        modal_fn = self.get_preprocess_env()(_preprocess)
        with modal.enable_output():
            with self.app.run(detach=True):
                modal_fn.remote(
                    config_yaml,
                    volumes={k: v[0] for k, v in self.volumes.items()},
                    *args,
                    **kwargs,
                )

    def get_train_timeout(self):
        if self.config.timeout:
            return int(self.config.timeout)
        return 60 * 60 * 24  # 24 hours

    def get_train_gpu(self):  # pylint: disable=too-many-return-statements
        count = self.config.gpu_count or 1
        family = self.config.gpu.lower() or "l40s"

        if family == "l40s":
            return modal.gpu.L40S(count=count)
        if family in ["a100", "a100-40gb"]:
            return modal.gpu.A100(count=count, size="40GB")
        if family == "a100-80gb":
            return modal.gpu.A100(count=count, size="80GB")
        if family in ["a10", "a10g"]:
            return modal.gpu.A10G(count=count)
        if family == "h100":
            return modal.gpu.H100(count=count)
        if family == "t4":
            return modal.gpu.T4(count=count)
        if family == "l4":
            return modal.gpu.L4(count=count)
        raise ValueError(f"Unsupported GPU family: {family}")

    def get_train_memory(self):
        memory = 128  # default to 128GiB
        if self.config.memory:
            memory = int(self.config.memory)
        return 1024 * memory

    def get_train_env(self, local_dirs=None):
        image = self.get_image()
        for mount, local_dir in (local_dirs or {}).items():
            image = image.add_local_dir(local_dir, mount)
        return self.app.function(
            image=image,
            volumes={k: v[0] for k, v in self.volumes.items()},
            cpu=16.0,
            gpu=self.get_train_gpu(),
            memory=self.get_train_memory(),
            timeout=self.get_train_timeout(),
            secrets=self.get_secrets(),
        )

    def train(
        self,
        config_yaml: str,
        accelerate: bool = True,
        local_dirs: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        modal_fn = self.get_train_env(local_dirs)(_train)
        with modal.enable_output():
            with self.app.run(detach=True):
                modal_fn.remote(
                    config_yaml,
                    accelerate=accelerate,
                    volumes={k: v[0] for k, v in self.volumes.items()},
                    **kwargs,
                )

    def lm_eval(self, config_yaml: str):
        modal_fn = self.get_train_env()(_lm_eval)
        with modal.enable_output():
            with self.app.run(detach=True):
                if self.config.get("spawn", False):
                    modal_fn_exec = modal_fn.spawn
                else:
                    modal_fn_exec = modal_fn.remote
                modal_fn_exec(
                    config_yaml,
                    volumes={k: v[0] for k, v in self.volumes.items()},
                )


def _preprocess(config_yaml: str, volumes=None):
    Path("/workspace/mounts").mkdir(parents=True, exist_ok=True)
    with open("/workspace/mounts/config.yaml", "w", encoding="utf-8") as f_out:
        f_out.write(config_yaml)
    run_folder = "/workspace/mounts"
    run_cmd(
        "axolotl preprocess /workspace/mounts/config.yaml --dataset-processes=8",
        run_folder,
        volumes,
    )


def _train(config_yaml: str, accelerate: bool = True, volumes=None, **kwargs):
    Path("/workspace/mounts").mkdir(parents=True, exist_ok=True)
    with open("/workspace/mounts/config.yaml", "w", encoding="utf-8") as f_out:
        f_out.write(config_yaml)
    run_folder = "/workspace/mounts"
    if accelerate:
        accelerate_args = "--accelerate"
    else:
        accelerate_args = "--no-accelerate"
    num_processes_args = ""
    if num_processes := kwargs.pop("num_processes", None):
        num_processes_args = f"--num-processes {num_processes}"
    run_cmd(
        f"axolotl train {accelerate_args} {num_processes_args} /workspace/mounts/config.yaml",
        run_folder,
        volumes,
    )


def _lm_eval(config_yaml: str, volumes=None):
    Path("/workspace/mounts").mkdir(parents=True, exist_ok=True)
    with open("/workspace/mounts/config.yaml", "w", encoding="utf-8") as f_out:
        f_out.write(config_yaml)
    run_folder = "/workspace/mounts"
    run_cmd(
        "axolotl lm-eval /workspace/mounts/config.yaml",
        run_folder,
        volumes,
    )
