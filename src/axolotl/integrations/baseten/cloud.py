"""Baseten Cloud CLI"""

import shutil
import subprocess  # nosec B404
import tempfile
from os.path import dirname
from pathlib import Path
from typing import Literal

import yaml

from axolotl.cli.cloud.base import CloudLauncher
from axolotl.cli.cloud.images import build_and_push_image

from .args import BasetenImageConfig


class BasetenCloud(CloudLauncher):
    """Baseten Cloud Axolotl CLI"""

    def __init__(self, config: dict, *, config_dir: Path | None = None):
        self.config = config
        self.image_config = BasetenImageConfig.from_config(
            config, config_dir=config_dir
        )
        if self.image_config.image_build and not self.image_config.image_build.tag:
            raise ValueError("Baseten image_build requires a registry tag")

    @classmethod
    def from_config(cls, config: dict, *, config_dir: Path | None = None):
        return cls(config, config_dir=config_dir)

    def preprocess(self, config_yaml: str, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Separate preprocess function for Baseten is not "
            "implemented and will happen during hte train step."
        )

    def train(
        self,
        config_yaml: str,
        launcher: Literal["accelerate", "torchrun", "python"] = "accelerate",
        launcher_args: list[str] | None = None,
        local_dirs: dict[str, str] | None = None,  # pylint: disable=unused-argument
        **kwargs,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self.config.copy()
            if build := self.image_config.image_build:
                config["image"] = build_and_push_image(build)
            config.pop("image_build", None)
            config["launcher"] = launcher
            config["launcher_args"] = launcher_args
            with open(tmp_dir + "/cloud.yaml", "w", encoding="utf-8") as cloud_fout:
                yaml.dump(config, cloud_fout)
            with open(tmp_dir + "/train.yaml", "w", encoding="utf-8") as config_fout:
                config_fout.write(config_yaml)
            shutil.copyfile(dirname(__file__) + "/template/run.sh", tmp_dir + "/run.sh")
            shutil.copyfile(
                dirname(__file__) + "/template/train_sft.py", tmp_dir + "/train_sft.py"
            )
            subprocess.run(  # nosec B603 B607
                ["truss", "train", "push", "train_sft.py"], cwd=tmp_dir, check=True
            )
