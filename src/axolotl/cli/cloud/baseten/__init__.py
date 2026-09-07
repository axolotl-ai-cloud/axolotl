"""Baseten Cloud CLI"""

import shutil
import subprocess  # nosec B404
import tempfile
from os.path import dirname
from typing import Literal

import yaml

from axolotl.cli.cloud.base import Cloud


class BasetenCloud(Cloud):
    """Baseten Cloud Axolotl CLI"""

    def __init__(self, config: dict):
        self.config = config

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
                ["truss", "train", "push", "train_sft.py"], cwd=tmp_dir, check=False
            )
