"""Launcher contract for complete remote Axolotl jobs."""

from abc import ABC, abstractmethod
from typing import Literal


class CloudLauncher(ABC):
    """
    Launcher plugin for a complete remote Axolotl process.

    Providers accept a cloud configuration dict and own its validation. Training
    receives the original YAML, local directory mounts, launcher arguments, and
    config overrides. Provider-specific training hooks belong in BasePlugin.
    Only train is required; optional operations fail explicitly by default.
    Per-step remote compute belongs in a training backend, not this interface.
    """

    def __init__(self, config: dict):
        self.config = config

    def preprocess(self, config_yaml: str, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support cloud preprocessing"
        )

    def lm_eval(self, config_yaml: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support cloud lm-eval"
        )

    @abstractmethod
    def train(
        self,
        config_yaml: str,
        launcher: Literal["accelerate", "torchrun", "python"] = "accelerate",
        launcher_args: list[str] | None = None,
        local_dirs: dict[str, str] | None = None,
        **kwargs,
    ):
        pass


Cloud = CloudLauncher
