"""
base class for cloud platforms from cli
"""

from abc import ABC, abstractmethod
from typing import Literal


class Cloud(ABC):
    """
    Abstract base class for cloud platforms.
    """

    @abstractmethod
    def preprocess(self, config_yaml: str, *args, **kwargs) -> None:
        pass

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
