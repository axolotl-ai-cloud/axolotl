"""
base class for cloud platforms from cli
"""

from abc import ABC, abstractmethod

from axolotl.utils.schemas.runtime import LauncherChoice


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
        launcher: LauncherChoice = "accelerate",
        launcher_args: list[str] | None = None,
        local_dirs: dict[str, str] | None = None,
        runtime_yaml: str | None = None,
        **kwargs,
    ):
        pass
