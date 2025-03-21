"""
base class for cloud platforms from cli
"""

from abc import ABC, abstractmethod


class Cloud(ABC):
    """
    Abstract base class for cloud platforms.
    """

    @abstractmethod
    def preprocess(self, config_yaml: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, config_yaml: str, accelerate: bool = True) -> str:
        pass
