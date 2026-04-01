"""Init for axolotl.core.trainers"""

# flake8: noqa

from .base import AxolotlTrainer

# noinspection PyUnresolvedReferences
__all__ = [
    "AxolotlTrainer",
    "AxolotlCPOTrainer",
    "AxolotlDPOTrainer",
    "AxolotlEBFTTrainer",
    "AxolotlKTOTrainer",
    "AxolotlMambaTrainer",
    "AxolotlORPOTrainer",
    "AxolotlPRMTrainer",
    "AxolotlRewardTrainer",
    "AxolotlStridedEBFTTrainer",
]

_LAZY_IMPORTS = {
    "AxolotlDPOTrainer": ".dpo.trainer",
    "AxolotlStridedEBFTTrainer": ".ebft.strided",
    "AxolotlEBFTTrainer": ".ebft.trainer",
    "AxolotlMambaTrainer": ".mamba",
    "AxolotlCPOTrainer": ".trl",
    "AxolotlKTOTrainer": ".trl",
    "AxolotlORPOTrainer": ".trl",
    "AxolotlPRMTrainer": ".trl",
    "AxolotlRewardTrainer": ".trl",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
