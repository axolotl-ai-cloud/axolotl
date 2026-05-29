"""Init for axolotl.core.trainers"""

# flake8: noqa

from axolotl.utils import make_lazy_getattr

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

__getattr__ = make_lazy_getattr(_LAZY_IMPORTS, __name__, globals())
