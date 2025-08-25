"""Init for axolotl.core.trainers"""

# flake8: noqa

from .base import AxolotlTrainer
from .dpo.trainer import AxolotlDPOTrainer
from .mamba import AxolotlMambaTrainer
from .trl import (
    AxolotlCPOTrainer,
    AxolotlKTOTrainer,
    AxolotlORPOTrainer,
    AxolotlPRMTrainer,
    AxolotlRewardTrainer,
)
