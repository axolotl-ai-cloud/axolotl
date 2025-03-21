"""Init for axolotl.core.trainers"""

# pylint: disable=unused-import
# flake8: noqa

from .base import AxolotlTrainer
from .dpo.trainer import AxolotlDPOTrainer
from .grpo.trainer import AxolotlGRPOTrainer
from .mamba import AxolotlMambaTrainer
from .relora import ReLoRATrainer
from .trl import (
    AxolotlCPOTrainer,
    AxolotlKTOTrainer,
    AxolotlORPOTrainer,
    AxolotlPRMTrainer,
    AxolotlRewardTrainer,
    TRLPPOTrainer,
)
