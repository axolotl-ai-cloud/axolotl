"""Init for axolotl.core.trainers"""

# pylint: disable=unused-import
# flake8: noqa

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.trainers.dpo import AxolotlDPOTrainer
from axolotl.core.trainers.grpo import AxolotlGRPOTrainer
from axolotl.core.trainers.mamba import AxolotlMambaTrainer
from axolotl.core.trainers.relora import ReLoRATrainer
from axolotl.core.trainers.trl import (
    AxolotlCPOTrainer,
    AxolotlKTOTrainer,
    AxolotlORPOTrainer,
    AxolotlPPOTrainer,
    AxolotlPRMTrainer,
    AxolotlRewardTrainer,
)
