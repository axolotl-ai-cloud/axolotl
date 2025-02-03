"""
Axolotl GRPO trainer
"""
from trl import GRPOTrainer

from axolotl.core.trainers.base import SchedulerMixin


class AxolotlGRPOTrainer(SchedulerMixin, GRPOTrainer):
    """
    Extend the base GRPOTrainer for axolotl helpers
    """
