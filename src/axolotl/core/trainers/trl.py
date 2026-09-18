"""Module for TRL RL trainers"""

from trl import KTOTrainer, RewardTrainer
from trl.experimental.cpo import CPOTrainer
from trl.experimental.orpo import ORPOTrainer

from axolotl.core.trainers.mixins import DistributedParallelMixin, RngLoaderMixin
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.core.trainers.mixins.scheduler import SchedulerMixin
from axolotl.core.training_args import PRM_TRL_ERROR

try:
    from trl.experimental.prm import PRMTrainer
except ImportError:

    class PRMTrainer:  # type: ignore[no-redef]
        """Placeholder so this module stays importable without trl's PRM."""

        # __new__, not __init__: the trainer mixins put transformers' Trainer
        # ahead of this placeholder in the MRO.
        def __new__(cls, *args, **kwargs):
            raise ImportError(PRM_TRL_ERROR)


class AxolotlORPOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    ORPOTrainer,
):
    """
    Extend the base ORPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "orpo"]


class AxolotlKTOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    KTOTrainer,
):
    """
    Extend the base KTOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "kto"]


class AxolotlCPOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    CPOTrainer,
):
    """
    Extend the base CPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "cpo"]


class AxolotlRewardTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    RewardTrainer,
):
    """
    Extend the base RewardTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "reward"]


class AxolotlPRMTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    PRMTrainer,
):
    """
    Extend the base trl.PRMTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "prm"]
