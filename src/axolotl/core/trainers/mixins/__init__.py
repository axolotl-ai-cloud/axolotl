"""Init for axolotl.core.trainers.mixins"""

# pylint: disable=unused-import
# flake8: noqa

from axolotl.core.trainers.mixins.optimizer import OptimizerMixin
from axolotl.core.trainers.mixins.rng_state_loader import RngLoaderMixin
from axolotl.core.trainers.mixins.scheduler import SchedulerMixin
from axolotl.core.trainers.mixins.sequence_parallel import SequenceParallelMixin


class TrainerMixins(
    OptimizerMixin, RngLoaderMixin, SchedulerMixin, SequenceParallelMixin
):
    """Stub class combining all mixins for Axolotl trainers."""
