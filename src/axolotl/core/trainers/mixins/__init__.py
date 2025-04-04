"""Init for axolotl.core.trainers.mixins"""

# pylint: disable=unused-import
# flake8: noqa

from axolotl.core.trainers.mixins.optimizer import OptimizerMixin
from axolotl.core.trainers.mixins.rng_state_loader import RngLoaderMixin
from axolotl.core.trainers.mixins.scheduler import SchedulerMixin


class TrainerMixins(
    OptimizerMixin, RngLoaderMixin, SchedulerMixin
):
    """Stub class combining all mixins for Axolotl trainers."""
