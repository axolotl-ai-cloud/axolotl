"""
Plugin init to add KD support to Axolotl.
"""
from axolotl.integrations.base import BasePlugin

from .args import KDArgs  # pylint: disable=unused-import. # noqa: F401


class KDPlugin(BasePlugin):
    """
    Plugin for KD support in Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.kd.KDArgs"

    def get_trainer_cls(self, cfg):
        if cfg.kd_trainer:
            from .trainer import AxolotlKDTrainer

            return AxolotlKDTrainer
        return None
