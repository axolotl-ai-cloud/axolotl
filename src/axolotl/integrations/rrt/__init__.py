"""
Axolotl Plugin for Relaxed Recursive Transformers
"""

import logging

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.rrt.modeling import register_rrt_model

LOG = logging.getLogger(__name__)


class RelaxedRecursiveTransformerPlugin(BasePlugin):
    """
    Plugin for Relaxed Recursive Transformers integration with Axolotl
    """

    def get_input_args(self):
        return "axolotl.integrations.rrt.args.RelaxedRecursiveTransformerArgs"

    def register(self):
        LOG.info(
            "Registering Relaxed Recursive Transformers modeling with transformers"
        )
        register_rrt_model()
