"""Definition of RALA plugin."""

import logging

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.rala.auto.llama.modeling_rala import register_rala_model

LOG = logging.getLogger(__name__)


class RalaPlugin(BasePlugin):
    """
    Plugin for Rala integration with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.rala.args.RalaArgs"

    def register(self):
        register_rala_model()
