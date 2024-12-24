"""Definition of differential transformer plugin."""

import logging

from axolotl.integrations.base import BasePlugin

LOG = logging.getLogger(__name__)


class DifferentialTransformerPlugin(BasePlugin):
    """Plugin for differential transformer integration with Axolotl."""

    def get_input_args(self):
        return "axolotl.integrations.diff_transformer.args.DifferentialTransformerArgs"
