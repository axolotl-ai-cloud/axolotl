"""
Axolotl custom modeling plugin
"""

from axolotl.integrations.base import BasePlugin


class AxolotlModelingPlugin(BasePlugin):
    """
    Axolotl custom modeling plugin
    """

    def get_input_args(self) -> str | None:
        return "axolotl.integrations.modeling.AxolotlModelingArgs"

    def register(self, cfg):  # pylint: disable=unused-argument
        if cfg.use_liger_fused_rms_add:
            from .gemma3 import patch_gemma3
            from .llama import patch_llama

            patch_gemma3()
            patch_llama()
