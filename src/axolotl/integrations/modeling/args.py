"""
Args for using Axolotl custom modeling
"""

from pydantic import BaseModel


class AxolotlModelingArgs(BaseModel):
    """
    Args for using Axolotl custom modeling
    """

    use_liger_fused_rms_add: bool = False
