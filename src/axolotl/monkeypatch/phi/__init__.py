"""
Patches to support multipack for phi2
"""
import transformers

from axolotl.monkeypatch.utils import get_unpad_data


def replace_phi_attn_with_multipack_flash_attn():
    transformers.models.phi.modeling_phi._get_unpad_data = (  # pylint: disable=protected-access
        get_unpad_data
    )
