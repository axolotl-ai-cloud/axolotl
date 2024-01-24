"""
Patches to support multipack for falcon
"""
import transformers

from axolotl.monkeypatch.utils import get_unpad_data


def replace_falcon_attn_with_multipack_flash_attn():
    transformers.models.falcon.modeling_falcon._get_unpad_data = (  # pylint: disable=protected-access
        get_unpad_data
    )
