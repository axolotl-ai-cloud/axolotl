"""
Patches to support multipack for mixtral
"""
import transformers

from axolotl.monkeypatch.utils import get_unpad_data


def replace_mixtral_attn_with_multipack_flash_attn():
    transformers.models.mixtral.modeling_mixtral._get_unpad_data = (  # pylint: disable=protected-access
        get_unpad_data
    )
