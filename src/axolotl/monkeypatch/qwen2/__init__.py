"""
Patches to support multipack for qwen2
"""
import transformers

from axolotl.monkeypatch.utils import get_unpad_data


def replace_qwen2_attn_with_multipack_flash_attn():
    transformers.models.qwen2.modeling_qwen2._get_unpad_data = (  # pylint: disable=protected-access
        get_unpad_data
    )
