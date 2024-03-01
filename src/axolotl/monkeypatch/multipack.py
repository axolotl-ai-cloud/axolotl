"""multipack patching for v2 of sample packing"""

import transformers
from transformers.integrations import is_deepspeed_zero3_enabled

from axolotl.monkeypatch.mixtral import patch_mixtral_moe_forward_zero3
from axolotl.monkeypatch.utils import get_unpad_data

SUPPORTED_MULTIPACK_MODEL_TYPES = ["mixtral", "qwen2", "falcon", "phi", "gemma"]


def patch_for_multipack(model_type):
    if model_type == "mixtral":
        transformers.models.mixtral.modeling_mixtral._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
        if is_deepspeed_zero3_enabled():
            patch_mixtral_moe_forward_zero3()
    elif model_type == "qwen2":
        transformers.models.qwen2.modeling_qwen2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "falcon":
        transformers.models.falcon.modeling_falcon._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "phi":
        transformers.models.phi.modeling_phi._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "gemma":
        transformers.models.gemma.modeling_gemma._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
