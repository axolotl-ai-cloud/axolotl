"""multipack patching for v2 of sample packing"""
import importlib

import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import is_deepspeed_zero3_enabled

from axolotl.monkeypatch.mixtral import patch_mixtral_moe_forward_zero3
from axolotl.monkeypatch.utils import get_unpad_data

SUPPORTED_MULTIPACK_MODEL_TYPES = [
    "mixtral",
    "qwen2",
    "falcon",
    "phi",
    "gemma",
    "gemmoe",
    "starcoder2",
]


def patch_for_multipack(model_type, model_name=None):
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
    elif model_type == "starcoder2":
        transformers.models.starcoder2.modeling_starcoder2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "gemmoe":
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # we need to load the model here in order for modeling_gemmoe to be available
        with init_empty_weights():
            AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        module_name = model_config.__class__.__module__.replace(
            ".configuration_gemmoe", ".modeling_gemmoe"
        )
        modeling_gemmoe = importlib.import_module(module_name)
        modeling_gemmoe._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
