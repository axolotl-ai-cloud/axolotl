"""multipack patching for v2 of sample packing"""
import importlib

import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import is_deepspeed_zero3_enabled

from axolotl.monkeypatch.mixtral import patch_mixtral_moe_forward_zero3
from axolotl.monkeypatch.utils import get_unpad_data

SUPPORTED_MULTIPACK_MODEL_TYPES = [
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "falcon",
    "phi",
    "gemma",
    "gemma2",
    "gemmoe",
    "starcoder2",
    "deepseek_v2",
]


def patch_for_multipack(model_type, model_name=None):
    if model_type == "gemmoe":
        patch_remote(model_name, ".configuration_gemmoe", ".modeling_gemmoe")
    elif model_type == "deepseek_v2":
        patch_remote(model_name, ".configuration_deepseek", ".modeling_deepseek")
    elif hasattr(transformers, "modeling_flash_attention_utils"):
        transformers.modeling_flash_attention_utils._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
        if model_type == "mixtral" and is_deepspeed_zero3_enabled():
            patch_mixtral_moe_forward_zero3()
        return

    # retain for legacy
    if model_type == "mixtral":
        transformers.models.mixtral.modeling_mixtral._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
        if is_deepspeed_zero3_enabled():
            patch_mixtral_moe_forward_zero3()
    elif model_type == "llama":
        if hasattr(transformers.models.llama.modeling_llama, "_get_unpad_data"):
            transformers.models.llama.modeling_llama._get_unpad_data = (  # pylint: disable=protected-access
                get_unpad_data
            )
    elif model_type == "mistral":
        if hasattr(transformers.models.mistral.modeling_mistral, "_get_unpad_data"):
            transformers.models.llama.modeling_llama._get_unpad_data = (  # pylint: disable=protected-access
                get_unpad_data
            )
    elif model_type == "qwen2":
        transformers.models.qwen2.modeling_qwen2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "qwen2_moe":
        transformers.models.qwen2_moe.modeling_qwen2_moe._get_unpad_data = (  # pylint: disable=protected-access
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
    elif model_type == "gemma2":
        transformers.models.gemma2.modeling_gemma2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
    elif model_type == "starcoder2":
        transformers.models.starcoder2.modeling_starcoder2._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )


def patch_remote(model_name, config_name, modeling_name):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # we need to load the model here in order for modeling_* to be available
    with init_empty_weights():
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    module_name = model_config.__class__.__module__.replace(config_name, modeling_name)
    modeling_arch = importlib.import_module(module_name)
    modeling_arch._get_unpad_data = get_unpad_data  # pylint: disable=protected-access
