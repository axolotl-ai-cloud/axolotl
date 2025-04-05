"""multipack patching for v2 of sample packing"""

import importlib

import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import is_deepspeed_zero3_enabled

from axolotl.monkeypatch.mixtral import patch_mixtral_moe_forward_zero3
from axolotl.monkeypatch.utils import get_unpad_data

SUPPORTED_MULTIPACK_MODEL_TYPES = [
    "mllama_text_model",
    "llama",
    "llama4",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "falcon",
    "phi",
    "phi3",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "cohere",
    "cohere2",
    "gemmoe",
    "starcoder2",
    "deepseek_v2",
    "deepseek_v3",
]


def patch_for_multipack(model_type, model_name=None, has_remote_code=False):
    if has_remote_code:
        patch_remote(model_name)
    elif hasattr(transformers, "modeling_flash_attention_utils"):
        transformers.modeling_flash_attention_utils._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )

    if model_type == "mixtral" and is_deepspeed_zero3_enabled():
        patch_mixtral_moe_forward_zero3()


def patch_remote(model_name):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # we need to load the model here in order for modeling_* to be available
    with init_empty_weights():
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    parts = model_config.__class__.__module__.split(".")
    parts[-1] = parts[-1].replace("configuration_", "modeling_", 1)
    module_name = ".".join(parts)
    modeling_arch = importlib.import_module(module_name)
    if hasattr(modeling_arch, "_get_unpad_data"):
        modeling_arch._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )
