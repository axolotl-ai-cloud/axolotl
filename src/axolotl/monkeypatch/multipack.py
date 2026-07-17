"""multipack patching for v2 of sample packing"""

import importlib

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import is_deepspeed_zero3_enabled

from axolotl.monkeypatch.mixtral import patch_mixtral_moe_forward_zero3
from axolotl.monkeypatch.utils import get_unpad_data

SUPPORTED_MULTIPACK_MODEL_TYPES = [
    "apertus",
    "mllama_text_model",
    "llama",
    "llama4",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_5",
    "qwen3_5_moe",
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
    "glm",
    "glm4",
    "glm4_moe",
    "glm_moe_dsa",
    "smollm3",
    "granite",
    "granitemoe",
    "granitemoeshared",
    "granitemoehybrid",
    "hunyuan_v1_dense",
    "hunyuan_v1_moe",
    "gpt_oss",
    "arcee",
    "seed_oss",
    "lfm2",
    "lfm2_moe",
    "ernie4_5",
    "ernie4_5_moe",
    "olmo",
    "olmo2",
    "olmo3",
    "ministral",
    "ministral3",
    "mistral4",
    "afmoe",
    "nemotron",
    "nemotron_h",
    "falcon_h1",
    "minimax_m2",
]


def patch_for_multipack(model_type, model_name=None, has_remote_code=False):
    # In-tree HF models handle sample packing natively via position_ids
    # (transformers `_is_packed_sequence` / `find_packed_sequence_indices`): the 2D
    # mask is cast to bool before flash attention, so the segment ids we encode in
    # it never reach `_get_unpad_data` and overriding it there is a no-op. Only
    # remote-code modeling files that call `_get_unpad_data` directly on the raw
    # segment-id mask still need the override.
    if has_remote_code:
        patch_remote(model_name)

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
        modeling_arch._get_unpad_data = get_unpad_data
