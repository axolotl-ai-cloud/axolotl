"""Enums for Axolotl input config"""

from enum import Enum

import torch


class TorchAOQuantDType(Enum):
    int4 = torch.int4
    int8 = torch.int8
    float8_e4m3fn = torch.float8_e4m3fn
    nvfp4 = "nvfp4"
    mxfp4 = "mxfp4"

    def from_string(str):
        if str == "int4":
            return TorchAOQuantDType.int4
        if str == "int8":
            return TorchAOQuantDType.int8
        if str in ["float8_e4m3fn", "fp8", "float8"]:
            return TorchAOQuantDType.float8_e4m3fn
        if str == "nvfp4":
            return TorchAOQuantDType.nvfp4
        if str == "mxfp4":
            return TorchAOQuantDType.mxfp4


class RLType(str, Enum):
    """RL trainer type configuration subset"""

    DPO = "dpo"
    GDPO = "gdpo"
    GRPO = "grpo"
    IPO = "ipo"
    ORPO = "orpo"
    KTO = "kto"
    SIMPO = "simpo"
    EBFT = "ebft"


class ChatTemplate(str, Enum):
    """Chat templates configuration subset"""

    alpaca = "alpaca"
    chatml = "chatml"
    mistral_v1 = "mistral_v1"
    mistral_v2v3 = "mistral_v2v3"
    mistral_v3_tekken = "mistral_v3_tekken"
    mistral_v7_tekken = "mistral_v7_tekken"
    gemma = "gemma"
    cohere = "cohere"
    llama3 = "llama3"
    llama3_2_vision = "llama3_2_vision"
    llama4 = "llama4"
    phi_3 = "phi_3"
    phi_35 = "phi_35"
    deepseek_v2 = "deepseek_v2"
    deepseek_v3 = "deepseek_v3"
    jamba = "jamba"
    jinja = "jinja"
    qwen_25 = "qwen_25"
    qwen3 = "qwen3"
    qwen3_5 = "qwen3_5"
    falcon_h1 = "falcon_h1"
    nemotron_h = "nemotron_h"
    tokenizer_default = "tokenizer_default"
    exaone = "exaone"
    exaone4 = "exaone4"
    metharme = "metharme"
    pixtral = "pixtral"
    llava = "llava"
    qwen2_vl = "qwen2_vl"
    gemma3 = "gemma3"
    gemma3n = "gemma3n"
    gemma4 = "gemma4"
    command_a = "command_a"
    command_a_tool_use = "command_a_tool_use"
    command_a_rag = "command_a_rag"
    aya = "aya"


class CustomSupportedOptimizers(str, Enum):
    """Custom supported optimizers"""

    optimi_adamw = "optimi_adamw"
    ao_adamw_4bit = "ao_adamw_4bit"
    ao_adamw_8bit = "ao_adamw_8bit"
    ao_adamw_fp8 = "ao_adamw_fp8"
    adopt_adamw = "adopt_adamw"
    came_pytorch = "came_pytorch"
    muon = "muon"
    dion = "dion"
    flash_adamw = "flash_adamw"
    flash_adam = "flash_adam"
    flash_sgd = "flash_sgd"
    flash_sgdw = "flash_sgdw"
    flash_lion = "flash_lion"


# Canonical values accepted for `attn_implementation`. These are passed to HF
# verbatim via `model.config._attn_implementation`. HF-native backends use HF's
# own names (`flash_attention_2`, `flex_attention`, ...); axolotl-owned backends
# (`xformers`, `sage`, `s2`, `fp8`) register into `ALL_ATTENTION_FUNCTIONS` under
# these exact names. Hub-kernel paths (e.g. `kernels-community/flash-attn3`) are
# not in this set — they pass through the validator via the "/" check.
CANONICAL_ATTN_IMPLS = frozenset(
    {
        "eager",
        "sdpa",
        "flash_attention_2",
        "flash_attention_3",
        "flex_attention",
        "xformers",
        "sage",
        "s2",
        "fp8",
    }
)

# Legacy boolean attention flags → canonical `attn_implementation`. Kept for
# backwards compatibility; the normalizer warns and strips these from the
# validated config. Priority order (first match wins) matches the old priority:
# specific backends beat the generic flash/sdp/eager fallbacks.
LEGACY_ATTN_FLAG_TO_IMPL = {
    "xformers_attention": "xformers",
    "s2_attention": "s2",
    "sage_attention": "sage",
    "flex_attention": "flex_attention",
    "flash_attention": "flash_attention_2",
    "sdp_attention": "sdpa",
    "eager_attention": "eager",
}

# Short-form aliases that were accepted by the in-progress branch but are
# rejected going forward. Mapped to canonical names only to produce a helpful
# error message pointing users at the right value.
SHORT_FORM_ALIAS_TO_CANONICAL = {
    "flash": "flash_attention_2",
    "flex": "flex_attention",
    "sdp": "sdpa",
}

# Backends that support varlen sample packing via `position_ids`.
ATTN_IMPLS_SUPPORTING_PACKING = frozenset(
    {
        "flash_attention_2",
        "flash_attention_3",
        "flex_attention",
        "xformers",
        "sage",
        "kernels-community/flash-attn3",
        "kernels-community/sage-attention",
    }
)

# Backends that require the flash_attn library (Dao-AILab/flash-attention) for
# axolotl's own monkeypatches (FA4 auto-apply, LLaMA flash hijack, ring-FA, ...).
ATTN_IMPLS_USING_FLASH_LIB = frozenset(
    {
        "flash_attention_2",
        "flash_attention_3",
        "s2",
        "kernels-community/flash-attn3",
    }
)

# Backends for which embeddings stay in fp32. Everything else needs fp16/bf16.
ATTN_IMPLS_WITHOUT_DTYPE_CAST = frozenset({"eager", "sdpa"})


class RingAttnFunc(str, Enum):
    """Enum class for supported `ring-flash-attn` implementations"""

    VARLEN_LLAMA3 = "varlen_llama3"
    BATCH_RING = "batch_ring"
    # VARLEN_RING = "varlen_ring"
    # VARLEN_ZIGZAG = "varlen_zigzag"
    # BATCH_ZIGZAG = "batch_zigzag"
    # BATCH_STRIPE = "batch_stripe"
