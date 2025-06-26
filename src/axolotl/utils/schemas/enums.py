"""Enums for Axolotl input config"""

# pylint: disable=invalid-name

from enum import Enum

import torch


class TorchIntDType(Enum):
    """Torch integer data types - `getattr` guards against torch < 2.6 which does not support int4"""

    uint1 = getattr(torch, "uint1", None)
    uint2 = getattr(torch, "uint2", None)
    uint3 = getattr(torch, "uint3", None)
    uint4 = getattr(torch, "uint4", None)
    uint5 = getattr(torch, "uint5", None)
    uint6 = getattr(torch, "uint6", None)
    uint7 = getattr(torch, "uint7", None)
    int4 = getattr(torch, "int4", None)
    int8 = getattr(torch, "int8", None)


class RLType(str, Enum):
    """RL trainer type configuration subset"""

    DPO = "dpo"
    GRPO = "grpo"
    IPO = "ipo"
    ORPO = "orpo"
    KTO = "kto"
    SIMPO = "simpo"


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
    falcon_h1 = "falcon_h1"
    tokenizer_default = "tokenizer_default"
    exaone = "exaone"
    metharme = "metharme"
    pixtral = "pixtral"
    llava = "llava"
    qwen2_vl = "qwen2_vl"
    gemma3 = "gemma3"
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


class RingAttnFunc(str, Enum):
    """Enum class for supported `ring-flash-attn` implementations"""

    VARLEN_LLAMA3 = "varlen_llama3"
    BATCH_RING = "batch_ring"
    # VARLEN_RING = "varlen_ring"
    # VARLEN_ZIGZAG = "varlen_zigzag"
    # BATCH_ZIGZAG = "batch_zigzag"
    # BATCH_STRIPE = "batch_stripe"
