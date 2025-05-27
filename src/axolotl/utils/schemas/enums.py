"""Enums for Axolotl input config"""

from enum import Enum

import torch
from packaging import version

torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
if version.parse(torch_version) < version.parse("2.6.0"):
    # no-op - config validation should handle erroring out. this guards a `torch.int4` import being
    # unavailable in torch < 2.6
    class TorchIntDType(Enum):
        """Dummy class for import guarding"""

else:

    class TorchIntDType(Enum):
        """Torch integer data types"""

        uint1 = torch.uint1  # pylint: disable=invalid-name
        uint2 = torch.uint2  # pylint: disable=invalid-name
        uint3 = torch.uint3  # pylint: disable=invalid-name
        uint4 = torch.uint4  # pylint: disable=invalid-name
        uint5 = torch.uint5  # pylint: disable=invalid-name
        uint6 = torch.uint6  # pylint: disable=invalid-name
        uint7 = torch.uint7  # pylint: disable=invalid-name
        int4 = torch.int4  # pylint: disable=invalid-name
        int8 = torch.int8  # pylint: disable=invalid-name


class RLType(str, Enum):
    """RL trainer type configuration subset"""

    DPO = "dpo"  # pylint: disable=invalid-name
    GRPO = "grpo"  # pylint: disable=invalid-name
    IPO = "ipo"  # pylint: disable=invalid-name
    ORPO = "orpo"  # pylint: disable=invalid-name
    KTO = "kto"  # pylint: disable=invalid-name
    SIMPO = "simpo"  # pylint: disable=invalid-name


class ChatTemplate(str, Enum):
    """Chat templates configuration subset"""

    alpaca = "alpaca"  # pylint: disable=invalid-name
    chatml = "chatml"  # pylint: disable=invalid-name
    mistral_v1 = "mistral_v1"  # pylint: disable=invalid-name
    mistral_v2v3 = "mistral_v2v3"  # pylint: disable=invalid-name
    mistral_v3_tekken = "mistral_v3_tekken"  # pylint: disable=invalid-name
    mistral_v7_tekken = "mistral_v7_tekken"  # pylint: disable=invalid-name
    gemma = "gemma"  # pylint: disable=invalid-name
    cohere = "cohere"  # pylint: disable=invalid-name
    llama3 = "llama3"  # pylint: disable=invalid-name
    llama3_2_vision = "llama3_2_vision"  # pylint: disable=invalid-name
    llama4 = "llama4"  # pylint: disable=invalid-name
    phi_3 = "phi_3"  # pylint: disable=invalid-name
    phi_35 = "phi_35"  # pylint: disable=invalid-name
    deepseek_v2 = "deepseek_v2"  # pylint: disable=invalid-name
    deepseek_v3 = "deepseek_v3"  # pylint: disable=invalid-name
    jamba = "jamba"  # pylint: disable=invalid-name
    jinja = "jinja"  # pylint: disable=invalid-name
    qwen_25 = "qwen_25"  # pylint: disable=invalid-name
    qwen3 = "qwen3"  # pylint: disable=invalid-name
    tokenizer_default = "tokenizer_default"  # pylint: disable=invalid-name
    exaone = "exaone"  # pylint: disable=invalid-name
    metharme = "metharme"  # pylint: disable=invalid-name
    pixtral = "pixtral"  # pylint: disable=invalid-name
    llava = "llava"  # pylint: disable=invalid-name
    qwen2_vl = "qwen2_vl"  # pylint: disable=invalid-name
    gemma3 = "gemma3"  # pylint: disable=invalid-name


class CustomSupportedOptimizers(str, Enum):
    """Custom supported optimizers"""

    optimi_adamw = "optimi_adamw"  # pylint: disable=invalid-name
    ao_adamw_4bit = "ao_adamw_4bit"  # pylint: disable=invalid-name
    ao_adamw_8bit = "ao_adamw_8bit"  # pylint: disable=invalid-name
    ao_adamw_fp8 = "ao_adamw_fp8"  # pylint: disable=invalid-name
    adopt_adamw = "adopt_adamw"  # pylint: disable=invalid-name
    came_pytorch = "came_pytorch"  # pylint: disable=invalid-name
    muon = "muon"  # pylint: disable=invalid-name


class RingAttnFunc(str, Enum):
    """Enum class for supported `ring-flash-attn` implementations"""

    # VARLEN_RING = "varlen_ring"
    # VARLEN_ZIGZAG = "varlen_zigzag"
    VARLEN_LLAMA3 = "varlen_llama3"
    BATCH_RING = "batch_ring"
    # BATCH_ZIGZAG = "batch_zigzag"
    # BATCH_STRIPE = "batch_stripe"
