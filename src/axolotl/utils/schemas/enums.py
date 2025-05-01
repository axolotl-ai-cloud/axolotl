"""Enums for Axolotl input config"""

from enum import Enum


class RLType(str, Enum):
    """RL trainer type configuration subset"""

    dpo = "dpo"  # pylint: disable=invalid-name
    grpo = "grpo"  # pylint: disable=invalid-name
    ipo = "ipo"  # pylint: disable=invalid-name
    orpo = "orpo"  # pylint: disable=invalid-name
    kto = "kto"  # pylint: disable=invalid-name
    simpo = "simpo"  # pylint: disable=invalid-name


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
    muon = "muon"  # pylint: disable=invalid-name
