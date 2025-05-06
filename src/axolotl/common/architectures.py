"""
Common architecture specific constants
"""

MOE_ARCH_BLOCK = {
    "dbrx": "DbrxFFN",
    "jamba": "JambaSparseMoeBlock",
    "jetmoe": [
        "JetMoeMoA",
        "JetMoeMoE",
    ],
    "mixtral": "MixtralSparseMoeBlock",
    "qwen2_moe": "Qwen2MoeSparseMoeBlock",
    "qwen3_moe": "Qwen3MoeSparseMoeBlock",
    "deepseek_v2": "DeepseekV2MoE",
}
