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
    "qwen3_vl_moe": "Qwen3VLMoeTextSparseMoeBlock",
    "deepseek_v2": "DeepseekV2MoE",
    "deepseek_v3": "DeepseekV3MoE",
    "gpt_oss": "GptOssDecoderLayer",
    "lfm2_moe": "Lfm2MoeSparseMoeBlock",
    "afmoe": "AfmoeMoE",
}
