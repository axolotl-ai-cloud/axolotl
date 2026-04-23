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
    "phimoe": "PhiMoESparseMoeBlock",
    "qwen2_moe": "Qwen2MoeSparseMoeBlock",
    "qwen3_moe": "Qwen3MoeSparseMoeBlock",
    "qwen3_5_moe": "Qwen3_5MoeSparseMoeBlock",
    "qwen3_vl_moe": "Qwen3VLMoeTextSparseMoeBlock",
    "deepseek_v2": "DeepseekV2MoE",
    "deepseek_v3": "DeepseekV3MoE",
    "mistral4": "Mistral4MoE",
    "gpt_oss": "GptOssDecoderLayer",
    "lfm2_moe": "Lfm2MoeSparseMoeBlock",
    "afmoe": "AfmoeMoE",
    "glm4_moe": "Glm4MoeDecoderLayer",
    "glm4_moe_lite": "Glm4MoeLiteDecoderLayer",
    "glm_moe_dsa": "GlmMoeDsaDecoderLayer",
    "nemotron_h": "NemotronHMoE",
}
