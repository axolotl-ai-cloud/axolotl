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

# MoE architectures whose expert weights are 3D nn.Parameter tensors (not nn.Linear).
# BnB 4-bit quantization skips these by default, causing OOM. This mapping provides
# the parameter names needed for `target_parameters` in BitsAndBytesConfig or for
# post-load quantization via bitsandbytes.nn.parametrize.
# Verified against transformers 5.0.0 source.
MOE_EXPERT_PARAMS = {
    # gate_up_proj/down_proj pattern: (num_experts, 2*intermediate, hidden) / (num_experts, hidden, intermediate)
    "deepseek_v2": ["gate_up_proj", "down_proj"],
    "deepseek_v3": ["gate_up_proj", "down_proj"],
    "dots1": ["gate_up_proj", "down_proj"],
    "ernie4_5_moe": ["gate_up_proj", "down_proj"],
    "ernie4_5_vl_moe": ["gate_up_proj", "down_proj"],
    "flex_olmo": ["gate_up_proj", "down_proj"],
    "glm4_moe": ["gate_up_proj", "down_proj"],
    "glm4_moe_lite": ["gate_up_proj", "down_proj"],
    "glm4v_moe": ["gate_up_proj", "down_proj"],
    "hunyuan_v1_moe": ["gate_up_proj", "down_proj"],
    "jamba": ["gate_up_proj", "down_proj"],
    "lfm2_moe": ["gate_up_proj", "down_proj"],
    "llama4": ["gate_up_proj", "down_proj"],
    "longcat_flash": ["gate_up_proj", "down_proj"],
    "minimax": ["gate_up_proj", "down_proj"],
    "minimax_m2": ["gate_up_proj", "down_proj"],
    "mixtral": ["gate_up_proj", "down_proj"],
    "olmoe": ["gate_up_proj", "down_proj"],
    "phimoe": ["gate_up_proj", "down_proj"],
    "qwen2_moe": ["gate_up_proj", "down_proj"],
    "qwen3_moe": ["gate_up_proj", "down_proj"],
    "qwen3_next": ["gate_up_proj", "down_proj"],
    "qwen3_omni_moe": ["gate_up_proj", "down_proj"],
    "qwen3_vl_moe": ["gate_up_proj", "down_proj"],
    "solar_open": ["gate_up_proj", "down_proj"],
    # gate_up_proj/down_proj + bias params
    "gpt_oss": ["gate_up_proj", "down_proj"],
    # weight-only pattern: (num_experts, output_size, input_size)
    "jetmoe": ["weight"],
    "granitemoe": ["weight"],
    "granitemoehybrid": ["weight"],
    "granitemoeshared": ["weight"],
    # dbrx uses different param names: w1, v1, w2 (2D packed)
    "dbrx": ["w1", "v1", "w2"],
}
