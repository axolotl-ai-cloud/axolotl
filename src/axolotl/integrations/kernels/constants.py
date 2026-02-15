"""
Supported MoE block mappings for kernel integrations.

Maps model_type to the SparseMoeBlock class name in transformers.
Used by both ScatterMoE and SonicMoE kernel paths.
"""

import importlib

SPARSE_MOE_BLOCK = {
    # softmax -> topk routing
    "qwen2_moe": "Qwen2MoeSparseMoeBlock",
    "qwen3_moe": "Qwen3MoeSparseMoeBlock",
    "qwen3_next": "Qwen3NextSparseMoeBlock",
    "olmoe": "OlmoeSparseMoeBlock",
    "mixtral": "MixtralSparseMoeBlock",
    "minimax": "MiniMaxSparseMoeBlock",
    # sigmoid -> topk routing (with group-based expert selection)
    "glm_moe_dsa": "GlmMoeDsaMoE",
    "deepseek_v3": "DeepseekV3MoE",
    "glm4_moe": "Glm4MoeMoE",
    "glm4_moe_lite": "Glm4MoeLiteMoE",
    "glm4v_moe": "Glm4vMoeTextMoE",
    # sigmoid -> topk routing (no group selection)
    "minimax_m2": "MiniMaxM2SparseMoeBlock",
    # Models below need custom routing (not yet implemented):
    # "ernie4_5_moe": "Ernie4_5_MoeSparseMoeBlock",  # softmax->topk, e_score_correction_bias between softmax and topk
    # "deepseek_v2": "DeepseekV2Moe",  # softmax->topk, group_limited_greedy, different attr names (num_group)
    # "hunyuan_v1_moe": "HunYuanMoEV1Moe",  # softmax->topk, gate.wg (not gate.weight), scatter routing
    # "gpt_oss": "GptOssMLP",  # topk->softmax, transposed layout [E,H,2*I], custom GLU, expert biases
}


def resolve_moe_block_cls(model_type: str):
    """Resolve the MoE block class from transformers for the given model type."""
    cls_name = SPARSE_MOE_BLOCK.get(model_type)
    if cls_name is None:
        raise ValueError(
            f"Unsupported MoE model type '{model_type}'. "
            f"Supported types: {list(SPARSE_MOE_BLOCK.keys())}"
        )

    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    module = importlib.import_module(module_path)
    moe_cls = getattr(module, cls_name, None)
    if moe_cls is None:
        raise ValueError(f"Could not find class '{cls_name}' in '{module_path}'")

    return moe_cls
