"""
Supported MoE block mappings for kernel integrations.

Maps model_type to the SparseMoeBlock class name(s) in transformers.
Used by both ScatterMoE and SonicMoE kernel paths.

Values can be a single class name (str) or a list of class names for models
with multiple MoE block types (e.g. qwen3_omni_moe has Thinker + Talker).
"""

import importlib

SPARSE_MOE_BLOCK = {
    # softmax -> topk routing
    "qwen2_moe": "Qwen2MoeSparseMoeBlock",
    "qwen3_moe": "Qwen3MoeSparseMoeBlock",
    "qwen3_5_moe": "Qwen3_5MoeSparseMoeBlock",
    "qwen3_next": "Qwen3NextSparseMoeBlock",
    "qwen3_vl_moe": "Qwen3VLMoeTextSparseMoeBlock",
    # qwen3_omni_moe: Thinker (standard) + Talker (shared experts + shared_expert_gate)
    "qwen3_omni_moe": [
        "Qwen3OmniMoeThinkerTextSparseMoeBlock",
        "Qwen3OmniMoeTalkerTextSparseMoeBlock",
    ],
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


def resolve_moe_block_classes(model_type: str):
    """Resolve all MoE block classes from transformers for the given model type.

    Returns a list of classes (one for most models, multiple for models with
    distinct MoE block types like qwen3_omni_moe).
    """
    entry = SPARSE_MOE_BLOCK.get(model_type)
    if entry is None:
        raise ValueError(
            f"Unsupported MoE model type '{model_type}'. "
            f"Supported types: {list(SPARSE_MOE_BLOCK.keys())}"
        )

    cls_names = entry if isinstance(entry, list) else [entry]
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    module = importlib.import_module(module_path)

    classes = []
    for cls_name in cls_names:
        moe_cls = getattr(module, cls_name, None)
        if moe_cls is None:
            raise ValueError(f"Could not find class '{cls_name}' in '{module_path}'")
        classes.append(moe_cls)

    return classes
