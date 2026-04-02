"""
Supported MoE block mappings for kernel integrations.

Maps model_type to the SparseMoeBlock class name(s) in transformers.
Used by both ScatterMoE and SonicMoE kernel paths.

Values can be a single class name (str) or a list of class names for models
with multiple MoE block types (e.g. qwen3_omni_moe has Thinker + Talker).

Models with custom routing (see sonicmoe/routing.py for implementations):
- ernie4_5_moe: softmax→bias correction→topk (softmax_bias_topk_routing)
- deepseek_v2: softmax→group_limited_greedy (softmax_group_limited_topk_routing)
- hunyuan_v1_moe: softmax→topk via gate.wg (softmax_topk_wg_routing)
"""

import importlib

SPARSE_MOE_BLOCK = {
    # softmax -> topk routing
    "qwen2_moe": "Qwen2MoeSparseMoeBlock",
    "qwen3_moe": "Qwen3MoeSparseMoeBlock",
    "qwen3_5_moe": "Qwen3_5MoeSparseMoeBlock",
    "qwen3_5_moe_text": "Qwen3_5MoeSparseMoeBlock",
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
    # softmax -> topk routing (with group-based expert selection)
    "mistral4": "Mistral4MoE",
    # sigmoid -> topk routing (with group-based expert selection)
    "glm_moe_dsa": "GlmMoeDsaMoE",
    "deepseek_v3": "DeepseekV3MoE",
    "glm4_moe": "Glm4MoeMoE",
    "glm4_moe_lite": "Glm4MoeLiteMoE",
    "glm4v_moe": "Glm4vMoeTextMoE",
    # sigmoid -> topk routing (no group selection)
    "minimax_m2": "MiniMaxM2SparseMoeBlock",
    # softmax->topk, e_score_correction_bias between softmax and topk
    "ernie4_5_moe": "Ernie4_5_MoeSparseMoeBlock",
    # softmax->topk, group_limited_greedy, different attr names (num_group)
    "deepseek_v2": "DeepseekV2Moe",
    # softmax->topk, gate.wg (not gate.weight)
    "hunyuan_v1_moe": "HunYuanMoEV1Moe",
    # TODO: gpt_oss deferred — transposed weight layout [E,H,2*I], expert biases,
    # and custom GLU activation require a dedicated forward path in patch.py.
    # "gpt_oss": "GptOssMLP",
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
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        # Text sub-model types (e.g. qwen3_5_moe_text) share the parent module
        if model_type.endswith("_text"):
            parent_type = model_type.removesuffix("_text")
            module_path = f"transformers.models.{parent_type}.modeling_{parent_type}"
            module = importlib.import_module(module_path)
        else:
            raise

    classes = []
    for cls_name in cls_names:
        moe_cls = getattr(module, cls_name, None)
        if moe_cls is None:
            raise ValueError(f"Could not find class '{cls_name}' in '{module_path}'")
        classes.append(moe_cls)

    return classes
