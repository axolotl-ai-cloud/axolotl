# Why: these families' vision patch_embed Conv3d has kernel_size == stride and
# no padding, so it is exactly a linear projection over flattened patches (vLLM
# implements patchify the same way). The GEMM path (cuBLAS) is ~11x faster than
# cuDNN's conv3d for this shape and, unlike Conv3d, never dispatches to
# slow_conv_dilated3d on torch builds without cuDNN (measured 41 s vs 1.4 ms
# per 3 MP image). Weights stay in each family's Conv3d module, so state_dicts
# are unchanged in both directions.

import importlib

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

logger = get_logger(__name__)

# model_config_type -> (modeling module, patch-embed class). Every entry was
# verified to construct its Conv3d with stride == kernel_size.
SUPPORTED_PATCH_EMBEDS: dict[str, tuple[str, str]] = {
    "qwen3_vl": (
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "Qwen3VLVisionPatchEmbed",
    ),
    "qwen3_vl_moe": (
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeVisionPatchEmbed",
    ),
    "qwen3_5": (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "Qwen3_5VisionPatchEmbed",
    ),
    "qwen3_5_moe": (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeVisionPatchEmbed",
    ),
    "qwen3_omni_moe": (
        "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe",
        "Qwen3OmniMoeVisionPatchEmbed",
    ),
    "qwen2_5_vl": (
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "Qwen2_5_VisionPatchEmbed",
    ),
    "qwen2_vl": (
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "PatchEmbed",
    ),
    "qwen2_5_omni": (
        "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni",
        "Qwen2_5_VisionPatchEmbed",
    ),
    "glm4v": (
        "transformers.models.glm4v.modeling_glm4v",
        "Glm4vVisionPatchEmbed",
    ),
    "glm4v_moe": (
        "transformers.models.glm4v_moe.modeling_glm4v_moe",
        "Glm4vMoeVisionPatchEmbed",
    ),
    "glm_ocr": (
        "transformers.models.glm_ocr.modeling_glm_ocr",
        "GlmOcrVisionPatchEmbed",
    ),
}


def _linear_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = self.proj.weight
    # weight is (embed_dim, C, T, Ph, Pw); the stock forward views the input to
    # the same trailing dims, so the flat layouts match element-for-element.
    hidden_states = hidden_states.reshape(-1, weight[0].numel()).to(dtype=weight.dtype)
    return F.linear(hidden_states, weight.reshape(weight.shape[0], -1), self.proj.bias)


def patch_vision_patch_embed_linear(model_config_type: str | None) -> None:
    entry = SUPPORTED_PATCH_EMBEDS.get(model_config_type or "")
    if entry is None:
        return

    module_path, class_name = entry
    cls = getattr(importlib.import_module(module_path), class_name)
    if getattr(cls, "_axolotl_patch_embed_linear", False):
        return

    cls.forward = _linear_forward
    cls._axolotl_patch_embed_linear = True
    logger.info(
        "Patched %s.forward to the equivalent patchify GEMM (F.linear)", class_name
    )
