# Why: patchify Conv3d (kernel_size == stride, no padding) is exactly F.linear
# over flat patches; ~11x faster than cuDNN conv3d, no slow_conv_dilated3d cliff.

import importlib

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.utils.logging import get_logger

logger = get_logger(__name__)

# model_config_type -> (modeling module, class); all build Conv3d with stride == kernel_size
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
    proj = self.proj
    if type(proj) is not nn.Conv3d:
        # PEFT-wrapped proj (LoRA / ModulesToSaveWrapper): keep its forward in the path
        return self._axolotl_patch_embed_original_forward(hidden_states)
    weight = proj.weight
    # input flat layout matches the (D, C, T, Ph, Pw) weight layout element-for-element
    hidden_states = hidden_states.reshape(-1, weight[0].numel()).to(dtype=weight.dtype)
    return F.linear(hidden_states, weight.reshape(weight.shape[0], -1), proj.bias)


def _resolve_class(model_config_type: str | None):
    entry = SUPPORTED_PATCH_EMBEDS.get(model_config_type or "")
    if entry is None:
        return None
    module_path, class_name = entry
    return getattr(importlib.import_module(module_path), class_name)


def patch_vision_patch_embed_linear(model_config_type: str | None) -> None:
    cls = _resolve_class(model_config_type)
    if cls is None or getattr(cls, "_axolotl_patch_embed_linear", False):
        return

    cls._axolotl_patch_embed_original_forward = cls.forward
    cls.forward = _linear_forward
    cls._axolotl_patch_embed_linear = True
    logger.info(
        "Patched %s.forward to the equivalent patchify GEMM (F.linear)", cls.__name__
    )


def unpatch_vision_patch_embed_linear(model_config_type: str | None) -> None:
    cls = _resolve_class(model_config_type)
    if cls is None:
        return
    original = cls.__dict__.get("_axolotl_patch_embed_original_forward")
    if original is None:
        return

    cls.forward = original
    del cls._axolotl_patch_embed_original_forward
    if "_axolotl_patch_embed_linear" in cls.__dict__:
        del cls._axolotl_patch_embed_linear
    logger.info("Restored %s.forward to the stock Conv3d path", cls.__name__)
