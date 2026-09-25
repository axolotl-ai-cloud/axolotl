"""GLM-4 MoE Lite (GLM-4.7-Flash) model support."""

from axolotl.model_support.base import ModelSupport, Supported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


def _before_model_build(context: ModelHookContext) -> None:
    if not context.cfg.fused_attn_kernel:
        return

    import torch

    if torch.cuda.is_available():
        from axolotl.monkeypatch.models.glm4_moe_lite.fused_attn import (
            patch_glm4_moe_lite_fused_attn,
        )

        patch_glm4_moe_lite_fused_attn()


@register_model_support
class Glm4MoeLiteSupport(ModelSupport):
    """Capabilities and opt-in MLA attention patch for GLM-4.7-Flash."""

    model_types = ("glm4_moe_lite",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities={
            "liger": Supported(
                "Verified RMSNorm, rotary embeddings, and dense/shared-expert SwiGLU."
            ),
            "cut_cross_entropy": Supported(),
            "fused_attn_kernel": Supported(
                "CUDA partial RoPE and MLA Q/K/V assembly; cached decoding retains the original forward."
            ),
        },
        hooks=ModelHooks(
            by_phase={ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,)}
        ),
    )
