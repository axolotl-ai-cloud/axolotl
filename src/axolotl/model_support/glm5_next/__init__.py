"""GLM-5.3-Flash model support (hybrid KDA / DSA sparse MLA + MoE + mHC)."""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT


def _get_processing_strategy_cls() -> type:
    from axolotl.processing_strategies import Glm4vProcessingStrategy

    return Glm4vProcessingStrategy


def _before_model_build(context: ModelHookContext) -> None:
    if context.cfg.sample_packing:
        from axolotl.monkeypatch.models.glm5_next.modeling import (
            patch_glm5_next_modeling_packing,
        )

        patch_glm5_next_modeling_packing()


@register_model_support
class Glm5NextSupport(ModelSupport):
    """Descriptor for glm5_next (GLM-5.3-Flash)."""

    model_types = ("glm5_next", "glm5_next_text")
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            "cut_cross_entropy": Unsupported(
                "ml-cross-entropy has no glm5_next forward, and the generic patch "
                "targets a ForCausalLM class this model does not have."
            ),
            "lora_kernels": Unsupported(
                "The KDA layers are named self_attn with q/k/v/o_proj, so the fused "
                "QKV/O rewrite matches on name but not on the KDA forward."
            ),
            "sdpa_varlen": Unsupported(
                "The DSA layers attend through the indexer's per-query top-k mask, "
                "which varlen attention would drop."
            ),
        },
        strategies=ModelStrategyOverrides(
            processing_strategy_cls=_get_processing_strategy_cls,
        ),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
            }
        ),
    )
