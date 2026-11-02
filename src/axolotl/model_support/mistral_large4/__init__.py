"""Mistral Large 4 model support (MLA + DSA sparse attention + MoE)."""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT


def _before_model_build(context: ModelHookContext) -> None:
    from axolotl.monkeypatch.models.mistral_large4.conversion_mapping import (
        patch_mistral_large4_moe_renamings,
    )

    patch_mistral_large4_moe_renamings()


@register_model_support
class MistralLarge4Support(ModelSupport):
    """Descriptor for mistral_large4 (Mistral Large 4)."""

    model_types = ("mistral_large4", "mistral_large4_text")
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            "lora_kernels": Unsupported(
                "MLA has no q/k/v projection triple for the fused QKV/O LoRA kernels."
            ),
            "sdpa_varlen": Unsupported(
                "The DSA indexer builds its per-query top-k mask from the attention mask, "
                "which the varlen path drops."
            ),
            "context_parallel": Unsupported(
                "The DSA indexer only sees the local shard's keys under context parallelism."
            ),
        },
        hooks=ModelHooks(
            by_phase={ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,)}
        ),
    )
