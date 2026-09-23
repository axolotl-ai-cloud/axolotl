"""Qwen3.5-MoE model support (hybrid Gated DeltaNet + attention, 256 routed experts).

Also covers derivatives that keep the architecture but ship their own chat
template (e.g. Nex-N2.5), so the multimodal collator dispatches on
``model_type`` rather than on ``chat_template: qwen3_5``.
"""

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import ModelProfile, ModelStrategyOverrides
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT


def _get_processing_strategy_cls() -> type:
    from axolotl.processing_strategies import Qwen3_5ProcessingStrategy

    return Qwen3_5ProcessingStrategy


@register_model_support
class Qwen3_5MoeSupport(ModelSupport):
    """Descriptor for Qwen3.5-MoE (`qwen3_5_moe`) checkpoints."""

    # No processor matcher: Qwen3VLProcessor is shared with qwen3_vl(_moe) and
    # dense qwen3_5, so only model_type can tell the families apart.
    model_types = ("qwen3_5_moe",)
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        strategies=ModelStrategyOverrides(
            processing_strategy_cls=_get_processing_strategy_cls,
        ),
    )
