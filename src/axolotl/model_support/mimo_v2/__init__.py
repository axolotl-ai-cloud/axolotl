"""MiMo-V2.6 (`mimo_v2`) MoE model support.

The checkpoint's remote code is inference-only, so the pre-load hooks serve its
module names from this package, which maps the text decoder onto transformers'
native MiMo-V2-Flash modules. The registered conversions dequantize the
FP8/MXFP4 checkpoint, split the fused ``qkv_proj`` and fuse the per-expert weights.
"""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelRegistrationOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM

MIMO_V2_PACKAGE = "axolotl.model_support.mimo_v2"
MIMO_V2_MODULES = ("configuration_mimo_v2", "modeling_mimo_v2")


def _matches_mimo_v2_cfg(cfg) -> bool:
    return any(
        "mimo-v2.6" in (getattr(cfg, field, None) or "").lower()
        for field in ("base_model_config", "tokenizer_config")
    )


def _redirect_remote_code(_context: ModelHookContext) -> None:
    from axolotl.model_support.remote_code import redirect_dynamic_modules

    redirect_dynamic_modules(MIMO_V2_PACKAGE, MIMO_V2_MODULES)


def _weight_conversions():
    from transformers.core_model_loading import (
        Concatenate,
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
    )

    from .modeling_mimo_v2 import MiMoV2Dequantize, SplitQKV

    def dequant(*names):
        return [f"{n}.weight{s}$" for n in names for s in ("", "_scale_inv")]

    experts = "mlp.experts.*"
    return {
        "mimo_v2": [
            WeightRenaming(r"\.weight_scale$", ".weight_scale_inv"),
            WeightRenaming("self_attn.attention_sink_bias", "self_attn.sinks"),
            WeightConverter(
                dequant("self_attn.qkv_proj"),
                [f"self_attn.{p}_proj.weight" for p in "qkv"],
                operations=[MiMoV2Dequantize(), SplitQKV()],
            ),
            WeightConverter(
                dequant(f"{experts}.gate_proj", f"{experts}.up_proj"),
                "mlp.experts.gate_up_proj",
                operations=[
                    MiMoV2Dequantize(),
                    MergeModulelist(dim=0),
                    Concatenate(dim=1),
                ],
            ),
            WeightConverter(
                dequant(f"{experts}.down_proj"),
                "mlp.experts.down_proj",
                operations=[MiMoV2Dequantize(), MergeModulelist(dim=0)],
            ),
            # first-match fallback for the remaining FP8 linears (dense MLP)
            WeightConverter(
                ["weight$", "weight_scale_inv"],
                "weight",
                operations=[MiMoV2Dequantize()],
            ),
        ]
    }


@register_model_support
class MiMoV2Support(ModelSupport):
    """Descriptor for MiMo-V2.6 (`mimo_v2`) sparse MoE checkpoints, text only."""

    model_types = ("mimo_v2",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities={
            "cut_cross_entropy": Unsupported(
                "The generic CCE patch imports transformers.models.mimo_v2, "
                "which does not exist."
            ),
            "lora_kernels": Unsupported(
                "The fused QKV/O rewrite cannot resolve an attention class for mimo_v2."
            ),
        },
        matchers=ModelMatchers(cfg=_matches_mimo_v2_cfg),
        registrations=ModelRegistrationOverrides(
            weight_conversions=_weight_conversions
        ),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_CONFIG_LOAD: (_redirect_remote_code,),
                ModelHookPhase.BEFORE_TOKENIZER_LOAD: (_redirect_remote_code,),
                ModelHookPhase.BEFORE_MODEL_BUILD: (_redirect_remote_code,),
            }
        ),
    )
