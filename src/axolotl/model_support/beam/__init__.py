"""Reflection (Beam) model support.

The architecture ships in Reflection's transformers fork, so config, tokenizer
and modeling code all load natively -- nothing here redirects dynamic modules.
What this descriptor carries is the set of features whose generic fallbacks are
wrong for this arch, so enabling one fails loudly instead of training quietly on
a different objective.
"""

from axolotl.model_support.base import Experimental, ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


def _register_cce_patch(_context: ModelHookContext) -> None:
    """Point cut_cross_entropy's registry at the patch in this package.

    Runs before the CCE plugin's pre_model_load, which resolves PATCH_FNS by
    string import, so the entry only has to exist by then.
    """
    try:
        from cut_cross_entropy.transformers.patch import PATCH_FNS
    except ImportError:
        return

    spec = ("axolotl.model_support.beam.cut_cross_entropy", "patch_beam")
    PATCH_FNS.setdefault("beam", spec)
    PATCH_FNS.setdefault(
        "reflection",
        ("axolotl.model_support.beam.cut_cross_entropy", "patch_reflection"),
    )


@register_model_support
class BeamSupport(ModelSupport):
    """Descriptor for Beam (`reflection` / `beam`) (ReflectionForCausalLM): hybrid
    sliding/full attention with gated output, DeepSeek-V3-style MoE routing,
    and experts fused into 3D ``nn.Parameter`` tensors."""

    # "reflection" in Reflection's transformers fork, "beam" upstream
    model_types = ("reflection", "beam")
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities={
            "cut_cross_entropy": Experimental(
                "Needs the beam patch registered in ml-cross-entropy; apply_lce "
                "takes the logit softcap, but the fp32 lm_head path is unverified."
            ),
            "lora_kernels": Experimental(
                "The fused QKV/O patch rewrites only the q/k/v projection lines "
                "(the q_norm/k_norm variant) and the o_proj call; the attention output "
                "gate, query scaling and sandwich norms stay in the surrounding forward. "
                "Loss parity was checked on a random real-width Beam, not on the 501B."
            ),
            "liger": Unsupported(
                "enable_sandwich_norm adds attn_output_layernorm/mlp_output_layernorm "
                "and the residuals are depth-scaled (see depth_scaled_residual_coefs); "
                "Liger's RMSNorm and SwiGLU replacements carry neither, and its SwiGLU "
                "patch expects per-expert nn.Linear rather than fused 3D parameters."
            ),
            "sdpa_varlen": Experimental(
                "Unverified for this arch: sliding_window=512 needs a torch build with "
                "varlen_attn(window_size=...), and ssmax scales attention off "
                "position_ids -- which sample packing rewrites per document. Prefer "
                "attn_implementation: flash_attention_2 until this is measured."
            ),
        },
        hooks=ModelHooks(
            by_phase={ModelHookPhase.BEFORE_MODEL_BUILD: (_register_cce_patch,)}
        ),
    )
