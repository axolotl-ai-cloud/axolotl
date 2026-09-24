"""Reflection (Beam) model support.

The architecture ships in Reflection's transformers fork, so config, tokenizer
and modeling code all load natively -- nothing here redirects dynamic modules.
What this descriptor carries is the set of features whose generic fallbacks are
wrong for this arch, so enabling one fails loudly instead of training quietly on
a different objective.
"""

from axolotl.model_support.base import Experimental, ModelSupport, Unsupported
from axolotl.model_support.profile import ModelProfile
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


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
            "lora_kernels": Unsupported(
                "ReflectionAttention adds an output gate (self_attn.gate_proj plus "
                "gate_bias) and q_norm/k_norm on top of q/k/v/o. The fused QKV/O "
                "kernels match the llama-like projection set by name and would "
                "bypass the gate entirely."
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
    )
