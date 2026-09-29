"""K2-Horizon (IFM) model support.

The published remote code trains as-is: it uses the transformers attention
interface, so sample packing flows through ``position_ids`` like an in-tree
model. The descriptor only declares which fused-kernel features cannot serve
its grouped RMSNorm, partial-RoPE gated attention, and MoVA value experts.
"""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import ModelProfile
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


@register_model_support
class K2HorizonSupport(ModelSupport):
    """Descriptor for K2-Horizon (`k2_horizon`) dense and MoVA MoE checkpoints."""

    model_types = ("k2_horizon",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities={
            "cut_cross_entropy": Unsupported(
                "The generic CCE patch imports transformers.models.<model_type>, "
                "which a remote-code model does not have."
            ),
            "liger": Unsupported(
                "Liger has no k2_horizon entry, and LigerRMSNorm would drop the "
                "layernorm_num_groups grouping of K2HorizonRMSNorm."
            ),
            "lora_kernels": Unsupported(
                "The fused QKV/O rewrite does not match K2HorizonAttention's forward: "
                "grouped q/k norm, a partial-RoPE split, an optional softplus gate, and "
                "MoVA layers route values through v_experts instead of v_proj."
            ),
        },
    )
