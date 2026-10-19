"""Beam model support.

The architecture loads natively through transformers, so config, tokenizer and
modeling code all resolve without redirecting dynamic modules. What this
descriptor carries is the set of features whose generic fallbacks are wrong for
this arch, so enabling one fails loudly instead of training quietly on a
different objective.
"""

from axolotl.model_support.base import (
    Experimental,
    ModelSupport,
    Supported,
    Unsupported,
)
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import VANILLA_CAUSAL_LM


def _cast_fp32_strict_params(context: ModelHookContext) -> None:
    """Keep the router (`mlp.gate.weight`, `e_score_correction_bias`) in fp32 on every rank.

    The modeling code lists them in ``_keep_in_fp32_modules_strict``, which only applies while real
    weights load (rank zero of the CPU-staged NF4 path). Ranks built on meta keep bf16, and the staged
    distribution plan then raises "rank zero has float32, this rank has bfloat16". Casting here, after
    the adapter wrap and before FSDP, makes the dtypes agree without touching the loaded values.
    """
    import torch

    model = context.model
    if model is None:
        return

    base = getattr(model, "get_base_model", lambda: model)()
    strict = tuple(getattr(type(base), "_keep_in_fp32_modules_strict", None) or ()) or (
        "mlp.gate.weight",
        "e_score_correction_bias",
    )
    n = 0
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.dtype == torch.float32:
            continue
        # Match whole dotted segments, not substrings: a PEFT adapter tensor whose
        # path merely contains a router name must stay in its training dtype.
        if any(name == s or name.endswith("." + s) for s in strict):
            tensor.data = tensor.data.to(torch.float32)
            n += 1
    if n:
        from axolotl.utils.logging import get_logger

        get_logger(__name__).info(
            f"beam: cast {n} router parameters to fp32 to match rank zero"
        )


@register_model_support
class BeamSupport(ModelSupport):
    """Descriptor for Beam (`BeamForCausalLM`): hybrid sliding/full attention
    with gated output, DeepSeek-V3-style MoE routing, and experts fused into 3D
    ``nn.Parameter`` tensors."""

    model_types = ("beam",)
    profile = ModelProfile(
        family=VANILLA_CAUSAL_LM,
        capabilities={
            "cut_cross_entropy": Supported(
                "Patched on BeamForCausalLM in the fork; final_logit_softcapping is "
                "applied to the fp32 accumulator, as Beam softcaps in fp32."
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
        },
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.AFTER_ADAPTER_LOAD: (_cast_fp32_strict_params,),
            }
        ),
    )
