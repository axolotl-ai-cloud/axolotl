"""Beam model support.

The architecture loads natively through transformers, so config, tokenizer and
modeling code all resolve without redirecting dynamic modules. What this
descriptor carries is the set of features whose generic fallbacks are wrong for
this arch, so enabling one fails loudly instead of training quietly on a
different objective.
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

    PATCH_FNS.setdefault(
        "beam", ("axolotl.model_support.beam.cut_cross_entropy", "patch_beam")
    )


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
    import os

    base = getattr(model, "get_base_model", lambda: model)()
    strict = tuple(getattr(type(base), "_keep_in_fp32_modules_strict", None) or ()) or (
        "mlp.gate.weight",
        "e_score_correction_bias",
    )
    strict += tuple(x for x in os.environ.get("BEAM_FP32_EXTRA", "").split(",") if x)
    n = 0
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if any(s in name for s in strict) and tensor.dtype != torch.float32:
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
            "cut_cross_entropy": Experimental(
                "Needs the beam patch registered in ml-cross-entropy; apply_lce "
                "takes the logit softcap, but the fp32 lm_head path is unverified."
            ),
            "lora_kernels": Unsupported(
                "Beam's attention adds an output gate (self_attn.gate_proj plus "
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
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.BEFORE_MODEL_BUILD: (_register_cce_patch,),
                ModelHookPhase.AFTER_ADAPTER_LOAD: (_cast_fp32_strict_params,),
            }
        ),
    )
