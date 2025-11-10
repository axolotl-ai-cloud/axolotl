"""
Liger-Kernel Plugin for Axolotl
"""

import inspect
import sys

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

from .models.base import patch_lce_forward
from .utils import patch_with_compile_disable

LOG = get_logger(__name__)


class LigerPlugin(BasePlugin):
    """
    Plugin for LIGER integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.liger.LigerArgs"

    def pre_model_load(self, cfg):
        if cfg.torch_compile:
            # torch compile will unnecessarily attempt to optimize the triton kernel unless explicitly disabled
            import liger_kernel.ops.fused_linear_cross_entropy

            patch_with_compile_disable(
                liger_kernel.ops.fused_linear_cross_entropy,
                "fused_linear_cross_entropy_forward",
            )
            patch_with_compile_disable(
                liger_kernel.ops.fused_linear_cross_entropy,
                "fused_linear_cross_entropy_backward",
            )
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
        from liger_kernel.transformers.functional import liger_cross_entropy
        from liger_kernel.transformers.layer_norm import LigerLayerNorm
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        if cfg.liger_cross_entropy and cfg.liger_fused_linear_cross_entropy:
            raise ValueError(
                "Cannot have both `liger_cross_entropy` and `liger_fused_linear_cross_entropy` set."
            )

        if cfg.liger_use_token_scaling:
            # Patch FLCE to set token_scaling=True for function and class API
            from liger_kernel.transformers import functional
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            old_liger_fused_linear_cross_entropy = (
                functional.liger_fused_linear_cross_entropy
            )

            def patched_liger_fused_linear_cross_entropy(*args, **kwargs):
                kwargs["use_token_scaling"] = True
                return old_liger_fused_linear_cross_entropy(*args, **kwargs)

            functional.liger_fused_linear_cross_entropy = (
                patched_liger_fused_linear_cross_entropy
            )

            old_init = LigerFusedLinearCrossEntropyLoss.__init__

            def patched_init(self, *args, **kwargs):
                kwargs["use_token_scaling"] = True
                return old_init(self, *args, **kwargs)

            LigerFusedLinearCrossEntropyLoss.__init__ = patched_init

        if cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
            apply_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[cfg.model_config_type]
            liger_fn_sig = inspect.signature(apply_liger_fn)
            kwargs = {}
            if "rope" in liger_fn_sig.parameters:
                kwargs["rope"] = cfg.liger_rope
            if "cross_entropy" in liger_fn_sig.parameters:
                kwargs["cross_entropy"] = cfg.liger_cross_entropy
            if "fused_linear_cross_entropy" in liger_fn_sig.parameters:
                kwargs["fused_linear_cross_entropy"] = (
                    cfg.liger_fused_linear_cross_entropy
                )
            if "rms_norm" in liger_fn_sig.parameters:
                kwargs["rms_norm"] = cfg.liger_rms_norm
            if "layer_norm" in liger_fn_sig.parameters:
                kwargs["layer_norm"] = cfg.liger_layer_norm
            if "geglu" in liger_fn_sig.parameters:
                kwargs["geglu"] = cfg.liger_glu_activation
            elif "swiglu" in liger_fn_sig.parameters:
                kwargs["swiglu"] = cfg.liger_glu_activation
            LOG.info(f"Applying LIGER to {cfg.model_config_type} with kwargs: {kwargs}")
            apply_liger_fn(**kwargs)
        elif cfg.model_config_type == "jamba":
            from transformers.models.jamba import modeling_jamba

            from .models.jamba import lce_forward as jamba_lce_forward

            if cfg.liger_rope:
                modeling_jamba.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_jamba.JambaRMSNorm = LigerRMSNorm
            if cfg.liger_glu_activation:
                modeling_jamba.JambaMLP = LigerSwiGLUMLP
            if cfg.liger_layer_norm:
                modeling_jamba.nn.LayerNorm = LigerLayerNorm
            if cfg.liger_cross_entropy:
                from transformers.loss.loss_utils import nn

                nn.functional.cross_entropy = liger_cross_entropy
            if cfg.liger_fused_linear_cross_entropy:
                modeling_jamba.JambaForCausalLM.forward = jamba_lce_forward
        elif cfg.model_config_type == "deepseek_v2":
            from accelerate import init_empty_weights
            from transformers import AutoModelForCausalLM

            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.base_model, trust_remote_code=cfg.trust_remote_code or False
                )
                modeling_mod = sys.modules[model.__class__.__module__]

            from .models.deepseekv2 import lce_forward as deepseekv2_lce_forward

            if cfg.liger_rope:
                # The DeepseekV2 version of RoPE is different than upstream LLaMA.
                # See https://github.com/linkedin/Liger-Kernel/issues/129#issuecomment-2313763528
                LOG.warning("Fused liger_rope is not supported for DeepseekV2.")
            if cfg.liger_rms_norm:
                modeling_mod.DeepseekV2RMSNorm = LigerRMSNorm
            if cfg.liger_glu_activation:
                modeling_mod.DeepseekV2MLP.forward = LigerSwiGLUMLP.forward
            if cfg.liger_layer_norm:
                LOG.warning("liger_layer_norm is not supported for DeepseekV2.")
            if cfg.liger_cross_entropy:
                # We do not patch `nn.functional.cross_entropy` for DeepseekV2 as it still uses
                # nn.CrossEntropyLoss in the forward method.
                modeling_mod.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                modeling_mod.DeepseekV2ForCausalLM.forward = deepseekv2_lce_forward
        elif cfg.model_config_type == "llama4":
            from axolotl.integrations.liger.models.llama4 import (
                apply_liger_kernel_to_llama4,
            )

            apply_liger_kernel_to_llama4(
                cross_entropy=cfg.liger_cross_entropy,
                fused_linear_cross_entropy=cfg.liger_fused_linear_cross_entropy,
                glu_activation=cfg.liger_glu_activation,
                rms_norm=cfg.liger_rms_norm,
                layer_norm=cfg.liger_layer_norm,
            )
        elif cfg.model_config_type == "qwen3":
            from axolotl.integrations.liger.models.qwen3 import (
                apply_liger_kernel_to_qwen3,
            )

            apply_liger_kernel_to_qwen3(
                cross_entropy=cfg.liger_cross_entropy,
                fused_linear_cross_entropy=cfg.liger_fused_linear_cross_entropy,
                glu_activation=cfg.liger_glu_activation,
                rms_norm=cfg.liger_rms_norm,
                layer_norm=cfg.liger_layer_norm,
            )
        elif cfg.model_config_type == "qwen3_moe":
            from axolotl.integrations.liger.models.qwen3_moe import (
                apply_liger_kernel_to_qwen3_moe,
            )

            apply_liger_kernel_to_qwen3_moe(
                cross_entropy=cfg.liger_cross_entropy,
                fused_linear_cross_entropy=cfg.liger_fused_linear_cross_entropy,
                glu_activation=cfg.liger_glu_activation,
                rms_norm=cfg.liger_rms_norm,
                layer_norm=cfg.liger_layer_norm,
            )
        elif cfg.model_config_type == "granitemoe":
            from liger_kernel.transformers import apply_liger_kernel_to_granite

            apply_liger_kernel_to_granite(
                rope=cfg.liger_rope,
                cross_entropy=cfg.liger_cross_entropy,
                fused_linear_cross_entropy=cfg.liger_fused_linear_cross_entropy,
                rms_norm=cfg.liger_rms_norm,
                swiglu=cfg.liger_glu_activation,
            )
        elif cfg.liger_fused_linear_cross_entropy:
            try:
                patch_lce_forward(cfg.model_config_type)
                LOG.warning_once(
                    f"Applied ONLY liger_fused_linear_cross_entropy genericpatches for model type: {cfg.model_config_type}"
                )
                LOG.warning_once(
                    f"Liger + {cfg.model_config_type} generic FLCE support is experimental and may not work as expected."
                )
            except RuntimeError:
                LOG.warning(
                    f"Unsupported model config type: {cfg.model_config_type}. Liger not applied."
                )
        else:
            LOG.warning(
                f"Unsupported model config type: {cfg.model_config_type}. Liger not applied."
            )
