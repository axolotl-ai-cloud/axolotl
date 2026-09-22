"""
Liger-Kernel Plugin for Axolotl
"""

import inspect
import sys
from types import SimpleNamespace

from axolotl.integrations.base import BasePlugin
from axolotl.model_support import check_capability, get_model_support
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_KERNEL_IMPL_ENV = "LIGER_KERNEL_IMPL"
# env value before this plugin's last owned write, and what it last wrote — lets
# a config that omits liger_kernel_impl undo a previous config's mutation (e.g.
# one rejected by validation after register() already ran)
_env_before_write: str | None = None
_last_written: str | None = None

LIGER_FLAGS = (
    "liger_rope",
    "liger_rms_norm",
    "liger_rms_norm_gated",
    "liger_layer_norm",
    "liger_glu_activation",
    "liger_cross_entropy",
    "liger_fused_linear_cross_entropy",
    "liger_use_token_scaling",
)


class LigerPlugin(BasePlugin):
    """
    Plugin for LIGER integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.liger.LigerArgs"

    def register(self, cfg):
        # earliest hook: pre-model-load patches (fused-attn kernels) import liger
        # before pre_model_load runs, which would fix the backend too soon.
        # cfg is the unparsed dict — invalid values fall through untouched so
        # schema validation raises the proper error instead of an env mutation
        from axolotl.integrations.liger.args import LIGER_KERNEL_IMPLS

        if cfg.get("liger_kernel_impl") in LIGER_KERNEL_IMPLS:
            self._set_kernel_impl(cfg["liger_kernel_impl"])
        elif cfg.get("liger_kernel_impl") is None:
            self._restore_kernel_impl()

    @staticmethod
    def _set_kernel_impl(impl: str):
        import os

        # liger reads LIGER_KERNEL_IMPL exactly once, at the first import of
        # liger_kernel.ops — setting it after that import is silently inert
        if "liger_kernel.ops" in sys.modules:
            # the env var is mutable after import, so compare against the backend
            # liger actually loaded, not the current env value
            loaded = LigerPlugin._loaded_kernel_impl()
            if loaded != impl:
                raise ValueError(
                    f"liger_kernel_impl: '{impl}' cannot take effect: liger_kernel.ops "
                    f"was already imported with backend {loaded or 'default'!r}. Another "
                    "component imported liger before the Liger plugin ran; set the "
                    "LIGER_KERNEL_IMPL env var before launching instead."
                )
            return
        global _env_before_write, _last_written
        current = os.environ.get(_KERNEL_IMPL_ENV)
        if _last_written is None or current != _last_written:
            # first write, or a foreign overwrite happened: that value is the new restore point
            _env_before_write = current
        os.environ[_KERNEL_IMPL_ENV] = impl
        _last_written = impl
        LOG.info(f"Set LIGER_KERNEL_IMPL={impl} for liger kernel backend selection")

    def on_config_validation_error(self, cfg):
        self._restore_kernel_impl()

    @staticmethod
    def _loaded_kernel_impl() -> str | None:
        # the applied impl's ops module is imported by _replace_with_impl_ops;
        # its presence in sys.modules identifies the backend liger loaded with
        from liger_kernel.ops.backends.registry import IMPL_REGISTRY

        for name, info in IMPL_REGISTRY.items():
            if info.module_path in sys.modules:
                return name
        return None

    @staticmethod
    def _restore_kernel_impl():
        import os

        global _env_before_write, _last_written
        if _last_written is None or "liger_kernel.ops" in sys.modules:
            return
        if os.environ.get(_KERNEL_IMPL_ENV) != _last_written:
            # foreign overwrite: relinquish ownership, never erase another component's value
            _env_before_write = None
            _last_written = None
            return
        if _env_before_write is None:
            os.environ.pop(_KERNEL_IMPL_ENV, None)
        else:
            os.environ[_KERNEL_IMPL_ENV] = _env_before_write
        _env_before_write = None
        _last_written = None

    def pre_model_load(self, cfg):
        if cfg.liger_kernel_impl:
            self._set_kernel_impl(cfg.liger_kernel_impl)

        if any(getattr(cfg, flag, False) for flag in LIGER_FLAGS):
            check_capability(
                get_model_support(cfg.model_config_type),
                "liger",
                cfg.model_config_type,
                feature="Liger",
                hint="Disable Liger flags for this model.",
            )

        # shim: liger imports ORPOTrainer from old trl.trainer (now trl.experimental.orpo)
        import trl.trainer
        from trl.experimental.orpo import ORPOTrainer

        trl.trainer.ORPOTrainer = ORPOTrainer

        if cfg.torch_compile:
            # torch compile will unnecessarily attempt to optimize the triton kernel unless explicitly disabled
            import liger_kernel.ops.fused_linear_cross_entropy

            from .utils import patch_with_compile_disable

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

        # liger natively dispatches these, but axolotl's branches add the fused
        # gated-RMSNorm kernel (Qwen3.5 linear-attention layers) that native lacks
        axolotl_override_liger_fn = {"qwen3_5", "qwen3_5_moe"}

        if (
            cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN
            and cfg.model_config_type not in axolotl_override_liger_fn
        ):
            apply_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[cfg.model_config_type]
            liger_fn_sig = inspect.signature(apply_liger_fn)
            kwargs = {}
            if "rope" in liger_fn_sig.parameters:
                rope_value = cfg.liger_rope
                # cfg.liger_rope defaults to None, which would override upstream's rope=True for Qwen-VL.
                if rope_value is None and cfg.model_config_type in (
                    "qwen2_vl",
                    "qwen2_5_vl",
                    "qwen3_vl",
                    "qwen3_vl_moe",
                    "qwen2_vl_text",
                    "qwen2_5_vl_text",
                    "qwen3_vl_text",
                    "qwen3_vl_moe_text",
                ):
                    rope_value = True
                kwargs["rope"] = rope_value
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
        elif cfg.model_config_type in ("mistral3", "ministral3"):
            # liger 0.8.0 has no mistral3/ministral3 entry, and its `ministral`
            # fn targets modeling_ministral, a module this arch does not use.
            from transformers.models.ministral3 import modeling_ministral3

            if cfg.liger_rope:
                modeling_ministral3.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_ministral3.Ministral3RMSNorm = LigerRMSNorm
            if cfg.liger_glu_activation:
                modeling_ministral3.Ministral3MLP = LigerSwiGLUMLP
            if cfg.liger_cross_entropy:
                from transformers.loss.loss_utils import nn

                nn.functional.cross_entropy = liger_cross_entropy
            if cfg.liger_fused_linear_cross_entropy:
                LOG.warning(
                    "Liger fused linear cross entropy is not implemented for the "
                    "Mistral3 multimodal wrapper. Skipping; use the "
                    "cut_cross_entropy plugin instead."
                )
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
        elif cfg.model_config_type == "qwen3_5":
            from axolotl.integrations.liger.models.qwen3_5 import (
                apply_liger_kernel_to_qwen3_5,
            )

            apply_liger_kernel_to_qwen3_5(
                cross_entropy=cfg.liger_cross_entropy,
                fused_linear_cross_entropy=cfg.liger_fused_linear_cross_entropy,
                glu_activation=cfg.liger_glu_activation,
                rms_norm=cfg.liger_rms_norm,
                rms_norm_gated=getattr(cfg, "liger_rms_norm_gated", False),
                layer_norm=cfg.liger_layer_norm,
            )
        elif cfg.model_config_type == "qwen3_5_moe":
            from axolotl.integrations.liger.models.qwen3_5_moe import (
                apply_liger_kernel_to_qwen3_5_moe,
            )

            apply_liger_kernel_to_qwen3_5_moe(
                cross_entropy=cfg.liger_cross_entropy,
                fused_linear_cross_entropy=cfg.liger_fused_linear_cross_entropy,
                glu_activation=cfg.liger_glu_activation,
                rms_norm=cfg.liger_rms_norm,
                rms_norm_gated=getattr(cfg, "liger_rms_norm_gated", False),
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
        elif cfg.model_config_type in ("gemma4_unified", "gemma4_unified_text"):
            # gemma4_unified mirrors gemma4's Liger compatibility: offset=0,
            # in_place=False (gradient-checkpoint safe), RoPE incompatible
            # (separate q/k application). Classes live in the unified namespace.
            from liger_kernel.transformers.geglu import LigerGEGLUMLP
            from transformers.models.gemma4_unified import modeling_gemma4_unified

            if cfg.liger_rms_norm:
                _OrigGemma4UnifiedRMSNorm = modeling_gemma4_unified.Gemma4UnifiedRMSNorm

                class _LigerGemma4UnifiedRMSNorm(LigerRMSNorm):
                    """LigerRMSNorm for Gemma4Unified (in_place=False, with_scale)."""

                    def __new__(cls, dim, eps=1e-6, with_scale=True):
                        if not with_scale:
                            return _OrigGemma4UnifiedRMSNorm(dim, eps, with_scale=False)
                        return super().__new__(cls)

                    def __init__(self, dim, eps=1e-6, with_scale=True):
                        if not with_scale:
                            return
                        super().__init__(
                            dim, eps, offset=0.0, casting_mode="llama", in_place=False
                        )

                modeling_gemma4_unified.Gemma4UnifiedRMSNorm = (
                    _LigerGemma4UnifiedRMSNorm
                )
            if cfg.liger_glu_activation:

                class _LigerGemma4UnifiedMLP(LigerGEGLUMLP):
                    def __init__(self, config, layer_idx=None):
                        super().__init__(config)

                modeling_gemma4_unified.Gemma4UnifiedTextMLP = _LigerGemma4UnifiedMLP
            if cfg.liger_rope:
                LOG.warning(
                    "Liger RoPE is not compatible with Gemma4Unified (separate "
                    "q/k application). Skipping."
                )
            if cfg.liger_layer_norm:
                modeling_gemma4_unified.nn.LayerNorm = LigerLayerNorm
            if cfg.liger_cross_entropy:
                modeling_gemma4_unified.nn.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                LOG.warning(
                    "Liger fused linear cross entropy is not compatible with "
                    "Gemma4Unified. Skipping."
                )
            LOG.info(
                f"Applied Liger kernels for gemma4_unified: "
                f"rms_norm={cfg.liger_rms_norm}, glu={cfg.liger_glu_activation}, "
                f"rope=False (incompatible), layer_norm={cfg.liger_layer_norm}"
            )
        elif cfg.model_config_type == "cohere_compass":
            from transformers.models.cohere_compass import modeling_cohere_compass

            if cfg.liger_rms_norm:
                LOG.warning(
                    "CohereCompass has no RMSNorm; every norm is the mean-subtracting "
                    "CohereCompassLayerNorm. Skipping."
                )
            if cfg.liger_glu_activation:
                modeling_cohere_compass.CohereCompassMLP = LigerSwiGLUMLP
            if cfg.liger_layer_norm:
                # Reaches the vision tower only. The decoder's own CohereCompassLayerNorm
                # is a custom fp32 norm, not nn.LayerNorm, and is deliberately left alone.
                modeling_cohere_compass.nn.LayerNorm = LigerLayerNorm
            if cfg.liger_rope:
                LOG.warning(
                    "Liger RoPE is not compatible with CohereCompass (interleaved "
                    "rope_style and 3D mrope). Skipping."
                )
            if cfg.liger_cross_entropy:
                from transformers.loss.loss_utils import nn as loss_nn

                loss_nn.functional.cross_entropy = liger_cross_entropy
            if cfg.liger_fused_linear_cross_entropy:
                LOG.warning(
                    "Liger fused linear cross entropy is not implemented for "
                    "CohereCompass. Use cut_cross_entropy instead. Skipping."
                )
            LOG.info(
                f"Applied Liger kernels for cohere_compass: "
                f"rms_norm=False (no RMSNorm in the arch), glu={cfg.liger_glu_activation}, "
                f"rope=False (incompatible), layer_norm={cfg.liger_layer_norm} (vision tower)"
            )
        elif cfg.model_config_type == "muse_glimmer":
            from transformers.models.muse_glimmer import modeling_muse_glimmer

            if cfg.liger_rms_norm:
                from liger_kernel.transformers.rms_norm import (
                    LigerRMSNormForGemma2,
                    LigerRMSNormForGemma4,
                )

                class _LigerMuseGlimmerRMSNorm(LigerRMSNormForGemma4):
                    """`dim` is omitted at the three with_scale=False call sites
                    (qk_norm, embed_norm, perception_emb_norm)."""

                    def __init__(self, dim=None, eps=1e-6, with_scale=True):
                        super().__init__(dim, eps, with_scale=with_scale)

                # The four decoder norms are Gemma2-shaped ((1 + w), zeros init); the
                # final norm and the weightless norms are Gemma4-shaped (w, ones init).
                modeling_muse_glimmer.MuseGlimmerTextCenteredRMSNorm = (
                    LigerRMSNormForGemma2
                )
                modeling_muse_glimmer.MuseGlimmerRMSNorm = _LigerMuseGlimmerRMSNorm
            if cfg.liger_glu_activation:

                class _LigerMuseGlimmerTextMLP(LigerSwiGLUMLP):
                    """SwiGLU MLP for MuseGlimmer.

                    MuseGlimmerTextConfig names the activation `hidden_activation`;
                    Liger reads `hidden_act`, which only exists on the vision config.
                    """

                    def __init__(self, config):
                        super().__init__(
                            SimpleNamespace(
                                hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_activation,
                            )
                        )
                        self.config = config

                modeling_muse_glimmer.MuseGlimmerTextMLP = _LigerMuseGlimmerTextMLP
            if cfg.liger_rope:
                from liger_kernel.transformers.rope import liger_rotary_pos_emb

                # NoPE layers never call this, so only the sliding layers are affected.
                modeling_muse_glimmer.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_layer_norm:
                # Patches the vision tower's four nn.LayerNorm sites; the text decoder
                # has none.
                modeling_muse_glimmer.nn.LayerNorm = LigerLayerNorm
            if cfg.liger_cross_entropy:
                LOG.warning(
                    "MuseGlimmer computes its loss through self.loss_function, not "
                    "nn.CrossEntropyLoss, so the Liger swap would be a no-op. Skipping."
                )
            if cfg.liger_fused_linear_cross_entropy:
                LOG.warning(
                    "Liger fused linear cross entropy is not compatible with MuseGlimmer: "
                    "logits are scaled by output_multiplier and tanh-softcapped after "
                    "lm_head. Use cut_cross_entropy instead. Skipping."
                )
            LOG.info(
                f"Applied Liger kernels for muse_glimmer: "
                f"rms_norm={cfg.liger_rms_norm}, glu={cfg.liger_glu_activation}, "
                f"rope={cfg.liger_rope}, layer_norm={cfg.liger_layer_norm} (vision tower)"
            )
        elif cfg.liger_fused_linear_cross_entropy:
            try:
                from .models.base import patch_lce_forward

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
