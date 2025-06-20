"""Patch manager class implementation to complement `axolotl.loaders.ModelLoader`.

Applies pre- and post-model load patches for various fixes and optimizations.
"""

import importlib.util
from functools import cached_property

import addict
import transformers
from transformers import PretrainedConfig, PreTrainedModel

from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.multipack import (
    SUPPORTED_MULTIPACK_MODEL_TYPES,
    patch_for_multipack,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
PLUGIN_MANAGER = PluginManager.get_instance()


class PatchManager:
    """Manages the application of patches during the model loading process."""

    def __init__(
        self,
        cfg: DictDefault,
        model_config: PretrainedConfig | addict.Dict,
        inference: bool = False,
    ):
        """Initialize the `PatchManager`.

        Args:
            cfg: Configuration dictionary with model and training settings.
            model_config: Configuration object for the model.
            inference: Whether the model is being loaded for inference mode.
        """
        self.cfg = cfg
        self.model_config = model_config
        self.inference = inference

    @cached_property
    def has_flash_attn(self) -> bool:
        """Check if flash attention is installed."""
        return importlib.util.find_spec("flash_attn") is not None

    def apply_pre_model_load_patches(self):
        """Apply pre-model load patches based on config."""
        self._apply_flash_attention_patches()
        self._apply_fsdp_patches()
        self._apply_adapter_patches()
        self._apply_flex_attention_patches()
        self._apply_model_specific_patches()
        self._apply_fp8_patches()
        self._apply_flash_attention_peft_patches()
        self._apply_gradient_checkpointing_patches()
        self._patch_attention()
        self._apply_multipack_patches()
        self._patch_loss_llama()
        self._patch_llama_derived_model()
        self._apply_mistral_cross_entropy_patch()
        self._apply_self_attention_lora_patch()

    def apply_post_model_load_patches(self, model: PreTrainedModel):
        """Apply patches that require the model instance."""
        self._apply_llama_flash_attn_patches(model)
        self._apply_unsloth_patches(model)
        self._apply_lora_kernel_patch(model)

    def _apply_flash_attention_patches(self):
        """Apply patches related to Flash Attention."""
        if self.cfg.xformers_attention and self.cfg.sample_packing:
            from axolotl.monkeypatch.attention import patch_xformers_attn_over_fa2

            patch_xformers_attn_over_fa2()
            self.cfg.flash_attention = True

    def _apply_fsdp_patches(self):
        """Apply patches for FSDP configurations."""
        if self.cfg.fsdp_config and str(self.cfg.fsdp_config.fsdp_version) == "2":
            from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2

            patch_accelerate_fsdp2()

    def _apply_adapter_patches(self):
        """Apply patches for adapter configurations."""
        if self.cfg.adapter and self.cfg.embeddings_skip_upcast:
            from axolotl.monkeypatch.peft.utils import patch_peft_prep_code

            patch_peft_prep_code()

    def _apply_flex_attention_patches(self):
        """Apply patches for flexible attention."""
        if self.cfg.flex_attention:
            from axolotl.monkeypatch.attention.flex_attn import (
                patch_flex_make_mask,
                patch_flex_wrapper,
            )

            flex_attn_compile_kwargs = self.cfg.flex_attn_compile_kwargs or {}
            patch_flex_wrapper(**flex_attn_compile_kwargs)
            patch_flex_make_mask()

    def _apply_model_specific_patches(self):
        """Apply patches specific to model architectures."""
        if (
            self.cfg.model_config_type == "llama4"
            and self.cfg.llama4_linearized_experts
        ):
            from axolotl.monkeypatch.models.llama4.modeling import (
                patch_llama4_linearized_modeling,
            )

            patch_llama4_linearized_modeling()

    def _apply_fp8_patches(self):
        """Apply patches for FP8 support."""
        if self.cfg.fp8:
            from axolotl.monkeypatch.trainer_accelerator_args import (
                patch_create_accelerate_code_for_fp8,
            )

            patch_create_accelerate_code_for_fp8()

    def _apply_flash_attention_peft_patches(self):
        """Apply patches for Flash Attention with PEFT."""
        if self.cfg.adapter:
            from axolotl.monkeypatch.transformers_fa_utils import (
                patch_fa_peft_integration,
            )

            patch_fa_peft_integration()

    def _apply_gradient_checkpointing_patches(self):
        """Apply patches for gradient checkpointing."""
        if self.cfg.gradient_checkpointing in ["unsloth", "offload"]:
            from axolotl.monkeypatch.gradient_checkpointing import (
                hf_grad_checkpoint_offload_wrapper,
            )

            transformers.modeling_utils.checkpoint = hf_grad_checkpoint_offload_wrapper
        if self.cfg.gradient_checkpointing == "offload_disk":
            from axolotl.monkeypatch.gradient_checkpointing import (
                hf_grad_checkpoint_disk_offload_wrapper,
            )

            transformers.modeling_utils.checkpoint = (
                hf_grad_checkpoint_disk_offload_wrapper
            )

    def _apply_mistral_cross_entropy_patch(self):
        """Apply Mistral cross entropy patch if configured."""
        if (
            self.cfg.model_config_type == "mistral"
            and self.cfg.flash_attn_cross_entropy_loss
        ):
            from axolotl.monkeypatch.mistral_attn_hijack_flash import (
                patch_mistral_cross_entropy,
            )

            patch_mistral_cross_entropy()

    def _apply_self_attention_lora_patch(self):
        """Apply self-attention LoRA patches if configured."""
        if self.cfg.lora_qkv_kernel or self.cfg.lora_o_kernel:
            # Only patch if conditions are met
            can_patch = (
                self.cfg.lora_dropout == 0
                if hasattr(self.cfg, "lora_dropout")
                else True
            )  # default to True if lora_dropout is not set

            if not can_patch:
                LOG.warning("Cannot patch self-attention - requires no dropout")
                return

            from axolotl.monkeypatch.lora_kernels import patch_self_attn_lora

            patch_self_attn_lora(self.cfg)

    def _apply_multipack_patches(self):
        """Apply multipack patches if necessary."""
        if (
            self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
            and (self.cfg.flash_attention or self.cfg.flex_attention)
            and self.cfg.sample_packing
        ):
            # Get automap config if it exists
            auto_map_config = None
            if isinstance(self.model_config, dict) and "auto_map" in self.model_config:
                auto_map_config = self.model_config["auto_map"]
            elif hasattr(self.model_config, "auto_map"):
                auto_map_config = self.model_config.auto_map

            # Determine if the model has remote code
            if auto_map_config is not None:
                has_remote_code = "AutoModelForCausalLM" in auto_map_config
            else:
                has_remote_code = False

            if has_remote_code and self.cfg.trust_remote_code is False:
                # If explicitly set in YAML, prefer that
                has_remote_code = self.cfg.trust_remote_code

            patch_for_multipack(
                self.cfg.model_config_type,
                model_name=self.cfg.base_model,
                has_remote_code=has_remote_code,
            )

    def _patch_attention(self):
        """Apply attention-specific patches based on model type."""
        if not (self.cfg.flash_attention and hasattr(self.model_config, "model_type")):
            return

        if self.model_config.model_type == "btlm":
            from axolotl.monkeypatch.btlm_attn_hijack_flash import (
                replace_btlm_attn_with_flash_attn,
            )

            replace_btlm_attn_with_flash_attn(self.cfg.base_model)

        if self.model_config.model_type == "stablelm_epoch" and self.cfg.sample_packing:
            from axolotl.monkeypatch.stablelm_attn_hijack_flash import (
                replace_stablelm_attn_with_flash_attn,
            )

            replace_stablelm_attn_with_flash_attn(self.cfg.base_model)

    def _patch_loss_llama(self):
        """Patch loss functions and other optimizations for LLaMA models."""
        if not self.cfg.is_llama_derived_model:
            return

        if self.cfg.flash_attn_cross_entropy and self.has_flash_attn:
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                patch_fa_llama_cross_entropy,
            )

            patch_fa_llama_cross_entropy()
        elif self.cfg.unsloth_cross_entropy_loss:
            from axolotl.monkeypatch.unsloth_ import integrate_cross_entropy_loss_patch

            integrate_cross_entropy_loss_patch(model_type="llama")

        if self.cfg.flash_attn_rms_norm and self.has_flash_attn:
            from axolotl.monkeypatch.llama_attn_hijack_flash import patch_llama_rms_norm

            patch_llama_rms_norm()
        elif self.cfg.unsloth_rms_norm:
            from axolotl.monkeypatch.unsloth_ import patch_unsloth_layernorm

            patch_unsloth_layernorm()

        if self.cfg.unsloth_lora_qkv or self.cfg.unsloth_lora_o:
            from axolotl.monkeypatch.unsloth_ import patch_self_attn_lora

            patch_self_attn_lora()

    def _patch_llama_flash_attention(self, packed=False):
        """Apply Flash Attention patches for LLaMA models."""
        from axolotl.monkeypatch.llama_attn_hijack_flash import (
            replace_llama_attn_with_flash_attn,
        )

        if packed:
            if self.cfg.device not in ["mps", "cpu"] and not self.inference:
                LOG.info("patching with flash attention for sample packing")
                replace_llama_attn_with_flash_attn(
                    packed=True,
                    cross_entropy=self.cfg.flash_attn_cross_entropy,
                    rms_norm=self.cfg.flash_attn_rms_norm,
                )
        elif self.cfg.s2_attention:
            LOG.info("patching w/ flash-enabled, shifted-sparse attention")
            replace_llama_attn_with_flash_attn(
                packed=False,
                cross_entropy=self.cfg.flash_attn_cross_entropy,
                rms_norm=self.cfg.flash_attn_rms_norm,
                use_shifted_sparse_attn=True,
            )
        elif self.cfg.flash_attn_cross_entropy or self.cfg.flash_attn_rms_norm:
            replace_llama_attn_with_flash_attn(
                packed=False,
                cross_entropy=self.cfg.flash_attn_cross_entropy,
                rms_norm=self.cfg.flash_attn_rms_norm,
            )

    def _patch_llama_xformers_attention(self):
        """Apply xformers attention patches for LLaMA models."""
        from axolotl.monkeypatch.llama_attn_hijack_xformers import (
            hijack_llama_attention,
        )

        LOG.info("Patching with xformers attention...")
        hijack_llama_attention()

    def _patch_llama_sample_packing(self):
        """Apply sample packing patches for LLaMA models."""
        from axolotl.monkeypatch.llama_patch_multipack import (
            hijack_llama_prepare_4d_mask,
        )

        LOG.info("Patching llama _prepare_4d_causal_attention_mask*...")
        hijack_llama_prepare_4d_mask()

    def _patch_llama_derived_model(self):
        """Modify all llama derived models in one block."""
        if self.cfg.is_llama_derived_model and not (
            self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
            and (self.cfg.flash_attention or self.cfg.flex_attention)
            and self.cfg.sample_packing
        ):
            if self.cfg.flash_attention:
                self._patch_llama_flash_attention(packed=self.cfg.sample_packing)
            elif self.cfg.xformers_attention:
                self._patch_llama_xformers_attention()
            elif self.cfg.sample_packing:
                self._patch_llama_sample_packing()
            elif self.cfg.s2_attention:
                raise NotImplementedError(
                    "Shifted-sparse attention not currently implemented without flash attention."
                )

    def _apply_llama_flash_attn_patches(self, model):
        """Apply LLaMA-specific flash attention patches."""
        if (
            self.model_config.model_type in ["llama", "llama4"]
            and not self.cfg.trust_remote_code
            and not self.cfg.gptq
            and self.cfg.flash_attention
            and not self.inference
        ):
            # TODO(MengqingCao): split these patches seperately
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                is_xformers_swiglu_available,
                replace_llama_mlp_with_swiglu,
                replace_llama_qkv_with_fused,
            )

            if self.cfg.flash_attn_fuse_mlp and is_xformers_swiglu_available():
                LOG.info("Patching with SwiGLU...")
                replace_llama_mlp_with_swiglu(model)

            if self.cfg.flash_attn_fuse_qkv:
                LOG.info("Patching with fused QKV...")
                replace_llama_qkv_with_fused(model)

    def _apply_unsloth_patches(self, model):
        """Apply unsloth optimization patches."""
        if self.cfg.unsloth_lora_mlp:
            from axolotl.monkeypatch.unsloth_ import integrate_lora_mlp_patch

            integrate_lora_mlp_patch(peft_model=model)

        if self.cfg.unsloth_lora_qkv or self.cfg.unsloth_lora_o:
            from axolotl.monkeypatch.unsloth_ import integrate_lora_patch

            integrate_lora_patch(peft_model=model, cfg=self.cfg)

        if self.cfg.unsloth_rope:
            from axolotl.monkeypatch.unsloth_ import integrate_rope_embeddings

            integrate_rope_embeddings()

    def _apply_lora_kernel_patch(self, model):
        """Apply LoRA kernel patches."""
        if (
            self.cfg.lora_mlp_kernel
            or self.cfg.lora_qkv_kernel
            or self.cfg.lora_o_kernel
        ):
            from axolotl.monkeypatch.lora_kernels import apply_lora_kernel_patches

            apply_lora_kernel_patches(model=model, cfg=self.cfg)
