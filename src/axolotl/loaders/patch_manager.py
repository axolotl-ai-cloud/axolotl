"""Patch manager class implementation to complement `axolotl.loaders.ModelLoader`.

Applies pre- and post-model load patches for various fixes and optimizations.
"""

import importlib.util
import os
from functools import cached_property

import addict
import torch
import transformers
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_flash_attention_utils import is_flash_attn_available

from axolotl.integrations.base import PluginManager
from axolotl.model_support import (
    check_capability,
    get_model_support,
    get_model_support_for_cfg,
)
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

    @staticmethod
    def apply_pre_config_load_patches(cfg: DictDefault):
        """
        Apply patches that must be set up before config loading.
        This is for patches that intercept remote code loading from HuggingFace,
        which needs to be in place before AutoConfig.from_pretrained() is called.

        Args:
            cfg: Configuration dictionary with model and training settings.
        """
        support = get_model_support_for_cfg(cfg)
        if support is not None:
            support.pre_config_load(cfg)

    @staticmethod
    def apply_pre_tokenizer_load_patches(cfg: DictDefault):
        """
        Apply patches that must be set up before tokenizer loading.
        This is for patches that intercept remote code loading from HuggingFace,
        which needs to be in place before AutoTokenizer.from_pretrained() is called.

        Args:
            cfg: Configuration dictionary with model and training settings.
        """
        support = get_model_support_for_cfg(cfg)
        if support is not None:
            support.pre_tokenizer_load(cfg)

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
        self._deactivate_hf_async_load()
        self._apply_torchao_patches()
        self._apply_transformers_patches()
        # self._apply_flex_attention_patches()
        self._apply_flash_attention_patches()
        self._apply_chunked_cross_entropy_patch()
        self._apply_sageattn_patches()
        self._apply_flash_attn_4_patches()
        self._apply_fsdp_patches()
        self._apply_adapter_patches()
        # Must precede fused-RoPE patches: re-parses ``Attention.forward``
        # via ``inspect.getsource``; the QKV regex misses on a patched body.
        self._apply_self_attention_lora_patch()
        self._apply_model_support_pre_load_hook()
        self._apply_model_specific_patches()
        self._apply_fp8_patches()
        self._apply_flash_attention_peft_patches()
        self._apply_gradient_checkpointing_patches()
        self._patch_attention()
        self._apply_multipack_patches()
        self._apply_sdpa_varlen_patch()
        self._apply_large_head_attention_patch()
        self._patch_loss_llama()
        self._patch_llama_derived_model()
        self._apply_mistral_cross_entropy_patch()
        self._apply_fsdp2_bnb_patches()
        self._apply_patch_deepspeed_zero3()
        self._apply_voxtral_patches()
        self._apply_apertus_patches()
        self._apply_trl_vllm_patches()
        self._apply_trl_trainer_utils_patches()

    def apply_post_plugin_pre_model_load_patches(self):
        """Apply post plugin-pre_model_load load patches based on config."""
        self._apply_tiled_mlp(self.cfg.model_config_type)
        self._apply_moe_expert_quantization_patch()

    @staticmethod
    def _apply_torchao_patches():
        from axolotl.monkeypatch.torchao_optim import patch_torchao_optim_state_8bit

        patch_torchao_optim_state_8bit()

    def _apply_transformers_patches(self):
        from axolotl.monkeypatch.transformers.trainer_loss_calc import (
            patch_evaluation_loop,
            patch_maybe_log_save_evaluate,
        )

        patch_evaluation_loop()
        patch_maybe_log_save_evaluate()

    def apply_post_model_build_patches(self, model: PreTrainedModel):
        """Apply patches right after model build, before post-load setup."""
        if self.cfg.model_config_type == "nemotron_h":
            # Must run after model build because NemotronHForCausalLM.__init__
            # calls register_nemotron_h_conversion_mapping() with overwrite=True,
            # which would clobber any earlier fix.
            self._fix_nemotron_h_conversion_mapping()

        # Gemma 4 hybrid attention runs here in post-build (NOT post-load):
        # the per-layer ``self_attn.config._attn_implementation="sdpa"``
        # override needs to walk the raw model tree, which is broken by
        # the post-load PEFT wrapping. The accompanying
        # ``patch_gemma4_hybrid_mask`` monkey-patch is module-level and
        # installation-time-independent, so both halves of the fix live
        # cleanly in the same call even though one is instance-scoped
        # and the other is module-scoped.
        self._apply_gemma_hybrid_attention(model)
        self._apply_gemma4_loss_kwargs()
        self._finalize_moe_expert_quantization(model)

    def _apply_model_support_pre_load_hook(self):
        support = get_model_support(self.cfg.model_config_type)
        if support is not None:
            support.pre_model_load(self.cfg)

    def apply_post_model_load_patches(self, model: PreTrainedModel):
        """Apply patches that require the model instance."""
        support = get_model_support(self.cfg.model_config_type)
        if support is not None:
            support.post_model_load(self.cfg, model)
        self._apply_llama_flash_attn_patches(model)
        self._apply_lora_kernel_patch(model)
        self._apply_scaling_softmax_patch(model)
        self._apply_fp8_attention_patches(model)
        self._apply_tiled_mlp_post_load(model)

    def _apply_gemma4_loss_kwargs(self):
        # Flip accepts_loss_kwargs True so the Trainer normalizes loss by
        # num_items_in_batch under grad accumulation (must run before trainer init).
        if self.cfg.model_config_type not in ("gemma4", "gemma4_unified"):
            return
        from axolotl.monkeypatch.gemma4_loss_kwargs import (
            patch_gemma4_accepts_loss_kwargs,
        )

        patch_gemma4_accepts_loss_kwargs()

    def _apply_gemma_hybrid_attention(self, model: PreTrainedModel):
        """Apply hybrid attention: FA2 for sliding window layers, SDPA for global layers.

        Gemma 4 has global (full_attention) layers with head_dim=512
        which exceeds flash attention's supported size. This patch loads the model
        with flash_attention_2 for the sliding window layers (head_dim=256), then
        gives each global layer a shallow-copied config with _attn_implementation="sdpa".

        We also install :func:`axolotl.monkeypatch.gemma4_hybrid_mask.patch_gemma4_hybrid_mask`
        which fixes the corresponding mask construction inside
        ``Gemma4TextModel.forward``. Without it, the per-layer SDPA config
        override is not enough — the forward still builds a 2D FA2-format mask
        at the model level and the SDPA layers crash at long context lengths
        with ``RuntimeError: The expanded size of the tensor ... must match``.
        """
        if not self.cfg.gemma4_hybrid_attn_impl:
            return

        import copy

        from axolotl.monkeypatch.attention.large_head import (
            resolve_large_head_policy,
            set_large_head_packed,
            set_large_head_policy,
        )
        from axolotl.monkeypatch.gemma4_hybrid_mask import (
            GLOBAL_PACKED_SDPA,
            patch_gemma4_hybrid_mask,
        )

        patch_gemma4_hybrid_mask()
        # Gemma-4 global layers reuse the generic large-head router. Default policy 'sdpa' (flash is
        # opt-in via large_head_attention / the deprecated flash_attn_d512), preserving prior default.
        set_large_head_policy(resolve_large_head_policy(self.cfg))
        set_large_head_packed(bool(self.cfg.sample_packing))

        # Navigate to the module that has 'layers' - varies by model structure:
        # Gemma4ForConditionalGeneration -> .model (Gemma4Model) -> .language_model (Gemma4TextModel) -> .layers
        # Gemma4ForCausalLM -> .model (Gemma4TextModel) -> .layers
        layers = None
        config_source = None
        for candidate in [model, getattr(model, "model", None)]:
            if candidate is None:
                continue
            # Check direct layers
            if hasattr(candidate, "layers"):
                layers = candidate.layers
                config_source = candidate
                break
            # Check language_model.layers (multimodal wrapper)
            lang_model = getattr(candidate, "language_model", None)
            if lang_model is not None and hasattr(lang_model, "layers"):
                layers = lang_model.layers
                config_source = lang_model
                break

        if layers is None:
            LOG.warning(
                "gemma4_hybrid_attn_impl: could not find decoder layers in model, skipping"
            )
            return

        config = getattr(config_source, "config", self.model_config)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            LOG.warning(
                "gemma4_hybrid_attn_impl: model config has no 'layer_types', skipping. "
                "This feature requires a model with mixed sliding/global attention layers."
            )
            return

        patched_count = 0
        for layer_idx, layer in enumerate(layers):
            if layer_types[layer_idx] != "sliding_attention":
                # Global / full_attention layer (head_dim=512, FA2 can't serve it). Use the
                # packing-aware SDPA impl: it rebuilds the block-diagonal mask from position_ids so
                # the layer respects document boundaries under sample packing (plain "sdpa" gets a
                # None mask here and would attend across packed documents).
                attn_module = getattr(layer, "self_attn", None)
                if attn_module is not None and hasattr(attn_module, "config"):
                    sdpa_config = copy.copy(attn_module.config)
                    sdpa_config._attn_implementation = GLOBAL_PACKED_SDPA
                    attn_module.config = sdpa_config
                    patched_count += 1

        LOG.info(
            "gemma4_hybrid_attn_impl: patched %d global layers to use packing-aware SDPA "
            "(remaining %d sliding layers use flash_attention_2)",
            patched_count,
            len(layers) - patched_count,
        )

    def _apply_flash_attention_patches(self):
        """Apply patches related to Flash Attention."""
        if self.cfg.attn_implementation == "xformers":
            from axolotl.monkeypatch.attention import register_xformers_attn

            register_xformers_attn()

            if self.cfg.sample_packing:
                # Also patch FA2 slot for legacy code paths that use it directly
                from axolotl.monkeypatch.attention import patch_xformers_attn_over_fa2

                patch_xformers_attn_over_fa2()

        if self.cfg.attn_implementation == "sage":
            from axolotl.monkeypatch.attention import register_sage_attn

            register_sage_attn()

    def _apply_fp8_attention_patches(self, model):
        """Apply FP8 low-precision attention via torchao."""
        if self.cfg.attn_implementation == "fp8":
            from axolotl.monkeypatch.attention.fp8_attn import patch_fp8_attention

            patch_fp8_attention(model)

    def _apply_chunked_cross_entropy_patch(self):
        if self.cfg.chunked_cross_entropy:
            from axolotl.monkeypatch.loss.chunked import patch_chunked_ce_loss_fn

            if self.cfg.chunked_cross_entropy_num_chunks:
                patch_chunked_ce_loss_fn(self.cfg.chunked_cross_entropy_num_chunks)
            else:
                patch_chunked_ce_loss_fn()

    def _apply_fsdp_patches(self):
        """Apply patches for FSDP configurations."""
        if self.cfg.fsdp_config:
            from axolotl.monkeypatch.accelerate.fsdp2 import (
                patch_initialize_missing_keys_for_fsdp,
            )

            patch_initialize_missing_keys_for_fsdp()

            # Only for adapter (frozen-base) runs: this patch leaves non-rank-0 base params on meta
            # (FSDP broadcasts rank-0's weights into them), which avoids world_size× CPU
            # materialization of large unrecognized-quantizer (NVFP4-modelopt) checkpoints. Those are
            # always trained with a frozen base, so no base optimizer state exists. For a FULL
            # fine-tune the base params DO carry optimizer state, and leaving them on meta deadlocks
            # the FSDP2 optimizer-state all-gather at checkpoint save (rank-0 real DTensors vs
            # non-rank-0 meta) — so fall back to the stock materialize-to-cpu path there.
            if self.cfg.fsdp_config.cpu_ram_efficient_loading and self.cfg.adapter:
                from axolotl.monkeypatch.accelerate.fsdp2 import (
                    patch_move_missing_keys_meta_for_fsdp,
                )

                patch_move_missing_keys_meta_for_fsdp()

        if self.cfg.context_parallel_size > 1 or (
            self.cfg.fsdp_config and str(self.cfg.fsdp_version) == "2"
        ):
            from axolotl.monkeypatch.accelerate.parallelism_config import (
                patch_parallelism_config,
            )

            patch_parallelism_config()
        if self.cfg.fsdp_config and str(self.cfg.fsdp_version) == "2":
            from axolotl.monkeypatch.accelerate.float8_fsdp import patch_float8_fsdp
            from axolotl.monkeypatch.accelerate.fsdp2 import (
                patch_accelerate_fsdp2,
                patch_tied_keys_for_meta_device,
            )

            patch_accelerate_fsdp2()
            # FSDP2 sharding for any torchao Float8Tensor weights (no-op without torchao)
            patch_float8_fsdp()
            if self.cfg.fsdp_config.cpu_ram_efficient_loading:
                patch_tied_keys_for_meta_device()
            if self.cfg.rl:
                from axolotl.monkeypatch.trainer.trl import patch_trl_prepare_fsdp2

                patch_trl_prepare_fsdp2()

    def _apply_adapter_patches(self):
        """Apply patches for adapter configurations."""
        if self.cfg.adapter and self.cfg.embeddings_skip_upcast:
            from axolotl.monkeypatch.peft.utils import patch_peft_prep_code

            patch_peft_prep_code()

    def _apply_flex_attention_patches(self):
        """Apply patches for flexible attention."""
        if self.cfg.attn_implementation == "flex_attention":
            from axolotl.monkeypatch.attention.flex_attn import (
                patch_flex_wrapper,
            )

            flex_attn_compile_kwargs = self.cfg.flex_attn_compile_kwargs or {}
            patch_flex_wrapper(**flex_attn_compile_kwargs)

    def _apply_sageattn_patches(self):
        """Apply patches for SageAttention."""
        if self.cfg.attn_implementation == "sage":
            from axolotl.monkeypatch.attention.sage_attn import patch_sageattn

            patch_sageattn()

    def _apply_flash_attn_4_patches(self):
        """Auto-apply FA4 when flash_attention is enabled and FA4 is available on SM90+."""
        if not self.cfg.attn_uses_flash_lib:
            return

        from axolotl.monkeypatch.attention.flash_attn_4 import patch_flash_attn_4

        patch_flash_attn_4(self.model_config)

    _FUSED_ATTN_KERNEL_SUPPORTED = (
        "qwen3",
        "qwen3_moe",
        "qwen3_vl",
        "qwen3_vl_text",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "gemma4",
        "gemma4_text",
        "gemma4_unified",
        "gemma4_unified_text",
    )

    @staticmethod
    def _warn_if_fused_attn_unsupported(cfg):
        """Warn when ``fused_attn_kernel`` targets an unsupported
        ``model_config_type`` (derived post-schema by ``normalize_config()``)."""
        if not getattr(cfg, "fused_attn_kernel", False):
            return
        mct = getattr(cfg, "model_config_type", None)
        if mct and mct not in PatchManager._FUSED_ATTN_KERNEL_SUPPORTED:
            LOG.warning(
                "`fused_attn_kernel: true` is set but model_config_type=%r is not "
                "in the supported set %s. The flag is a silent no-op for this "
                "model. Remove the flag or use one of the supported model families.",
                mct,
                sorted(PatchManager._FUSED_ATTN_KERNEL_SUPPORTED),
            )

    def _apply_model_specific_patches(self):
        """Apply patches specific to model architectures."""
        self._warn_if_fused_attn_unsupported(self.cfg)

        if getattr(self.cfg, "use_kernels", None):
            from axolotl.monkeypatch.kernelize_fixes import patch_kernelize_fixes

            patch_kernelize_fixes()

        if (
            self.cfg.model_config_type == "llama4"
            and self.cfg.llama4_linearized_experts
        ):
            from axolotl.monkeypatch.models.llama4.modeling import (
                patch_llama4_linearized_modeling,
            )

            patch_llama4_linearized_modeling()

        ssm_hybrid_patch_needed = (
            self.cfg.sample_packing or self.cfg.context_parallel_size > 1
        )

        if self.cfg.model_config_type == "nemotron_h" and ssm_hybrid_patch_needed:
            from transformers.models.nemotron_h.modeling_nemotron_h import (
                NemotronHPreTrainedModel,
            )

            from axolotl.monkeypatch.models.nemotron_h.modeling import (
                patch_nemotron_h_modeling_packing,
            )

            patch_nemotron_h_modeling_packing()
            # supports_gradient_checkpointing is only enabled after
            # patch_nemotron_h_modeling_packing() installs the GC-compatible
            # NemotronHBlock.forward. Without the patch, upstream marks this
            # False because the original block forward is not GC-safe.
            NemotronHPreTrainedModel.supports_gradient_checkpointing = True

        if self.cfg.model_config_type == "falcon_h1" and ssm_hybrid_patch_needed:
            from axolotl.monkeypatch.models.falcon_h1.modeling import (
                patch_falcon_h1_modeling_packing,
            )

            patch_falcon_h1_modeling_packing()

        if self.cfg.model_config_type == "granitemoehybrid" and ssm_hybrid_patch_needed:
            from axolotl.monkeypatch.models.granitemoehybrid.modeling import (
                patch_granitemoehybrid_modeling_packing,
            )

            patch_granitemoehybrid_modeling_packing()

        # Patches requiring CUDA
        if torch.cuda.is_available():
            if self.cfg.model_config_type == "qwen3_next" and self.cfg.sample_packing:
                from axolotl.monkeypatch.models.qwen3_next.modeling import (
                    patch_qwen3_next_modeling_packing,
                )

                patch_qwen3_next_modeling_packing()

            if self.cfg.model_config_type == "qwen3_5" and self.cfg.sample_packing:
                from axolotl.monkeypatch.models.qwen3_5.modeling import (
                    patch_qwen3_5_modeling_packing,
                )

                patch_qwen3_5_modeling_packing()

            if self.cfg.model_config_type == "qwen3_5_moe" and self.cfg.sample_packing:
                from axolotl.monkeypatch.models.qwen3_5.modeling import (
                    patch_qwen3_5_moe_modeling_packing,
                )

                patch_qwen3_5_moe_modeling_packing()

            if (
                self.cfg.model_config_type in ["qwen3_5", "qwen3_5_moe"]
                and self.cfg.is_multimodal
                and self.cfg.attn_uses_flash_lib
            ):
                from axolotl.monkeypatch.models.qwen3_5.modeling import (
                    patch_qwen3_5_vlm_flash_attention,
                )

                patch_qwen3_5_vlm_flash_attention()

            if self.cfg.model_config_type in (
                "gemma4",
                "gemma4_text",
                "gemma4_unified",
                "gemma4_unified_text",
            ):
                # Shared-KV side channel when activation checkpointing (PR #3611).
                fsdp_cfg = self.cfg.fsdp_config
                needs_shared_kv_workaround = (not self.inference) and bool(
                    self.cfg.gradient_checkpointing
                    or self.cfg.activation_offloading
                    or (fsdp_cfg is not None and fsdp_cfg.activation_checkpointing)
                )
                if self.cfg.model_config_type in (
                    "gemma4_unified",
                    "gemma4_unified_text",
                ):
                    from axolotl.monkeypatch.models.gemma4_unified.fused_attn import (
                        patch_gemma4_unified_fused_attn as patch_fused_attn,
                    )
                else:
                    from axolotl.monkeypatch.models.gemma4.fused_attn import (
                        patch_gemma4_fused_attn as patch_fused_attn,
                    )
                patch_fused_attn(
                    install_shared_kv_workaround=needs_shared_kv_workaround
                )

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type == "qwen3":
                from axolotl.monkeypatch.models.qwen3.fused_attn import (
                    patch_qwen3_fused_attn,
                )

                patch_qwen3_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type == "qwen3_moe":
                from axolotl.monkeypatch.models.qwen3_moe.fused_attn import (
                    patch_qwen3_moe_fused_attn,
                )

                patch_qwen3_moe_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type in (
                "qwen3_vl",
                "qwen3_vl_text",
            ):
                from axolotl.monkeypatch.models.qwen3_vl.fused_attn import (
                    patch_qwen3_vl_fused_attn,
                )

                patch_qwen3_vl_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type in (
                "qwen3_5",
                "qwen3_5_text",
            ):
                from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
                    patch_qwen3_5_fused_attn,
                )

                patch_qwen3_5_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type in (
                "qwen3_5_moe",
                "qwen3_5_moe_text",
            ):
                from axolotl.monkeypatch.models.qwen3_5_moe.fused_attn import (
                    patch_qwen3_5_moe_fused_attn,
                )

                patch_qwen3_5_moe_fused_attn()

    @staticmethod
    def _fix_nemotron_h_conversion_mapping():
        """Remove the spurious embedding→embeddings WeightRenaming from the
        nemotron_h checkpoint conversion mapping.

        The nvidia Hub model registers:
            WeightRenaming("embedding.weight", "embeddings.weight")
        to handle a legacy checkpoint variant. Its reverse (applied on save)
        converts ``embeddings`` back to ``embedding``, which silently renames
        ``backbone.embeddings.weight`` → ``backbone.embedding.weight`` when
        merging LoRA adapters back into the base model.
        """
        try:
            from transformers.conversion_mapping import (
                WeightRenaming,
                get_checkpoint_conversion_mapping,
                register_checkpoint_conversion_mapping,
            )
        except ImportError:
            return

        mapping = get_checkpoint_conversion_mapping("nemotron_h")
        if mapping is None:
            return

        filtered = [
            entry
            for entry in mapping
            if not (
                isinstance(entry, WeightRenaming)
                and entry.source_patterns == ["embedding.weight"]
                and entry.target_patterns == ["embeddings.weight"]
            )
        ]
        if len(filtered) != len(mapping):
            register_checkpoint_conversion_mapping(
                "nemotron_h", filtered, overwrite=True
            )
            LOG.info(
                "Removed embedding→embeddings WeightRenaming from nemotron_h "
                "checkpoint conversion mapping"
            )

    def _apply_fp8_patches(self):
        """Apply patches for FP8 support."""
        if self.cfg.fp8:
            from axolotl.monkeypatch.accelerate.float8_moe_filter import (
                patch_fp8_exclude_moe_router,
            )
            from axolotl.monkeypatch.trainer_accelerator_args import (
                patch_create_accelerate_code_for_fp8,
            )

            patch_create_accelerate_code_for_fp8(
                self.cfg.fp8_enable_fsdp_float8_all_gather
            )
            patch_fp8_exclude_moe_router()

    def _apply_flash_attention_peft_patches(self):
        """Apply patches for Flash Attention with PEFT."""
        if self.cfg.adapter:
            from axolotl.monkeypatch.transformers_fa_utils import (
                patch_fa_peft_integration,
            )

            patch_fa_peft_integration()

    def _apply_gradient_checkpointing_patches(self):
        """Apply patches for gradient checkpointing."""
        if (
            self.cfg.gradient_checkpointing
            and self.cfg.activation_offloading == "legacy"
        ):
            from axolotl.monkeypatch.gradient_checkpointing import (
                hf_grad_checkpoint_offload_wrapper,
            )

            transformers.modeling_utils.checkpoint = hf_grad_checkpoint_offload_wrapper
        elif (
            self.cfg.gradient_checkpointing
            and self.cfg.activation_offloading == "offload_disk"
        ):
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
            check_capability(
                get_model_support(self.cfg.model_config_type),
                "lora_kernels",
                self.cfg.model_config_type,
                feature="LoRA QKV/O kernels",
                hint="Set lora_qkv_kernel: false and lora_o_kernel: false.",
            )
            from axolotl.monkeypatch.lora_kernels import patch_self_attn_lora

            patch_self_attn_lora(self.cfg)

    def _apply_large_head_attention_patch(self):
        """Generic head_dim>256 capability for plain SDPA models. Gemma-4's hybrid path routes its
        globals through its own impl, so skip the generic sdpa wrapper there to avoid double-wiring."""
        from axolotl.monkeypatch.attention.large_head import (
            resolve_large_head_policy,
            set_large_head_packed,
            set_large_head_policy,
            unpatch_sdpa_large_head,
        )

        policy = resolve_large_head_policy(self.cfg)
        # Always (re)set the policy global from this run's config so a long-lived process can't
        # inherit a previous run's stale auto/triton_flash policy on an sdpa run.
        set_large_head_policy(policy)
        set_large_head_packed(bool(self.cfg.sample_packing))
        if policy == "sdpa" or self.cfg.gemma4_hybrid_attn_impl:
            unpatch_sdpa_large_head()
            return
        from axolotl.monkeypatch.attention.large_head import patch_sdpa_large_head

        patch_sdpa_large_head(policy)

    def _apply_sdpa_varlen_patch(self):
        """Route packed-row SDPA through cu_seqlens ``varlen_attn`` (no 4D mask).

        Auto-enabled for ``sdpa`` + ``sample_packing`` when the varlen kernel can serve
        the model (torch >= 2.10, head_dim <= 256, no sliding window). Opt in/out
        explicitly with ``sdpa_varlen``. When varlen is unavailable/unsuitable, stock SDPA
        stays and packing is still isolated via the dropped-mask block-diagonal path.
        """
        # False -> explicit opt-out; True -> explicit opt-in; None -> auto for sdpa packing.
        if self.cfg.sdpa_varlen is False:
            return
        explicit = self.cfg.sdpa_varlen is True
        auto = (
            self.cfg.sdpa_varlen is None
            and self.cfg.attn_implementation == "sdpa"
            and self.cfg.sample_packing
        )
        if not (explicit or auto):
            return

        from axolotl.monkeypatch.attention.sdpa_varlen import (
            _VARLEN_MAX_HEAD_DIM,
            patch_sdpa_varlen,
            varlen_available,
        )

        if not varlen_available():
            return  # torch < 2.10; block-diagonal packing path is correct

        def _attr(name):
            mc = self.model_config
            return mc.get(name) if isinstance(mc, dict) else getattr(mc, name, None)

        head_dim = _attr("head_dim")
        if not head_dim and _attr("hidden_size") and _attr("num_attention_heads"):
            head_dim = _attr("hidden_size") // _attr("num_attention_heads")
        sliding = _attr("sliding_window")
        layer_types = _attr("layer_types")
        uses_sliding = bool(sliding) and (
            any(lt == "sliding_attention" for lt in layer_types)
            if layer_types
            else True
        )

        if (head_dim and head_dim > _VARLEN_MAX_HEAD_DIM) or uses_sliding:
            if explicit:
                LOG.info(
                    "sdpa_varlen: model has head_dim > %d or a sliding window; keeping "
                    "stock SDPA (packing still isolated via the block-diagonal mask).",
                    _VARLEN_MAX_HEAD_DIM,
                )
            return

        patch_sdpa_varlen()

    def _apply_multipack_patches(self):
        """Apply multipack patches if necessary."""
        if (
            self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
            and self.cfg.attn_supports_packing
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

            if has_remote_code and self.cfg.trust_remote_code is not None:
                # If explicitly set in YAML, prefer that
                has_remote_code = self.cfg.trust_remote_code

            patch_for_multipack(
                self.cfg.model_config_type,
                model_name=self.cfg.base_model,
                has_remote_code=has_remote_code,
            )

        if self.cfg.sample_packing:
            from axolotl.monkeypatch.data.batch_dataset_fetcher import (
                apply_multipack_dataloader_patch,
            )

            LOG.info("Applying multipack dataloader patch for sample packing...")
            apply_multipack_dataloader_patch()

    def _apply_fsdp2_bnb_patches(self):
        """Apply FSDP2 BNB patches."""
        if (
            self.cfg.fsdp_config
            and str(self.cfg.fsdp_version) == "2"
            and (self.cfg.load_in_4bit or self.cfg.load_in_8bit)
        ):
            from axolotl.monkeypatch.fsdp2_qlora import (
                apply_init_dtype_attrs_patch,
                apply_init_sharded_param_patch,
                apply_init_unsharded_param_patch,
                apply_linear8bitlt_save_patch,
            )

            apply_init_sharded_param_patch()
            apply_init_unsharded_param_patch()
            apply_init_dtype_attrs_patch()
            if self.cfg.load_in_8bit:
                apply_linear8bitlt_save_patch()

    def _deactivate_hf_async_load(self):
        """Load weights synchronously so they can be converted and not OOM."""
        if self.cfg.load_in_4bit or self.cfg.load_in_8bit:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

    def _apply_moe_expert_quantization_patch(self):
        """Patch transformers weight loading and PEFT for MoE expert quantization."""
        has_target_params = bool(getattr(self.cfg, "lora_target_parameters", None))

        if not self.cfg.quantize_moe_experts and not has_target_params:
            return

        from axolotl.monkeypatch.moe_quant import (
            patch_peft_target_parameters_matching,
        )

        if self.cfg.quantize_moe_experts:
            from axolotl.monkeypatch.moe_quant import patch_moe_quantization_on_load

            patch_moe_quantization_on_load(self.cfg)

        patch_peft_target_parameters_matching()

    def _finalize_moe_expert_quantization(self, model: PreTrainedModel):
        """Log quantization results and set model flag for downstream use."""
        import torch

        model._moe_experts_quantized = False
        if self.cfg.quantize_moe_experts:
            from axolotl.monkeypatch.moe_quant import get_moe_quantized_count

            count = get_moe_quantized_count()
            if count > 0:
                import gc

                model._moe_experts_quantized = True
                LOG.info(
                    "Quantized %d MoE expert parameter(s) to %s during model loading",
                    count,
                    "4-bit" if self.cfg.load_in_4bit else "8-bit",
                )
                gc.collect()
                torch.cuda.empty_cache()

    def _apply_tiled_mlp(self, model_type: str):
        if self.cfg.tiled_mlp:
            from axolotl.monkeypatch.tiled_mlp import (
                patch_tiled_mlp,
            )

            patch_tiled_mlp(
                model_type,
                use_original_mlp=self.cfg.tiled_mlp_use_original_mlp,
                cfg_num_shards=self.cfg.tiled_mlp_num_shards,
                use_scattermoe=bool(self.cfg.use_scattermoe),
            )

    def _apply_tiled_mlp_post_load(self, model):
        """Re-wrap MoE block instances after kernels have installed their forward.

        Needed only when scattermoe-lora is active — ``model.kernelize()``
        binds ``HFScatterMoEGatedMLP.forward`` per instance, which shadows
        the class-level tiled patch. See
        :func:`axolotl.monkeypatch.tiled_mlp.patch_tiled_mlp_moe_instances`.
        """
        if not (self.cfg.tiled_mlp and self.cfg.use_scattermoe):
            return
        from axolotl.monkeypatch.tiled_mlp import patch_tiled_mlp_moe_instances

        patch_tiled_mlp_moe_instances(
            model,
            self.cfg.model_config_type,
            cfg_num_shards=self.cfg.tiled_mlp_num_shards,
        )

    def _apply_voxtral_patches(self):
        """Apply patches for Voxtral model."""
        if self.cfg.model_config_type == "voxtral":
            from axolotl.monkeypatch.models.voxtral.modeling import (
                patch_voxtral_conditional_generation_forward,
            )

            patch_voxtral_conditional_generation_forward()

    def _patch_attention(self):
        """Apply attention-specific patches based on model type."""
        if not (
            self.cfg.attn_uses_flash_lib and hasattr(self.model_config, "model_type")
        ):
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

        if self.model_config.model_type in ("mistral3", "llava"):
            from axolotl.monkeypatch.models.pixtral.modeling_flash_attention_utils import (
                apply_patch_is_packed_sequence,
            )

            apply_patch_is_packed_sequence()

    def _patch_loss_llama(self):
        """Patch loss functions and other optimizations for LLaMA models."""
        if not self.cfg.is_llama_derived_model:
            return

        if self.cfg.flash_attn_cross_entropy and self.has_flash_attn:
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                patch_fa_llama_cross_entropy,
            )

            patch_fa_llama_cross_entropy()

    def _patch_llama_xformers_attention(self):
        """Apply xformers attention patches for LLaMA models."""
        from axolotl.monkeypatch.llama_attn_hijack_xformers import (
            hijack_llama_attention,
        )

        LOG.info("Patching with xformers attention...")
        hijack_llama_attention()

    def _patch_llama_derived_model(self):
        """Modify all llama derived models in one block."""
        if (
            self.cfg.is_llama_derived_model
            and self.cfg.attn_implementation == "xformers"
            and not (
                self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
                and self.cfg.attn_supports_packing
                and self.cfg.sample_packing
            )
        ):
            self._patch_llama_xformers_attention()

    def _apply_llama_flash_attn_patches(self, model):
        """Apply LLaMA-specific flash attention patches."""

        if (
            self.model_config.model_type
            in ["llama", "llama4", "ernie4_5", "ernie4_5_moe"]
            and not self.cfg.trust_remote_code
            and not self.cfg.gptq
            and self.cfg.attn_uses_flash_lib
            and is_flash_attn_available()
            and not self.inference
        ):
            try:
                # TODO(MengqingCao): split these patches separately
                from axolotl.monkeypatch.llama_attn_hijack_flash import (
                    is_xformers_swiglu_available,
                    replace_llama_mlp_with_swiglu,
                )

                if self.cfg.flash_attn_fuse_mlp and is_xformers_swiglu_available():
                    LOG.info("Patching with SwiGLU...")
                    replace_llama_mlp_with_swiglu(model)
            except ImportError as e:
                LOG.warning(f"Flash Attention patches not applied: {e}")

    def _apply_lora_kernel_patch(self, model):
        """Apply LoRA kernel patches."""
        if (
            self.cfg.lora_mlp_kernel
            or self.cfg.lora_qkv_kernel
            or self.cfg.lora_o_kernel
        ):
            from axolotl.monkeypatch.lora_kernels import apply_lora_kernel_patches

            apply_lora_kernel_patches(model=model, cfg=self.cfg)

    def _apply_patch_deepspeed_zero3(self):
        try:
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

            from axolotl.monkeypatch.deepspeed_utils import apply_deepspeed_patches

            if self.cfg.activation_offloading is True and (
                is_deepspeed_zero3_enabled()
                or os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3"
            ):
                apply_deepspeed_patches()
        except ImportError as e:
            LOG.warning(f"DeepSpeed patches not applied: {e}")

    def _apply_apertus_patches(self):
        """Apply patches for Apertus model."""
        if self.cfg.model_config_type == "apertus":
            from axolotl.monkeypatch.models.apertus.activation import (
                patch_apertus_xielu_activation,
            )

            patch_apertus_xielu_activation()

    def _apply_trl_vllm_patches(self):
        """Apply TRL vLLM patches for batched weight sync, NaN logprobs fix, and scalar handling."""
        if (
            self.cfg.rl
            and getattr(self.cfg, "trl", None)
            and getattr(self.cfg.trl, "use_vllm", False)
        ):
            from axolotl.monkeypatch.trainer.trl_vllm import patch_trl_vllm

            patch_trl_vllm()

    def _apply_trl_trainer_utils_patches(self):
        """Replace trl.trainer.utils.{selective_log_softmax, entropy_from_logits} with Triton kernels."""
        if not self.cfg.rl:
            return

        try:
            from axolotl.monkeypatch.trainer.utils import (
                entropy_from_logits,
                selective_log_softmax,
            )
        except (ImportError, ModuleNotFoundError):
            LOG.warning("Triton not available — skipping trl.trainer.utils patches")
            return

        import trl.trainer.utils

        # Guard against repeated calls: only stash the original if trl still
        # points at its own implementation (not our wrapper).
        if trl.trainer.utils.selective_log_softmax is not selective_log_softmax:
            from axolotl.monkeypatch.trainer import utils as _axolotl_trainer_utils

            _axolotl_trainer_utils.selective_log_softmax_original = (
                trl.trainer.utils.selective_log_softmax
            )
            trl.trainer.utils.selective_log_softmax = selective_log_softmax

        if trl.trainer.utils.entropy_from_logits is not entropy_from_logits:
            trl.trainer.utils.entropy_from_logits = entropy_from_logits

        LOG.info(
            "Patched trl.trainer.utils with Triton selective_log_softmax and entropy_from_logits"
        )

    def _apply_scaling_softmax_patch(self, model: PreTrainedModel):
        """Apply Scaling Softmax (SSMax) patch.  Ref: https://arxiv.org/abs/2501.19399"""
        if self.cfg.scaling_softmax:
            from axolotl.monkeypatch.scaled_softmax_attn import (
                patch_scaled_softmax_attention,
            )

            patch_scaled_softmax_attention(
                scaling_factor_init=self.cfg.scaling_softmax_factor or 0.43,
                bias=self.cfg.scaling_softmax_bias or 0.0,
                model=model,
            )
