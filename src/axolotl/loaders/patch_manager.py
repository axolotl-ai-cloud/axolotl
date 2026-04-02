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
        if (
            hasattr(cfg, "base_model_config")
            and cfg.base_model_config
            and "kimi-linear" in cfg.base_model_config.lower()
        ):
            from axolotl.monkeypatch.models.kimi_linear.patch_kimi_linear import (
                patch_kimi_config,
            )

            patch_kimi_config()

    @staticmethod
    def apply_pre_tokenizer_load_patches(cfg: DictDefault):
        """
        Apply patches that must be set up before tokenizer loading.
        This is for patches that intercept remote code loading from HuggingFace,
        which needs to be in place before AutoTokenizer.from_pretrained() is called.

        Args:
            cfg: Configuration dictionary with model and training settings.
        """
        if (
            hasattr(cfg, "tokenizer_config")
            and cfg.tokenizer_config
            and "kimi-linear" in cfg.tokenizer_config.lower()
        ):
            from axolotl.monkeypatch.models.kimi_linear.patch_kimi_linear import (
                patch_kimi_tokenizer,
            )

            patch_kimi_tokenizer()

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

        if self.cfg.context_parallel_size > 1:
            from axolotl.monkeypatch.transformers.trainer_context_parallel import (
                patch_prepare_context_parallel_inputs,
            )

            patch_prepare_context_parallel_inputs()

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
        self._finalize_moe_expert_quantization(model)

    def apply_post_model_load_patches(self, model: PreTrainedModel):
        """Apply patches that require the model instance."""
        self._apply_llama_flash_attn_patches(model)
        self._apply_lora_kernel_patch(model)
        self._apply_scaling_softmax_patch(model)

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

        from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

        patch_gemma4_hybrid_mask()

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
                # Global / full_attention layer - use SDPA instead of FA2
                attn_module = getattr(layer, "self_attn", None)
                if attn_module is not None and hasattr(attn_module, "config"):
                    sdpa_config = copy.copy(attn_module.config)
                    sdpa_config._attn_implementation = "sdpa"
                    attn_module.config = sdpa_config
                    patched_count += 1

        LOG.info(
            "gemma4_hybrid_attn_impl: patched %d global layers to use SDPA "
            "(remaining %d sliding layers use flash_attention_2)",
            patched_count,
            len(layers) - patched_count,
        )

    def _apply_flash_attention_patches(self):
        """Apply patches related to Flash Attention."""
        if self.cfg.xformers_attention and self.cfg.sample_packing:
            from axolotl.monkeypatch.attention import patch_xformers_attn_over_fa2

            patch_xformers_attn_over_fa2()
            self.cfg.flash_attention = True

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

        if self.cfg.context_parallel_size > 1 or (
            self.cfg.fsdp_config and str(self.cfg.fsdp_version) == "2"
        ):
            from axolotl.monkeypatch.accelerate.parallelism_config import (
                patch_parallelism_config,
            )

            patch_parallelism_config()
        if self.cfg.fsdp_config and str(self.cfg.fsdp_version) == "2":
            from axolotl.monkeypatch.accelerate.fsdp2 import (
                patch_accelerate_fsdp2,
                patch_tied_keys_for_meta_device,
            )

            patch_accelerate_fsdp2()
            if self.cfg.fsdp_config.cpu_ram_efficient_loading:
                patch_tied_keys_for_meta_device()
            if self.cfg.rl:
                from axolotl.monkeypatch.trainer.trl import patch_trl_prepare_fsdp2

                patch_trl_prepare_fsdp2()

        # if self.cfg.fsdp_config:
        #     # see transformers#39152
        #     from axolotl.monkeypatch.trainer_fsdp_optim import (
        #         patch_training_loop_for_fsdp,
        #     )
        #
        #     patch_training_loop_for_fsdp()

    def _apply_adapter_patches(self):
        """Apply patches for adapter configurations."""
        if self.cfg.adapter and self.cfg.embeddings_skip_upcast:
            from axolotl.monkeypatch.peft.utils import patch_peft_prep_code

            patch_peft_prep_code()

    def _apply_flex_attention_patches(self):
        """Apply patches for flexible attention."""
        if self.cfg.flex_attention:
            from axolotl.monkeypatch.attention.flex_attn import (
                patch_flex_wrapper,
            )

            flex_attn_compile_kwargs = self.cfg.flex_attn_compile_kwargs or {}
            patch_flex_wrapper(**flex_attn_compile_kwargs)

    def _apply_sageattn_patches(self):
        """Apply patches for SageAttention."""
        if self.cfg.sage_attention:
            from axolotl.monkeypatch.attention.sage_attn import patch_sageattn

            patch_sageattn()

    def _apply_flash_attn_4_patches(self):
        """Auto-apply FA4 when flash_attention is enabled and FA4 is available on SM90+."""
        if not self.cfg.flash_attention:
            return

        from axolotl.monkeypatch.attention.flash_attn_4 import patch_flash_attn_4

        patch_flash_attn_4(self.model_config)

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

        if self.cfg.model_config_type == "kimi_linear":
            from axolotl.monkeypatch.models.kimi_linear.patch_kimi_linear import (
                patch_kimi_model,
            )

            patch_kimi_model()

        if self.cfg.model_config_type in ("nemotron_h", "falcon_h1"):
            # Prefer the installed mamba_ssm over Hub-cached kernels: the Hub build
            # targets an older causal_conv1d API incompatible with causal_conv1d>=1.5.
            try:
                import mamba_ssm
                from transformers.integrations.hub_kernels import _KERNEL_MODULE_MAPPING

                _KERNEL_MODULE_MAPPING.setdefault("mamba-ssm", mamba_ssm)
            except ImportError:
                pass

        if self.cfg.model_config_type == "nemotron_h":
            if self.cfg.sample_packing:
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

        if self.cfg.model_config_type == "falcon_h1":
            if self.cfg.sample_packing:
                from axolotl.monkeypatch.models.falcon_h1.modeling import (
                    patch_falcon_h1_modeling_packing,
                )

                patch_falcon_h1_modeling_packing()
                # FalconH1PreTrainedModel already sets supports_gradient_checkpointing=True
                # (FalconH1DecoderLayer inherits GradientCheckpointingLayer), so no
                # manual override is needed here.

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
                and self.cfg.flash_attention
            ):
                from axolotl.monkeypatch.models.qwen3_5.modeling import (
                    patch_qwen3_5_vlm_flash_attention,
                )

                patch_qwen3_5_vlm_flash_attention()

            if self.cfg.model_config_type in ("gemma4", "gemma4_text"):
                # The fused attn path is now compatible with
                # ``gemma4_hybrid_attn_impl``: the kernel handles partial
                # rotary (cos.shape[-1] < head_dim) and the fused forward
                # mirrors the current ``Gemma4TextAttention.forward`` API
                # for shared kv (read from / write to
                # ``past_key_values.shared_layers``). See
                # ``src/axolotl/kernels/GEMMA4_FUSED_ROPE_HYBRID_ATTN_BUG.md``
                # for the history.
                from axolotl.monkeypatch.models.gemma4.fused_attn import (
                    patch_gemma4_fused_attn,
                )

                # Shared-KV side channel when activation checkpointing (PR #3611).
                fsdp_cfg = self.cfg.fsdp_config
                needs_shared_kv_workaround = (not self.inference) and bool(
                    self.cfg.gradient_checkpointing
                    or self.cfg.activation_offloading
                    or (fsdp_cfg is not None and fsdp_cfg.activation_checkpointing)
                )
                patch_gemma4_fused_attn(
                    install_shared_kv_workaround=needs_shared_kv_workaround
                )

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
            from axolotl.monkeypatch.trainer_accelerator_args import (
                patch_create_accelerate_code_for_fp8,
            )

            patch_create_accelerate_code_for_fp8(
                self.cfg.fp8_enable_fsdp_float8_all_gather
            )

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

            # nemotron_h and falcon_h1 are native transformers types (5.x+); their Hub
            # repos still carry auto_map pointing to legacy remote code that imports
            # removed symbols. Exclude them regardless of auto_map presence.
            has_remote_code = (
                auto_map_config is not None
                and "AutoModelForCausalLM" in auto_map_config
                and self.cfg.model_config_type not in ("nemotron_h", "falcon_h1")
            )
            if self.cfg.trust_remote_code is not None:
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
        if self.cfg.flash_attn_rms_norm and self.has_flash_attn:
            from axolotl.monkeypatch.llama_attn_hijack_flash import patch_llama_rms_norm

            patch_llama_rms_norm()

    def _patch_llama_flash_attention(self):
        """Apply Flash Attention patches for LLaMA models."""
        from axolotl.monkeypatch.llama_attn_hijack_flash import (
            replace_llama_attn_with_flash_attn,
        )

        if self.cfg.s2_attention:
            LOG.info("patching w/ flash-enabled, shifted-sparse attention")
            replace_llama_attn_with_flash_attn(
                cross_entropy=self.cfg.flash_attn_cross_entropy,
                rms_norm=self.cfg.flash_attn_rms_norm,
                use_shifted_sparse_attn=True,
            )
        elif self.cfg.flash_attn_cross_entropy or self.cfg.flash_attn_rms_norm:
            replace_llama_attn_with_flash_attn(
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

    def _patch_llama_derived_model(self):
        """Modify all llama derived models in one block."""
        if self.cfg.is_llama_derived_model and not (
            self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
            and (self.cfg.flash_attention or self.cfg.flex_attention)
            and self.cfg.sample_packing
        ):
            if self.cfg.flash_attention:
                self._patch_llama_flash_attention()
            elif self.cfg.xformers_attention:
                self._patch_llama_xformers_attention()
            elif self.cfg.s2_attention:
                raise NotImplementedError(
                    "Shifted-sparse attention not currently implemented without flash attention."
                )

    def _apply_llama_flash_attn_patches(self, model):
        """Apply LLaMA-specific flash attention patches."""
        if (
            self.model_config.model_type
            in ["llama", "llama4", "ernie4_5", "ernie4_5_moe"]
            and not self.cfg.trust_remote_code
            and not self.cfg.gptq
            and self.cfg.flash_attention
            and is_flash_attn_available()
            and not self.inference
        ):
            # TODO(MengqingCao): split these patches separately
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                is_xformers_swiglu_available,
                replace_llama_mlp_with_swiglu,
            )

            if self.cfg.flash_attn_fuse_mlp and is_xformers_swiglu_available():
                LOG.info("Patching with SwiGLU...")
                replace_llama_mlp_with_swiglu(model)

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
