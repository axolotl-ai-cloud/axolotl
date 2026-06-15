"""Patch manager class implementation to complement `axolotl.loaders.ModelLoader`.

Applies pre- and post-model load patches for various fixes and optimizations.
"""

import importlib.util
import os
import re
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
        # Must precede fused-RoPE patches: re-parses ``Attention.forward``
        # via ``inspect.getsource``; the QKV regex misses on a patched body.
        self._apply_self_attention_lora_patch()
        self._apply_model_specific_patches()
        self._apply_fp8_patches()
        self._apply_flash_attention_peft_patches()
        self._apply_gradient_checkpointing_patches()
        self._patch_attention()
        self._apply_multipack_patches()
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

    def apply_post_model_load_patches(self, model: PreTrainedModel):
        """Apply patches that require the model instance."""
        self._apply_llama_flash_attn_patches(model)
        self._apply_lora_kernel_patch(model)
        self._apply_scaling_softmax_patch(model)
        self._apply_fp8_attention_patches(model)
        self._apply_nvfp4_training(model)
        self._apply_tiled_mlp_post_load(model)
        self._mark_nvfp4_ddp_ignore(model)

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

    def _apply_nvfp4_training(self, model: PreTrainedModel):
        """Swap eligible linears for NVFP4-GEMM linears (Blackwell FP4 compute).

        Runs in post-load (after weights + any merge AND after PEFT wraps the
        model in ``ModelLoader._load_adapters``) so the swap sees real linear
        modules in their final tree position. FFT swaps raw ``nn.Linear``;
        adapter modes swap the FROZEN base_layer inside each ``lora.Linear``,
        which only exists once PEFT has wrapped the model — hence post-load.
        """
        nvfp4 = self.cfg.nvfp4_training
        if not (nvfp4 and nvfp4.enabled):
            return

        # In-process merge (legacy merge-lora) writes base_layer.weight.data +=
        # delta; the FP4 base modules expose weight read-only, so that write
        # would silently no-op. Keep the base in bf16 for merge — merge_and_unload
        # then merges into the real weight. (FP4 training is irrelevant at merge.)
        if self.cfg.merge_lora:
            return

        from axolotl.kernels.lora import set_nvfp4_shared_base_fprop
        from axolotl.utils.nvfp4_training import (
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
            convert_to_nvfp4_training,
        )

        recipe = NVFP4Recipe(
            stochastic_rounding=nvfp4.stochastic_rounding,
            hadamard=nvfp4.hadamard,
        )
        shared_lora_base_fprop = set_nvfp4_shared_base_fprop(
            getattr(nvfp4, "shared_lora_base_fprop", None)
        )
        exclude_modules = list(nvfp4.exclude_modules or [])
        if getattr(nvfp4, "quantize_lm_head", False):
            exclude_modules = self._nvfp4_unexclude_lm_head(model, exclude_modules)
        exclude = tuple(exclude_modules) + self._nvfp4_block_exclusions(
            model, nvfp4.skip_first_n_blocks or 0, nvfp4.skip_last_n_blocks or 0
        )

        adapter = self.cfg.adapter
        if shared_lora_base_fprop:
            LOG.info("NVFP4 LoRA shared base fprop enabled")

        # Transformer Engine backend: swap eligible linears to te.Linear; the
        # trainer wraps the step in te.fp8_autocast (set up below via a stored
        # recipe). FFT swaps raw nn.Linear; LoRA swaps the frozen base_layer
        # inside each lora.Linear (adapters stay high-precision).
        if getattr(nvfp4, "backend", "native") == "te":
            from axolotl.utils.nvfp4_training import (
                convert_lora_base_to_te_nvfp4,
                convert_to_te_nvfp4_training,
                te_nvfp4_available,
                te_nvfp4_recipe,
            )

            ok, reason = te_nvfp4_available()
            if not ok:
                raise RuntimeError(reason)
            # te keeps an HP (bf16) base and quantizes on the fly: it has no
            # FP4-storage path, so a requested storage/compute base_mode would
            # silently give no memory saving. Steer the user to backend: native.
            if adapter in ("lora", "qlora") and (
                nvfp4.quantize_base
                or getattr(nvfp4, "base_mode", None) in ("storage", "compute")
            ):
                LOG.warning(
                    "nvfp4_training.backend: te ignores base_mode/quantize_base "
                    "(te keeps a high-precision base, no FP4 storage saving). Use "
                    "backend: native for FP4-stored/compute LoRA bases."
                )
            if adapter in ("lora", "qlora"):
                count = convert_lora_base_to_te_nvfp4(model, recipe, exclude=exclude)
                empty_msg = (
                    "nvfp4_training(te) enabled but no eligible LoRA base layers "
                    "were swapped (is the model PEFT-wrapped?)"
                )
            elif adapter:
                raise ValueError(
                    f"nvfp4_training.backend: te supports full fine-tune or "
                    f"lora/qlora, not adapter={adapter}."
                )
            else:
                count = convert_to_te_nvfp4_training(model, recipe, exclude=exclude)
                empty_msg = "nvfp4_training(te) enabled but no nn.Linear swapped"
            model._te_nvfp4_recipe = te_nvfp4_recipe(recipe)
            if count == 0:
                LOG.warning(empty_msg)
            return

        if adapter in ("lora", "qlora"):
            # Resolve the base mode. Explicit base_mode wins; otherwise
            # qlora/quantize_base => FP4 storage, else default to FP4 compute
            # (pre-quantized base, fastest, the recommended LoRA path).
            base_mode = getattr(nvfp4, "base_mode", None)
            if base_mode is None:
                base_mode = (
                    "storage"
                    if (bool(nvfp4.quantize_base) or adapter == "qlora")
                    else "compute"
                )
            compute_base = base_mode == "compute"
            quantized_storage = base_mode == "storage"
            # The MSLK fast path wraps quant in registered custom ops; the torchao
            # fallback is pure-torch quant (nvfp4_quantize / _sr_dither) + an aten
            # _scaled_mm GEMM, which is also compile-safe (verified: a compiled
            # compute-base step with MSLK forced off completes with no graph breaks
            # beyond the model's own data-dependent ones). MSLK is just faster, so
            # surface it as info, not a warning.
            if compute_base and self.cfg.torch_compile:
                from axolotl.utils.nvfp4_training import _mslk_available

                if not _mslk_available():
                    LOG.info(
                        "nvfp4_training compute-base under torch_compile is using the "
                        "torchao fallback (MSLK not installed); this is compile-safe "
                        "but slower. Install MSLK for the faster custom-op quant path."
                    )
            # Both FP4 base modes need the NVFP4 all-gather hooks to shard under
            # FSDP2 (storage: one layout; compute: fprop+dgrad layouts).
            use_fsdp = (quantized_storage or compute_base) and bool(
                self.cfg.fsdp_config
            )
            count = convert_lora_base_to_nvfp4(
                model,
                recipe,
                quantized_storage=quantized_storage,
                compute_base=compute_base,
                fsdp=use_fsdp,
                exclude=exclude,
            )
            empty_msg = (
                "nvfp4_training enabled but no eligible LoRA base layers were "
                "swapped (is the model PEFT-wrapped?)"
            )
        else:
            base_mode = getattr(nvfp4, "base_mode", None) or "compute"
            count = convert_to_nvfp4_training(model, recipe, exclude=exclude)
            empty_msg = (
                "nvfp4_training enabled but no eligible nn.Linear layers were swapped"
            )
        if count == 0:
            LOG.warning(empty_msg)

        # lm_head / input-embedding / tied-shared-weight swaps (each opt-in). The
        # LoRA base converter only touches lora.Linear base_layers, so a frozen
        # lm_head/embedding that isn't a LoRA target is invisible to it and is
        # handled here.
        self._nvfp4_apply_tied_or_lm_head(model, recipe, base_mode)

        # Vision-tower encoder linears (multimodal, opt-in): frozen nn.Linear
        # under the vision module stay bf16 otherwise (not lora.Linear).
        if getattr(nvfp4, "quantize_vision_tower", False):
            from axolotl.utils.nvfp4_training import convert_vision_tower_to_nvfp4

            convert_vision_tower_to_nvfp4(model, recipe, base_mode=base_mode)

        # Fuse decoder RMSNorm + activation quant into one kernel so the base
        # linear reuses the norm's pre-quantized activation (native backend only;
        # the fused norm emits single-level FP4, matching the compute-base path).
        if getattr(nvfp4, "fuse_rmsnorm", True):
            from axolotl.utils.nvfp4_training import convert_norms_to_nvfp4_fused

            convert_norms_to_nvfp4_fused(model)

        self._nvfp4_load_packed_sidecar(model)

        # Fused FP4 lm_head + cross-entropy: skip materializing the [M, vocab]
        # logits (memory win). Opt-in and only when the lm_head became an FP4
        # store above; falls back to the materialized CE path otherwise.
        if getattr(nvfp4, "fused_fp4_cross_entropy", False) and getattr(
            nvfp4, "quantize_lm_head", False
        ):
            from axolotl.kernels.nvfp4_fused_ce import patch_model_fused_fp4_ce

            patch_model_fused_fp4_ce(
                model,
                fp4_matmul=True
                if getattr(nvfp4, "fused_fp4_cross_entropy_fp4_matmul", False)
                else None,
                vocab_block=getattr(nvfp4, "fused_ce_vocab_block", None),
            )

        if getattr(nvfp4, "fp8_lm_head", False):
            from axolotl.kernels.fp8_lm_head import patch_model_fp8_lm_head

            patch_model_fp8_lm_head(
                model,
                granularity=getattr(nvfp4, "fp8_lm_head_granularity", "rowwise"),
            )

        if getattr(nvfp4, "fp8_lm_head_cross_entropy", False):
            from axolotl.kernels.fp8_fused_ce import (
                patch_model_fp8_lm_head_cross_entropy,
            )

            patch_model_fp8_lm_head_cross_entropy(
                model,
                granularity=getattr(nvfp4, "fp8_lm_head_granularity", "rowwise"),
            )

        if getattr(nvfp4, "bf16_lm_head_cross_entropy", False):
            from axolotl.kernels.bf16_fused_ce import (
                patch_model_bf16_lm_head_cross_entropy,
            )

            patch_model_bf16_lm_head_cross_entropy(model)

    def _mark_nvfp4_ddp_ignore(self, model: PreTrainedModel):
        """Exclude NVFP4 frozen-base buffers from DDP's param/buffer sync.

        DDP NCCL-broadcasts module states across ranks (at init and, with
        broadcast_buffers, every step), but NCCL has no support for the packed
        ``Float4_e2m1fn_x2`` / fp8-scale dtypes the NVFP4 base stores — it raises
        "Input tensor data type is not supported for NCCL process group". Those
        buffers are frozen and bit-identical on every rank (deterministic quant),
        so they never need syncing; naming them in
        ``_ddp_params_and_buffers_to_ignore`` (read natively by DDP) skips them.
        """
        nvfp4 = self.cfg.nvfp4_training
        if not (nvfp4 and nvfp4.enabled):
            return

        exotic = {torch.float8_e4m3fn, torch.float8_e5m2}
        fp4 = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4 is not None:
            exotic.add(fp4)

        ignore = [
            name
            for name, buf in model.named_buffers()
            if buf is not None
            and (type(buf).__name__ == "NVFP4Tensor" or buf.dtype in exotic)
        ]
        if not ignore:
            return

        existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
        model._ddp_params_and_buffers_to_ignore = list(dict.fromkeys(existing + ignore))
        LOG.info("NVFP4: excluded %d FP4 base buffers from DDP sync", len(ignore))

    def _nvfp4_load_packed_sidecar(self, model: PreTrainedModel):
        """Restore FP4-packed weights from a save_nvfp4 sidecar, if one exists.

        Looks in resume_from_checkpoint first (resume), then the base model dir
        (loading a save_nvfp4-exported model). No-op when no sidecar is present —
        the frozen base otherwise reconstructs deterministically from the bf16
        weights, so this only matters for save_nvfp4 exports / exact FP4 reload.
        """
        import os

        from axolotl.utils.nvfp4_training import (
            NVFP4_PACKED_SIDECAR,
            load_nvfp4_packed,
        )

        candidates = [self.cfg.resume_from_checkpoint, self.cfg.base_model]
        for cand in candidates:
            if not cand or not isinstance(cand, str):
                continue
            if os.path.isfile(os.path.join(cand, NVFP4_PACKED_SIDECAR)):
                load_nvfp4_packed(model, cand)
                return

    @staticmethod
    def _nvfp4_block_exclusions(
        model: PreTrainedModel, skip_first: int, skip_last: int
    ) -> tuple[str, ...]:
        """Translate skip_first/last_n_blocks into ``layers.<i>.`` name fragments.

        Block count is only known here (the model is built), so the block-range
        policy is resolved in the integration layer and passed to the swap as
        explicit ``exclude`` fragments.
        """
        if skip_first <= 0 and skip_last <= 0:
            return ()

        block_re = re.compile(r"(.*\blayers)\.(\d+)\.")
        prefixes: dict[str, set[int]] = {}
        for name, _ in model.named_modules():
            m = block_re.match(name)
            if m:
                prefixes.setdefault(m.group(1), set()).add(int(m.group(2)))

        fragments: list[str] = []
        for prefix, indices in prefixes.items():
            ordered = sorted(indices)
            skip = set(ordered[:skip_first])
            if skip_last > 0:
                skip |= set(ordered[len(ordered) - skip_last :])
            fragments.extend(f"{prefix}.{i}." for i in sorted(skip))
        return tuple(fragments)

    def _nvfp4_unexclude_lm_head(
        self, model: PreTrainedModel, exclude_modules: list[str]
    ) -> list[str]:
        """Drop ``lm_head`` from the NVFP4 exclusion so the converter swaps it.

        Tied embeddings are handled by the quantize-once path (see
        ``_nvfp4_apply_tied_or_lm_head``), so a FROZEN tied weight is allowed
        here; a TRAINABLE tied weight still raises (FP4-storing it would corrupt
        training). A fused-linear cross-entropy path (cut_cross_entropy) consumes
        the lm_head weight directly and bypasses the NVFP4 forward — raise. If the
        lm_head dims aren't FP4-swappable (%32), leave it excluded with a warning
        rather than crash. Only ``lm_head`` is removed; ``embed_tokens`` stays.
        """
        from axolotl.utils.nvfp4_training import _is_swappable

        if self._model_ties_embeddings(model):
            if self._tied_weight_trainable(model):
                raise RuntimeError(
                    "nvfp4_training.quantize_lm_head with tied embeddings requires a "
                    "FROZEN shared weight: the output and input embeddings share one "
                    "weight, and FP4-storing a TRAINABLE shared weight would corrupt "
                    "training. Freeze the embedding (e.g. use LoRA, which freezes the "
                    "base), or set quantize_lm_head: false."
                )
            # Frozen tied: the shared weight is quantized once and routed to both
            # roles in _nvfp4_apply_tied_or_lm_head; the name-fragment exclusion is
            # irrelevant for the tied path, so return unchanged.
            return exclude_modules

        nvfp4_cfg = self.cfg.nvfp4_training
        want_fused_ce = bool(getattr(nvfp4_cfg, "fused_fp4_cross_entropy", False))
        if self.cfg.cut_cross_entropy and not want_fused_ce:
            raise RuntimeError(
                "nvfp4_training.quantize_lm_head is incompatible with "
                "cut_cross_entropy: the fused linear cross-entropy kernel reads the "
                "lm_head weight directly to fuse the projection with the loss, which "
                "bypasses the NVFP4 lm_head forward (the FP4 head would be ignored, "
                "or the kernel would fail on the NVFP4 module's missing .weight). "
                "Disable one of them (cut_cross_entropy: false or "
                "quantize_lm_head: false), or set "
                "nvfp4_training.fused_fp4_cross_entropy: true to use the FP4-aware "
                "fused cross-entropy (reads the NVFP4-packed lm_head directly)."
            )

        out_emb = model.get_output_embeddings()
        if not isinstance(out_emb, torch.nn.Linear) or not _is_swappable(out_emb):
            in_f = getattr(out_emb, "in_features", "?")
            out_f = getattr(out_emb, "out_features", "?")
            LOG.warning(
                "nvfp4_training.quantize_lm_head: lm_head is not NVFP4-swappable "
                "(in=%s out=%s, both must be divisible by 32); keeping it in high "
                "precision.",
                in_f,
                out_f,
            )
            return exclude_modules

        without_lm_head = [m for m in exclude_modules if m != "lm_head"]
        if "lm_head" in exclude_modules:
            LOG.info(
                "nvfp4_training.quantize_lm_head: removing lm_head from the "
                "high-precision exclusion (it will be quantized to NVFP4)."
            )
        return without_lm_head

    @staticmethod
    def _nvfp4_swap_frozen_lm_head(model, recipe, base_mode: str) -> None:
        """Swap a bare frozen lm_head (LoRA, not a target module) to NVFP4.

        Locates the output-embedding module by identity in the (possibly
        PEFT-wrapped) tree. If it's already an NVFP4 module (e.g. the user added
        lm_head to lora_target_modules and the LoRA converter handled it), this
        is a no-op.
        """
        import torch.nn as nn

        from axolotl.utils.nvfp4_training import swap_frozen_linear_to_nvfp4

        out_emb = model.get_output_embeddings()
        if not isinstance(out_emb, nn.Linear):
            return  # already swapped (NVFP4 module) or wrapped — nothing bare to do
        name = next(
            (n for n, m in model.named_modules() if m is out_emb),
            None,
        )
        if name is None:
            LOG.warning(
                "nvfp4_training.quantize_lm_head: could not locate the lm_head "
                "module in the model tree; leaving it in high precision."
            )
            return
        swap_frozen_linear_to_nvfp4(model, name, recipe, base_mode=base_mode)

    @staticmethod
    def _model_ties_embeddings(model: PreTrainedModel) -> bool:
        """Detect weight tying between the output and input embeddings.

        Checks both the config flag and weight identity — a model can tie via
        config or via a shared parameter object even if the flag is stale.
        """
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            return True
        try:
            out_emb = model.get_output_embeddings()
            in_emb = model.get_input_embeddings()
        except (AttributeError, NotImplementedError):
            return False
        out_w = getattr(out_emb, "weight", None)
        in_w = getattr(in_emb, "weight", None)
        return out_w is not None and in_w is not None and out_w is in_w

    @staticmethod
    def _tied_weight_trainable(model: PreTrainedModel) -> bool:
        """Whether the shared (tied) embedding weight requires grad.

        FP4-storing a trainable shared weight would corrupt training, so the
        quantize-once path is gated on this being False.
        """
        try:
            in_w = getattr(model.get_input_embeddings(), "weight", None)
        except (AttributeError, NotImplementedError):
            return False
        return bool(getattr(in_w, "requires_grad", False))

    def _nvfp4_apply_tied_or_lm_head(self, model, recipe, base_mode: str) -> None:
        """Route the tied / lm_head / embedding NVFP4 swaps post linear-conversion.

        Three independent flags (all OFF by default):
        - tied + quantize_lm_head (frozen): quantize the SHARED weight once and
          point both the embedding lookup and the lm_head GEMM at it.
        - quantize_lm_head (untied): swap the bare frozen lm_head.
        - quantize_embeddings: swap the frozen input embedding (also covers the
          tied case when quantize_lm_head is off — the shared weight is stored
          FP4 for the lookup, lm_head left HP).
        """
        import torch.nn as nn

        from axolotl.utils.nvfp4_training import (
            swap_frozen_embedding_to_nvfp4,
            swap_tied_embedding_and_lm_head_to_nvfp4,
        )

        nvfp4 = self.cfg.nvfp4_training
        want_lm_head = bool(getattr(nvfp4, "quantize_lm_head", False))
        want_embed = bool(getattr(nvfp4, "quantize_embeddings", False))
        if not (want_lm_head or want_embed):
            return

        tied = self._model_ties_embeddings(model)

        if tied and want_lm_head:
            # Frozen-tied is guaranteed here (the trainable case raised in
            # _nvfp4_unexclude_lm_head). Quantize the shared weight once.
            in_name = self._module_name(model, model.get_input_embeddings())
            out_name = self._module_name(model, model.get_output_embeddings())
            if in_name and out_name:
                swap_tied_embedding_and_lm_head_to_nvfp4(
                    model, in_name, out_name, recipe
                )
            return

        if want_lm_head:
            # The fused FP4 cross-entropy needs a row-sliceable (non-swizzled)
            # lm_head store; force the torchao storage class for it. Otherwise use
            # the requested base mode (compute/storage/hp).
            if bool(getattr(nvfp4, "fused_fp4_cross_entropy", False)):
                import torch.nn as _nn

                from axolotl.utils.nvfp4_training import swap_frozen_lm_head_tileable

                out_emb = model.get_output_embeddings()
                if isinstance(out_emb, _nn.Linear):
                    name = self._module_name(model, out_emb)
                    if name:
                        swap_frozen_lm_head_tileable(model, name, recipe)
            else:
                self._nvfp4_swap_frozen_lm_head(model, recipe, base_mode)

        if want_embed:
            in_emb = model.get_input_embeddings()
            if isinstance(in_emb, nn.Embedding):
                in_name = self._module_name(model, in_emb)
                if in_name:
                    swap_frozen_embedding_to_nvfp4(model, in_name)

    @staticmethod
    def _module_name(model: PreTrainedModel, target) -> str | None:
        if target is None:
            return None
        return next((n for n, m in model.named_modules() if m is target), None)

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

        if self.cfg.model_config_type == "kimi_linear":
            from axolotl.monkeypatch.models.kimi_linear.patch_kimi_linear import (
                patch_kimi_model,
            )

            patch_kimi_model()

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

                nvfp4 = getattr(self.cfg, "nvfp4_training", None)
                patch_qwen3_5_modeling_packing(
                    fla_causal_conv_compile_boundary=bool(
                        nvfp4
                        and nvfp4.enabled
                        and nvfp4.fla_causal_conv_compile_boundary
                    )
                )

            if self.cfg.model_config_type == "qwen3_5_moe" and self.cfg.sample_packing:
                from axolotl.monkeypatch.models.qwen3_5.modeling import (
                    patch_qwen3_5_moe_modeling_packing,
                )

                nvfp4 = getattr(self.cfg, "nvfp4_training", None)
                patch_qwen3_5_moe_modeling_packing(
                    fla_causal_conv_compile_boundary=bool(
                        nvfp4
                        and nvfp4.enabled
                        and nvfp4.fla_causal_conv_compile_boundary
                    )
                )

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
