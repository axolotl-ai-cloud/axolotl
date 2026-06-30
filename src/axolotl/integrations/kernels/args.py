from pydantic import BaseModel, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# deep_ep[_*] are EP-plugin composites, passed through when expert_parallel_size > 1.
_BUILTIN_EXPERTS_IMPLS = {"eager", "batched_mm", "grouped_mm"}
_KERNEL_EXPERTS_IMPLS = {"scattermoe", "sonicmoe"}
_EP_EXPERTS_IMPLS = {
    "deep_ep",
    "deep_ep_grouped_mm",
    "deep_ep_scattermoe",
    "deep_ep_sonicmoe",
}
_VALID_EXPERTS_IMPLS = (
    _BUILTIN_EXPERTS_IMPLS | _KERNEL_EXPERTS_IMPLS | _EP_EXPERTS_IMPLS
)


class KernelsArgs(BaseModel):
    # --- intent-based capability surface (preferred) ----------------------------------------
    # MoE expert backend, an alias for use_scattermoe/use_sonicmoe so new models opt into a
    # capability by name rather than a vendor flag: scattermoe | sonicmoe | eager | builtin.
    expert_backend: str | None = None
    # Non-expert linear quantization policy (replaces the per-model gemma4_*_nonexpert flags):
    # none | bf16 | fp8_blockwise | nf4. Resolved per-model by the adapters.
    nonexpert_quantization: str | None = None
    # Grouped NVFP4 MoE base-GEMM backend selection: auto (capability-select; default) | marlin |
    # cutlass | deepgemm. An unavailable choice warns + falls back to auto.
    moe_grouped_backend: str | None = None

    # bnb-4bit MoE experts (quantize_moe_experts + load_in_4bit): number of experts dequantized to
    # bf16 per chunk in the chunked-dequant grouped path. None = fixed default (memory-safe for
    # smaller GPUs). Raise it on large GPUs to trade VRAM for throughput (bigger grouped GEMMs).
    moe_dequant_chunk_size: int | None = None

    # bnb-4bit MoE experts: route through the 1-launch parallel_linear (scatter2scatter) path instead
    # of the chunked torch._grouped_mm path. Faster (fewer kernel launches) at the same low memory --
    # the dequant'd bf16 is recomputed in backward via a recipe, not saved. None defaults to True; set
    # False to force the chunked path (bounds the per-pass transient for large-expert MoEs / tiny GPUs).
    moe_bnb_fast: bool | None = None

    # --- legacy / low-level flags (kept for backwards compatibility) -------------------------
    use_scattermoe: bool | None = None
    use_sonicmoe: bool | None = None
    # Fused Triton training kernels for DeepSeek-V4 (attention / RoPE / mHC).
    use_dsv4_kernels: bool | None = None
    # GLM-5.2 (glm_moe_dsa) DSA fused attention: MLA-absorption head-batched sparse-gather attn
    # (fwd+bwd) + fused Lightning-Indexer + length-aware dense/sparse dispatch, replacing the dense
    # [B,S,T]-mask eager/sdpa path. The router is kept fp32. Composes with use_scattermoe (experts)
    # and context_parallel_size (the attention shards the sequence on the cp axis).
    use_glm_dsa_kernels: bool | None = None
    # DeepSeek-V4 FP8 non-expert weight storage: "float8tensor" (default, 1-byte torchao
    # Float8Tensor base) or "bf16" (dequantize to bf16 at load).
    dsv4_fp8_nonexpert_mode: str | None = None
    # Native blockwise-fp8 fused LoRA kernel for the large attention projections (q_b/o_b).
    # Off by default; the e2e gain is small for this expert-dominated model.
    dsv4_fp8_lora_kernel: bool | None = None
    # Fused clamped-SwiGLU LoRA kernel for the DSV4 shared-expert MLPs. Distinct from the generic
    # `lora_mlp_kernel` (dense-MLP fusion, force-disabled under MoE kernels). A legacy
    # `lora_mlp_kernel: true` on a DSV4 run is translated into this flag (see disable_mlp_kernel).
    dsv4_shared_mlp_lora_kernel: bool | None = None
    # Fallback: cast ALL residual fp32 params (incl. keep_in_fp32 mHC/norms) to the compute
    # dtype for the fused kernels, instead of preserving keep_in_fp32 in fp32.
    dsv4_bf16_all: bool | None = None
    # Grouped fp4 MoE experts path (variable-M contiguous-grouped, base-fp4 GEMM + LoRA): off by
    # default (existing fused/chunked paths unchanged). "nvfp4" = fp4 act x fp4 weight (fastest,
    # lossy acts); "fp8" = fp8 act x mxfp4 weight (accurate). Auto-dispatches the base GEMM:
    # DeepGEMM (sm90/sm100) -> CUTLASS grouped (sm120) -> chunked-dequant fallback.
    dsv4_fp4_grouped_mode: str | None = None
    # Gemma-4 frankenstein: fp8-quantize non-expert linears in-place after loading (per-channel
    # e4m3, dequant-in-forward).  Experts remain NVFP4Tensor.  ~2 GB resident savings.
    gemma4_fp8_nonexpert: bool | None = None
    # Gemma-4: NF4-quantize non-expert linears (bnb 4-bit, double-quant) after loading, the same
    # non-expert compute path unsloth uses, for apples-to-apples experts-only LoRA comparison.
    gemma4_nf4_nonexpert: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def check_dsv4_fp4_grouped_mode(cls, data):
        mode = data.get("dsv4_fp4_grouped_mode")
        if mode is None:
            return data
        m = str(mode).lower()
        if m == "fp8":
            # Documented historically but never implemented in the training path
            # (grouped_fp4_available/_train_backend only support 'nvfp4'). Reject loudly rather
            # than silently no-op.
            raise ValueError(
                "dsv4_fp4_grouped_mode='fp8' is not implemented for training (only 'nvfp4' is "
                "supported). Use 'nvfp4' or omit dsv4_fp4_grouped_mode."
            )
        if m != "nvfp4":
            raise ValueError(f"dsv4_fp4_grouped_mode must be 'nvfp4', got {mode!r}")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_dsv4_fp8_nonexpert_mode(cls, data):
        mode = data.get("dsv4_fp8_nonexpert_mode")
        if mode is not None and str(mode).lower() not in ("float8tensor", "bf16"):
            raise ValueError(
                f"dsv4_fp8_nonexpert_mode must be 'float8tensor' or 'bf16', got {mode!r}"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_moe_dequant_chunk_size(cls, data):
        chunk = data.get("moe_dequant_chunk_size")
        if chunk is None:
            return data
        err = ValueError(
            f"moe_dequant_chunk_size must be a positive integer, got {chunk!r}"
        )
        if isinstance(chunk, bool):
            raise err
        try:
            chunk_int = int(chunk)
        except (TypeError, ValueError):
            raise err from None
        if isinstance(chunk, float) and not chunk.is_integer():
            raise err
        if chunk_int <= 0:
            raise err
        return data

    @staticmethod
    def _canonicalize_expert_backend(data):
        """Canonicalize the intent ``expert_backend`` onto use_scattermoe/use_sonicmoe (the rest of
        the stack reads those). ``expert_backend`` is authoritative: it sets the chosen backend True
        and REJECTS an explicit legacy flag that contradicts it (so the result is never ambiguous
        regardless of validator ordering). Idempotent: every consumer below calls this first because
        pydantic runs same-mode validators in REVERSE definition order, so the before-validator alone
        would run AFTER its consumers and they'd never see its writes."""
        eb = data.get("expert_backend")
        if eb is None:
            return data
        eb = str(eb).lower()
        valid = {"scattermoe", "sonicmoe", "eager", "builtin"}
        if eb not in valid:
            raise ValueError(
                f"expert_backend must be one of {sorted(valid)}, got {eb!r}"
            )
        want_scatter = eb == "scattermoe"
        want_sonic = eb == "sonicmoe"
        if data.get("use_scattermoe") is True and not want_scatter:
            raise ValueError(
                f"expert_backend={eb!r} conflicts with use_scattermoe=true; set only one."
            )
        if data.get("use_sonicmoe") is True and not want_sonic:
            raise ValueError(
                f"expert_backend={eb!r} conflicts with use_sonicmoe=true; set only one."
            )
        # Only the chosen backend is written, leaving the other untouched (None), so the end state is
        # byte-identical to setting the equivalent legacy flag directly. eager/builtin write nothing.
        if want_scatter:
            data["use_scattermoe"] = True
        elif want_sonic:
            data["use_sonicmoe"] = True
        return data

    @model_validator(mode="before")
    @classmethod
    def normalize_expert_backend(cls, data):
        return cls._canonicalize_expert_backend(data)

    @model_validator(mode="before")
    @classmethod
    def check_moe_grouped_backend(cls, data):
        backend = data.get("moe_grouped_backend")
        if backend is None:
            return data
        b = str(backend).lower()
        if b == "dequant":
            # The chunked-dequant fallback has no training/autograd path (the training dispatch only
            # wires marlin/deepgemm/cutlass); accepting it would silently run cutlass. Reject loudly.
            raise ValueError(
                "moe_grouped_backend='dequant' is not implemented for training (the chunked-dequant "
                "fallback has no autograd path). Use 'auto' (default), 'marlin', 'deepgemm', or "
                "'cutlass', or omit moe_grouped_backend."
            )
        valid = {"auto", "marlin", "cutlass", "deepgemm"}
        if b not in valid:
            raise ValueError(
                f"moe_grouped_backend must be one of {sorted(valid)}, got {backend!r}"
            )
        # The override only takes effect once the grouped NVFP4 MoE path is enabled.
        if not data.get("dsv4_fp4_grouped_mode"):
            LOG.warning(
                "moe_grouped_backend=%r has no effect unless the grouped NVFP4 MoE path is enabled "
                "(set dsv4_fp4_grouped_mode: nvfp4).",
                backend,
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_nonexpert_quantization(cls, data):
        """Validate the non-expert quantization intent and warn on the deprecated per-model flags."""
        nq = data.get("nonexpert_quantization")
        if nq is not None:
            valid = {"none", "bf16", "fp8", "fp8_blockwise", "nf4", "nvfp4"}
            if str(nq).lower() not in valid:
                raise ValueError(
                    f"nonexpert_quantization must be one of {sorted(valid)}, got {nq!r}"
                )
        if data.get("gemma4_fp8_nonexpert") or data.get("gemma4_nf4_nonexpert"):
            LOG.warning(
                "gemma4_fp8_nonexpert / gemma4_nf4_nonexpert are deprecated; prefer "
                "`nonexpert_quantization: fp8_blockwise` or `nonexpert_quantization: nf4`."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_mutually_exclusive(cls, data):
        data = cls._canonicalize_expert_backend(data)
        if data.get("use_scattermoe") and data.get("use_sonicmoe"):
            raise ValueError(
                "Cannot use both ScatterMoE and SonicMoE simultaneously. "
                "Please set only one of `use_scattermoe` or `use_sonicmoe` to true."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_use_kernels(cls, data):
        if data.get("use_kernels") is not True:
            LOG.warning(
                "`use_kernels` must be set to True to use this. Automatically setting it to True."
            )
            data["use_kernels"] = True

        return data

    @model_validator(mode="before")
    @classmethod
    def check_dsv4_attention_lora_unsupported(cls, data):
        """Reject module-level (attention) LoRA on a DSV4 fused-kernel run.

        The fused indexer feeds only gradientless topk indices, so LoRA on the indexer/scorer
        projections trains against no gradient; and attention-level LoRA reintroduces data-dependent
        FSDP2 backward collectives that break the experts-only invariant the DSV4 recipe relies on.
        Experts-only LoRA (lora_target_parameters) is the supported surface. This also covers
        lora_target_linear: true, which expands (find_all_linear_names) to every Linear including
        attention q/k/v/o. lora_exclude_modules is the explicit opt-out: setting it signals the user
        has excluded the indexer scorer projections, so we downgrade to a warning."""
        if not data.get("use_dsv4_kernels"):
            return data
        if not (data.get("lora_target_modules") or data.get("lora_target_linear")):
            return data
        if data.get("lora_exclude_modules"):
            LOG.warning(
                "attention/module-level LoRA (lora_target_modules / lora_target_linear) with "
                "use_dsv4_kernels: ensure lora_exclude_modules excludes the indexer scorer "
                "projections (the fused indexer is gradientless); keep LoRA experts-only "
                "(lora_target_parameters) where possible."
            )
            return data
        raise ValueError(
            "attention/module-level LoRA is not supported with use_dsv4_kernels (this includes "
            "lora_target_modules and lora_target_linear: true, which expands to all linear layers "
            "incl. attention q/k/v/o): the fused indexer is gradientless (topk indices only) and "
            "attention LoRA reintroduces data-dependent FSDP2 backward collectives that break the "
            "experts-only invariant. Either (1) keep LoRA experts-only via lora_target_parameters, "
            "or (2) explicitly exclude the indexer scorer projections via lora_exclude_modules."
        )

    @model_validator(mode="before")
    @classmethod
    def check_sonicmoe_ep_unsupported(cls, data):
        """SonicMoE + EP is not yet implemented (EP `_sonicmoe_local` raises)."""
        data = cls._canonicalize_expert_backend(data)
        if data.get("use_sonicmoe") and (data.get("expert_parallel_size") or 1) > 1:
            raise ValueError(
                "use_sonicmoe=true is not supported with expert_parallel_size > 1. "
                "Use use_scattermoe=true under EP, or set expert_parallel_size=1."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_experts_implementation(cls, data):
        """Auto-select impl from kernel flags; reject mismatched/unknown values."""
        data = cls._canonicalize_expert_backend(data)
        experts_implementation = data.get("experts_implementation")
        use_scattermoe = bool(data.get("use_scattermoe"))
        use_sonicmoe = bool(data.get("use_sonicmoe"))

        if experts_implementation is None:
            if use_scattermoe:
                data["experts_implementation"] = "scattermoe"
            elif use_sonicmoe:
                data["experts_implementation"] = "sonicmoe"
            else:
                # Transformers defaults to a non-eager backend when unset; pin to
                # eager unless the user explicitly opts in.
                data["experts_implementation"] = "eager"
            return data

        if experts_implementation == "scattermoe" and not use_scattermoe:
            LOG.warning(
                "`experts_implementation='scattermoe'` requires `use_scattermoe: true`. "
                "Automatically setting to 'eager'."
            )
            data["experts_implementation"] = "eager"
        elif experts_implementation == "sonicmoe" and not use_sonicmoe:
            LOG.warning(
                "`experts_implementation='sonicmoe'` requires `use_sonicmoe: true`. "
                "Automatically setting to 'eager'."
            )
            data["experts_implementation"] = "eager"
        elif experts_implementation not in _VALID_EXPERTS_IMPLS:
            LOG.warning(
                f"`experts_implementation={experts_implementation!r}` is not recognized. "
                f"Valid options: {sorted(_VALID_EXPERTS_IMPLS)}. "
                f"Automatically setting to 'eager'."
            )
            data["experts_implementation"] = "eager"

        return data

    @model_validator(mode="before")
    @classmethod
    def disable_mlp_kernel(cls, data):
        data = cls._canonicalize_expert_backend(data)
        if data.get("use_scattermoe") is True or data.get("use_sonicmoe") is True:
            # DSV4's shared/routed expert MLP needs the dedicated clamped-SwiGLU kernel, not the
            # generic dense-MLP one; translate the intent and disable the generic path for DSV4.
            if (
                data.get("lora_mlp_kernel") is True
                and data.get("use_dsv4_kernels") is True
            ):
                if data.get("dsv4_shared_mlp_lora_kernel") is None:
                    data["dsv4_shared_mlp_lora_kernel"] = True
                    LOG.warning(
                        "Translated lora_mlp_kernel -> dsv4_shared_mlp_lora_kernel for the DSV4 MoE "
                        "run (the generic lora_mlp_kernel is disabled under DSV4 kernels)."
                    )
                data["lora_mlp_kernel"] = False
            # Otherwise keep lora_mlp_kernel: under the custom MoE expert kernels it only fuses the
            # DENSE shared MLP (layer.mlp via find_mlp_in_layer), which is a plain gated Linear MLP
            # separate from the routed experts (those are handled by the MoE kernel and aren't PEFT
            # nn.Linear, so the can_patch_mlp lora_A guard skips them). The fused kernel dequantizes
            # bnb-4bit / fp8 bases, so it composes with quantized non-experts.
            data["mlp_kernel"] = False  # the non-LoRA mlp kernel stays off under MoE

        return data
