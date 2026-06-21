from pydantic import BaseModel, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# Valid experts_implementation values:
# - "eager"      : transformers' per-token loop reference implementation
# - "batched_mm" : transformers' built-in batched matmul path
# - "grouped_mm" : transformers' built-in grouped matmul path (cache-efficient)
# - "scattermoe" : axolotl-registered Triton kernels with LoRA support
# - "sonicmoe"   : axolotl-registered CUTLASS / cute-DSL kernels with LoRA support
# - "deep_ep[_*]": EP-plugin composites; passed through when expert_parallel_size > 1
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

    # --- legacy / low-level flags (kept for backwards compatibility) -------------------------
    use_scattermoe: bool | None = None
    use_sonicmoe: bool | None = None
    # Fused Triton training kernels for DeepSeek-V4 (attention / RoPE / mHC).
    use_dsv4_kernels: bool | None = None
    # DeepSeek-V4 FP8 non-expert weight storage: "float8tensor" (default, 1-byte torchao
    # Float8Tensor base) or "bf16" (dequantize to bf16 at load).
    dsv4_fp8_nonexpert_mode: str | None = None
    # Native blockwise-fp8 fused LoRA kernel for the large attention projections (q_b/o_b).
    # Off by default — the e2e gain is small for this expert-dominated model.
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
    # Gemma-4: NF4-quantize non-expert linears (bnb 4-bit, double-quant) after loading — the same
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
    def normalize_expert_backend(cls, data):
        """Map the intent ``expert_backend`` onto the legacy use_scattermoe/use_sonicmoe flags so
        the rest of the stack (which reads those) is unchanged. eager/builtin leave them unset."""
        eb = data.get("expert_backend")
        if eb is None:
            return data
        eb = str(eb).lower()
        valid = {"scattermoe", "sonicmoe", "eager", "builtin"}
        if eb not in valid:
            raise ValueError(
                f"expert_backend must be one of {sorted(valid)}, got {eb!r}"
            )
        if eb == "scattermoe":
            data["use_scattermoe"] = True
        elif eb == "sonicmoe":
            data["use_sonicmoe"] = True
        return data

    @model_validator(mode="before")
    @classmethod
    def check_nonexpert_quantization(cls, data):
        """Validate the non-expert quantization intent and warn on the deprecated per-model flags."""
        nq = data.get("nonexpert_quantization")
        if nq is not None:
            valid = {"none", "bf16", "fp8", "fp8_blockwise", "nf4"}
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
    def check_sonicmoe_ep_unsupported(cls, data):
        """SonicMoE + EP is not yet implemented (EP `_sonicmoe_local` raises)."""
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
        if data.get("use_scattermoe") is True or data.get("use_sonicmoe") is True:
            if data.get("lora_mlp_kernel") is True:
                # The generic lora_mlp_kernel (dense-MLP LoRA fusion) is incompatible with the
                # custom MoE expert kernels. On DSV4 it historically also drove the shared-expert
                # MLP fused LoRA; preserve that intent via the dedicated flag before disabling, so
                # the shared-MLP patch still runs (it reads dsv4_shared_mlp_lora_kernel).
                if (
                    data.get("use_dsv4_kernels") is True
                    and data.get("dsv4_shared_mlp_lora_kernel") is None
                ):
                    data["dsv4_shared_mlp_lora_kernel"] = True
                    LOG.warning(
                        "Translated lora_mlp_kernel -> dsv4_shared_mlp_lora_kernel for the DSV4 MoE "
                        "run (the generic lora_mlp_kernel is disabled under custom MoE kernels)."
                    )
                LOG.warning(
                    "Disabling lora_mlp_kernel when using custom MoE kernels due to compatibility issues."
                )
                data["lora_mlp_kernel"] = False
            data["mlp_kernel"] = False

        return data
