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
    use_scattermoe: bool | None = None
    use_sonicmoe: bool | None = None

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
                LOG.warning(
                    "Disabling lora_mlp_kernel when using custom MoE kernels due to compatibility issues."
                )
                data["lora_mlp_kernel"] = False
            data["mlp_kernel"] = False

        return data
