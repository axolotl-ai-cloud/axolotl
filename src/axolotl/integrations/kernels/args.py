from pydantic import BaseModel, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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
    def check_experts_implementation(cls, data):
        experts_implementation = data.get("experts_implementation")
        allowed = {"eager", "scattermoe"}
        if experts_implementation is None:
            # transformers may default to batched_mm when unset
            data["experts_implementation"] = "eager"
        elif experts_implementation not in allowed:
            LOG.warning(
                f"`experts_implementation={experts_implementation!r}` is not compatible with "
                f"custom MoE kernels (allowed: {allowed}). Automatically setting to 'eager'."
            )
            data["experts_implementation"] = "eager"

        return data

    @model_validator(mode="before")
    @classmethod
    def warn_sonicmoe_lora_overhead(cls, data):
        if data.get("use_sonicmoe") is True and data.get("adapter") in (
            "lora",
            "qlora",
        ):
            lora_target = data.get("lora_target_modules") or []
            lora_linear = data.get("lora_target_linear_modules") or []
            targets = (
                lora_target if isinstance(lora_target, list) else [lora_target]
            ) + (lora_linear if isinstance(lora_linear, list) else [lora_linear])
            expert_keywords = ("gate_up_proj", "down_proj", "experts")
            if any(kw in t for t in targets for kw in expert_keywords):
                LOG.info(
                    "SonicMoE + LoRA on expert modules uses runtime weight materialization "
                    "(W_eff = W + scaling*B@A per forward). This has slightly higher overhead "
                    "than ScatterMoE's fused Triton LoRA kernels but works with any CUTLASS kernel."
                )

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
