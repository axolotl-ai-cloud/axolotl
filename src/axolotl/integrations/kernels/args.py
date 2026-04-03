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
        use_scattermoe = data.get("use_scattermoe", False)
        if experts_implementation is None:
            # transformers may default to batched_mm when unset
            data["experts_implementation"] = "eager"
        elif experts_implementation == "scattermoe" and not use_scattermoe:
            LOG.warning(
                "`experts_implementation='scattermoe'` requires `use_scattermoe: true`. "
                "Automatically setting to 'eager'."
            )
            data["experts_implementation"] = "eager"
        elif experts_implementation not in ("eager", "scattermoe"):
            LOG.warning(
                f"`experts_implementation={experts_implementation!r}` is not compatible with "
                f"custom MoE kernels. Automatically setting to 'eager'."
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
