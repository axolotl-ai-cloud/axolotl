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
        if experts_implementation is None:
            # transformers may default to batched_mm when unset
            data["experts_implementation"] = "eager"
        elif experts_implementation != "eager":
            LOG.warning(
                "`experts_implementation` must be set to 'eager' to use this. Automatically setting it to 'eager'."
            )
            data["experts_implementation"] = "eager"

        return data
