"""Pydantic input args for the Gemma3 text-from-multimodal plugin."""

from pydantic import BaseModel, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class Gemma3TextFromMultimodalArgs(BaseModel):
    """Configuration args for loading a Gemma3 multimodal checkpoint as text-only."""

    gemma3_text_from_multimodal: bool = True
    extract_text_config: bool = False

    @model_validator(mode="before")
    @classmethod
    def set_model_type(cls, data):
        if not isinstance(data, dict):
            return data

        if not data.get("gemma3_text_from_multimodal", True):
            return data

        if not data.get("model_type"):
            LOG.info(
                "Gemma3TextFromMultimodalPlugin: auto-setting model_type to Gemma3ForCausalLM"
            )
            data["model_type"] = "Gemma3ForCausalLM"

        return data
