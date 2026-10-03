"""Configuration exposed by the diffusion training plugin."""

from collections.abc import Mapping

from pydantic import BaseModel, Field, model_validator

from .schema import DiffusionConfig, DiffusionLMConfig


class DiffusionArgs(BaseModel):
    """Diffusion configuration under one runtime key."""

    diffusion: DiffusionLMConfig | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def normalize_diffusion_alias(cls, value):
        return normalize_diffusion_blocks(value)

    @model_validator(mode="after")
    def validate_diffusion_attention(self):
        diffusion = self.diffusion
        if diffusion is None or diffusion.from_causal_lm:
            return self
        if "attn_implementation" not in type(self).model_fields:
            return self
        attention = self.attn_implementation
        if attention == "varlen":
            import torch
            from packaging.version import Version

            env = getattr(self, "env_capabilities", None)
            version = (
                env.get("torch_version")
                if isinstance(env, Mapping)
                else getattr(env, "torch_version", None)
            ) or torch.__version__
            if Version(str(version).split("+", maxsplit=1)[0]) < Version("2.14.0"):
                raise ValueError("attn_implementation: varlen requires torch >= 2.14")
        return self


__all__ = ["DiffusionArgs", "DiffusionConfig", "DiffusionLMConfig"]


def normalize_diffusion_blocks(value):
    """Normalize the deprecated block before plugin args are validated."""
    if not isinstance(value, Mapping):
        return value
    result = dict(value)
    if "diffusion" in result and "diffusion_lm" in result:
        raise ValueError("Configure only one of `diffusion` and `diffusion_lm`.")
    canonical = result.get("diffusion")
    alias = result.pop("diffusion_lm", None)
    selected = alias if alias is not None else canonical
    if selected is None:
        return result
    if isinstance(selected, Mapping):
        block = dict(selected)
    elif callable(getattr(selected, "model_dump", None)):
        block = selected.model_dump()
    else:
        raise TypeError("diffusion configuration must be a mapping")
    block.setdefault("from_causal_lm", alias is None)
    result["diffusion"] = block
    return result
