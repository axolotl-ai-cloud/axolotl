"""Select the implementation without changing Transformers' global registries."""


def model_class(config):
    backend = getattr(config, "mamba_backend", "transformers")
    if backend not in ("transformers", "fla"):
        raise ValueError("model_config.mamba_backend must be transformers or fla")
    if backend == "fla":
        from .modeling import FlaMamba2ForCausalLM, FlaMambaForCausalLM

        return {"mamba": FlaMambaForCausalLM, "mamba2": FlaMamba2ForCausalLM}[
            config.model_type
        ]
    from transformers import Mamba2ForCausalLM, MambaForCausalLM

    return {"mamba": MambaForCausalLM, "mamba2": Mamba2ForCausalLM}[config.model_type]


class MambaModelLoader:
    """ModelSupport loader preserving the selected backend in saved configs."""

    def __new__(cls, config, **kwargs):
        return cls.from_config(config, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        implementation = model_class(config)
        if getattr(config, "mamba_backend", "transformers") == "transformers":
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_config(config, **kwargs)
        kwargs.pop("trust_remote_code", None)
        return implementation(config, **kwargs)

    @classmethod
    def from_pretrained(cls, path, *, config, **kwargs):
        implementation = model_class(config)
        if getattr(config, "mamba_backend", "transformers") == "transformers":
            from transformers import AutoModelForCausalLM

            implementation = AutoModelForCausalLM
        return implementation.from_pretrained(path, config=config, **kwargs)
