"""Construct tiny Mamba models through Axolotl registration and Transformers."""

from transformers import AutoModelForCausalLM

from axolotl.loaders.patch_manager import PatchManager
from axolotl.utils.dict import DictDefault


def _register(config):
    manager = PatchManager(DictDefault(model_config_type=config.model_type), config)
    manager._apply_model_support_registrations()


class MambaModelLoader:
    def __new__(cls, config, **kwargs):
        return cls.from_config(config, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        _register(config)
        return AutoModelForCausalLM.from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, path, *, config, **kwargs):
        _register(config)
        return AutoModelForCausalLM.from_pretrained(path, config=config, **kwargs)
