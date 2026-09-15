"""Init for axolotl.cli.utils module."""

from axolotl.utils import make_lazy_getattr

from .args import (
    add_options_from_config,
    add_options_from_config_options,
    add_options_from_dataclass,
    filter_none_kwargs,
)
from .fetch import fetch_from_github
from .sweeps import generate_sweep_configs
from .train import build_command, generate_config_files, launch_training

__all__ = [
    "filter_none_kwargs",
    "add_options_from_dataclass",
    "add_options_from_config",
    "add_options_from_config_options",
    "build_command",
    "generate_config_files",
    "generate_sweep_configs",
    "load_model_and_tokenizer",
    "resolve_chat_template_str",
    "launch_training",
    "fetch_from_github",
]

_LAZY_IMPORTS = {
    "load_model_and_tokenizer": ".load",
    "resolve_chat_template_str": ".load",
}

__getattr__ = make_lazy_getattr(_LAZY_IMPORTS, __name__, globals())
