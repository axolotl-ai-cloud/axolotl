"""Load-time mapping for native diffusion varlen attention."""

from types import SimpleNamespace

from axolotl.loaders.model import ModelLoader


def test_native_varlen_loads_transformers_model_eagerly():
    loader = ModelLoader.__new__(ModelLoader)
    loader.cfg = SimpleNamespace(attn_implementation="varlen", low_cpu_mem_usage=False)
    loader.model_config = SimpleNamespace()
    loader.model_kwargs = {}

    loader._set_attention_config()

    assert loader.cfg.attn_implementation == "varlen"
    assert loader.model_kwargs["attn_implementation"] == "eager"
    assert loader.model_config._attn_implementation == "eager"
