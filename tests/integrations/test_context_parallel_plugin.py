"""CPU unit tests for the ringmaster context-parallel plugin (no distributed)."""

from types import SimpleNamespace

import pytest

from axolotl.integrations.context_parallel import (
    ContextParallelConfig,
    ContextParallelPlugin,
)
from axolotl.utils.config import ensure_context_parallel_plugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.enums import RLType

PLUGIN_PATH = "axolotl.integrations.context_parallel.ContextParallelPlugin"


def _cfg(**cp):
    block = ContextParallelConfig(**cp) if cp else None
    return SimpleNamespace(
        context_parallel=block,
        flash_attention=False,
        attn_implementation=None,
        gradient_accumulation_steps=1,
    )


def test_get_input_args():
    assert (
        ContextParallelPlugin().get_input_args()
        == "axolotl.integrations.context_parallel.args.ContextParallelArgs"
    )


def test_enabled_flag():
    plugin = ContextParallelPlugin()
    assert plugin._enabled(_cfg(size=8)) is True
    assert plugin._enabled(_cfg(size=1)) is False
    assert plugin._enabled(_cfg()) is False


def test_config_degree_product_validation():
    with pytest.raises(Exception):
        ContextParallelConfig(size=8, ulysses_size=4, ring_size=4)
    ContextParallelConfig(size=8, ulysses_size=2, ring_size=4)  # ok


def test_resolve_inner_attn():
    plugin = ContextParallelPlugin()
    # nothing configured -> safe sdpa default, no silent flash forcing
    assert plugin._resolve_inner_attn(_cfg(size=8)) == "sdpa"
    flash_cfg = _cfg(size=8)
    flash_cfg.flash_attention = True
    assert plugin._resolve_inner_attn(flash_cfg) == "flash_attention_2"
    fa3 = _cfg(size=8)
    fa3.attn_implementation = "flash_attention_3"
    assert plugin._resolve_inner_attn(fa3) == "flash_attention_3"


def test_register_syncs_block_and_flat_size():
    plugin = ContextParallelPlugin()

    cfg = {"context_parallel": {"size": 8}}
    plugin.register(cfg)
    assert cfg["context_parallel_size"] == 8

    cfg = {"context_parallel_size": 4}
    plugin.register(cfg)
    assert cfg["context_parallel"] == {"size": 4}

    with pytest.raises(ValueError, match="conflicts"):
        plugin.register({"context_parallel": {"size": 8}, "context_parallel_size": 4})


def test_gather_outputs_enum_detection():
    """cfg.rl is an RLType enum after validation; GRPO/EBFT must enable gathering."""
    assert RLType.GRPO in (RLType.GRPO, RLType.EBFT)
    assert str(RLType.GRPO).lower() != "grpo"  # the bug this guards against
    assert "grpo" in (RLType.GRPO, RLType.EBFT)  # str-enum equality both ways


def test_strip_logits_to_keep_pre_hook():
    hook = ContextParallelPlugin._strip_logits_to_keep_pre_hook
    args, kwargs = hook(None, (), {"logits_to_keep": 42, "input_ids": "x"})
    assert "logits_to_keep" not in kwargs and kwargs["input_ids"] == "x"
    # zero (keep-all) and non-int (mask) values pass through
    _, kwargs = hook(None, (), {"logits_to_keep": 0})
    assert kwargs["logits_to_keep"] == 0
    _, kwargs = hook(None, (), {"num_logits_to_keep": 7})
    assert "num_logits_to_keep" not in kwargs


def test_ensure_context_parallel_plugin_auto_enables():
    cfg = DictDefault({"context_parallel_size": 4})
    ensure_context_parallel_plugin(cfg)
    assert any("ContextParallelPlugin" in str(p) for p in cfg["plugins"])
    assert cfg["context_parallel"] == {"size": 4}

    # deprecated alias also enables the plugin
    cfg = DictDefault({"sequence_parallel_degree": 4})
    ensure_context_parallel_plugin(cfg)
    assert any("ContextParallelPlugin" in str(p) for p in cfg["plugins"])
    assert cfg["context_parallel"] == {"size": 4}

    # nested-only block enables the plugin and backfills the flat size
    cfg = DictDefault({"context_parallel": {"size": 8}})
    ensure_context_parallel_plugin(cfg)
    assert any("ContextParallelPlugin" in str(p) for p in cfg["plugins"])
    assert cfg["context_parallel_size"] == 8

    # explicitly-listed plugin with only the flat key still gets the block
    cfg = DictDefault({"plugins": [PLUGIN_PATH], "context_parallel_size": 4})
    ensure_context_parallel_plugin(cfg)
    assert cfg["plugins"].count(PLUGIN_PATH) == 1
    assert cfg["context_parallel"] == {"size": 4}

    with pytest.raises(ValueError, match="conflicts"):
        ensure_context_parallel_plugin(
            DictDefault({"context_parallel": {"size": 8}, "context_parallel_size": 4})
        )

    # disabled configs are untouched
    cfg = DictDefault({"context_parallel_size": 1})
    ensure_context_parallel_plugin(cfg)
    assert not cfg.get("plugins")


def test_plugin_merges_into_axolotl_schema():
    """The plugin's context_parallel args merge into axolotl's input config schema."""
    from axolotl.integrations.base import PluginManager
    from axolotl.integrations.config import merge_input_args

    pm = PluginManager.get_instance()
    pm.register("axolotl.integrations.context_parallel.ContextParallelPlugin")
    _wcap, input_cfg_cls = merge_input_args()

    # The plugin's nested context_parallel block is now part of axolotl's schema and
    # parses a dict from YAML into a validated ContextParallelConfig.
    assert "context_parallel" in input_cfg_cls.model_fields
    field = input_cfg_cls.model_fields["context_parallel"]
    cp = field.annotation.__args__[0](  # Optional[ContextParallelConfig]
        **{"size": 8, "backend": "auto", "ring_impl": "hf_kernels"}
    )
    assert cp.size == 8 and cp.ring_impl == "hf_kernels"


def test_num_kv_heads_reads_text_config():
    plugin = ContextParallelPlugin()
    model = SimpleNamespace(
        config=SimpleNamespace(num_key_value_heads=2, num_attention_heads=16)
    )
    assert plugin._num_kv_heads(model) == 2

    # multimodal: nested text_config
    mm = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(num_key_value_heads=4, num_attention_heads=32)
        )
    )
    assert plugin._num_kv_heads(mm) == 4

    # real HF configs resolve through get_text_config()
    from transformers import LlamaConfig

    model = SimpleNamespace(
        config=LlamaConfig(num_key_value_heads=8, num_attention_heads=32)
    )
    assert plugin._num_kv_heads(model) == 8
