"""CPU unit tests for the ringmaster context-parallel plugin (no distributed)."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from axolotl.integrations.context_parallel import (
    ContextParallelConfig,
    ContextParallelPlugin,
)
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

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
    with pytest.raises(ValidationError):
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


def test_strip_logits_to_keep_pre_hook():
    hook = ContextParallelPlugin._strip_logits_to_keep_pre_hook
    args, kwargs = hook(None, (), {"logits_to_keep": 42, "input_ids": "x"})
    assert "logits_to_keep" not in kwargs and kwargs["input_ids"] == "x"
    # zero (keep-all) and non-int (mask) values pass through
    _, kwargs = hook(None, (), {"logits_to_keep": 0})
    assert kwargs["logits_to_keep"] == 0
    _, kwargs = hook(None, (), {"num_logits_to_keep": 7})
    assert "num_logits_to_keep" not in kwargs


@pytest.mark.parametrize(
    "settings",
    [
        {"context_parallel_size": 4},
        {"sequence_parallel_degree": 4},
        {"context_parallel": {"size": 4}},
        {"context_parallel_size": 4, "context_parallel": {"backend": "ring"}},
    ],
)
def test_builtin_cp_without_plugin_entry(min_base_cfg, settings):
    from axolotl.integrations.base import BUILTIN_PLUGINS, PluginManager

    cfg = min_base_cfg | DictDefault(settings, attn_implementation="sdpa")
    validated = validate_config(cfg)
    assert validated.context_parallel_size == 4
    assert validated.context_parallel.size == 4
    assert not validated.plugins
    assert PLUGIN_PATH in BUILTIN_PLUGINS
    assert PLUGIN_PATH in PluginManager.get_instance().plugins


@pytest.mark.parametrize("explicit", [False, True])
def test_builtin_preparation_is_idempotent(min_base_cfg, explicit):
    from axolotl.cli.config import prepare_plugins as cli_prepare_plugins
    from axolotl.integrations.base import PluginManager

    cfg = min_base_cfg | DictDefault(
        context_parallel_size=4, attn_implementation="sdpa"
    )
    if explicit:
        cfg.plugins = [PLUGIN_PATH]
    cfg = validate_config(cfg)
    plugin = PluginManager.get_instance().plugins[PLUGIN_PATH]
    for prepare in (prepare_plugins, cli_prepare_plugins):
        prepare(cfg)
        assert PluginManager.get_instance().plugins[PLUGIN_PATH] is plugin
        assert cfg.context_parallel.size == 4
    assert cfg.plugins == ([PLUGIN_PATH] if explicit else None)


def test_builtin_is_inactive_without_cp(min_base_cfg):
    from axolotl.integrations.base import PluginManager

    validated = validate_config(min_base_cfg)
    plugin = PluginManager.get_instance().plugins[PLUGIN_PATH]
    assert not plugin._enabled(validated)
    assert not validated.plugins
    plugin.pre_model_load(validated)


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


def test_size_sync_preserves_backend(min_base_cfg):
    cfg = validate_config(
        min_base_cfg
        | DictDefault(
            context_parallel_size=4,
            context_parallel={"backend": "ring"},
            attn_implementation="sdpa",
        )
    )
    assert cfg.context_parallel.backend == "ring"
    assert cfg.context_parallel.size == cfg.context_parallel_size == 4


@pytest.mark.parametrize("nested,flat", [(1, 4), (4, 1), (4, 8)])
def test_explicit_size_conflicts(min_base_cfg, nested, flat):
    cfg = min_base_cfg | DictDefault(
        context_parallel_size=flat, context_parallel={"size": nested}
    )
    with pytest.raises(ValueError, match="conflicts"):
        validate_config(cfg)


def test_schema_rejects_disabled_nested_cp(min_base_cfg):
    from axolotl.integrations.context_parallel.args import ContextParallelArgs
    from axolotl.utils.schemas.config import AxolotlInputConfig

    class Config(ContextParallelArgs, AxolotlInputConfig):
        pass

    with pytest.raises(ValueError, match="conflicts"):
        Config(
            **(
                min_base_cfg
                | dict(
                    context_parallel_size=4,
                    context_parallel={"size": 1},
                    plugins=[PLUGIN_PATH],
                    attn_implementation="sdpa",
                )
            )
        )


@pytest.mark.parametrize("size", [1, 4])
def test_prepare_after_validation_preserves_cp_settings(min_base_cfg, size):
    from axolotl.integrations.base import PluginManager

    cfg = validate_config(
        min_base_cfg
        | DictDefault(context_parallel_size=size, attn_implementation="sdpa")
    )
    prepare_plugins(cfg)
    normalize_config(cfg)
    plugin = PluginManager.get_instance().plugins[PLUGIN_PATH]
    assert plugin._cp_cfg(cfg).size == size
    assert plugin._enabled(cfg) is (size > 1)
    if size == 1:
        plugin.pre_model_load(cfg)


@pytest.mark.parametrize("size", [1, 4])
def test_runtime_accepts_plain_cp_dict(size):
    cfg = DictDefault()
    cfg["context_parallel"] = {"size": size, "backend": "ulysses"}
    plugin = ContextParallelPlugin()
    assert plugin._cp_cfg(cfg).backend == "ulysses"
    assert plugin._enabled(cfg) is (size > 1)


def test_prepare_after_validation_without_cp_is_inactive(min_base_cfg):
    from axolotl.integrations.base import PluginManager

    cfg = validate_config(min_base_cfg)
    prepare_plugins(cfg)
    normalize_config(cfg)
    plugin = PluginManager.get_instance().plugins[PLUGIN_PATH]
    plugin.pre_model_load(cfg)
    plugin.post_model_load(cfg, None)
    plugin.post_trainer_create(cfg, None)
    assert not plugin._enabled(cfg)


def test_validation_does_not_call_register(min_base_cfg, monkeypatch):
    from unittest.mock import Mock

    from axolotl.integrations.base import PluginManager

    plugin = PluginManager.get_instance().plugins[PLUGIN_PATH]
    register = Mock()
    monkeypatch.setattr(plugin, "register", register)
    cfg = validate_config(
        min_base_cfg | DictDefault(context_parallel_size=4, attn_implementation="sdpa")
    )
    register.assert_not_called()
    assert cfg.context_parallel.size == 4
    prepare_plugins(cfg)
    register.assert_called_once_with(cfg)


def test_no_plugins_without_builtins(min_base_cfg, monkeypatch):
    from unittest.mock import Mock

    from axolotl.cli.config import plugin_set_cfg
    from axolotl.integrations import base

    monkeypatch.setattr(base, "BUILTIN_PLUGINS", ())
    merge = Mock(side_effect=AssertionError("plugin schema should not be merged"))
    monkeypatch.setattr("axolotl.utils.config.merge_input_args", merge)
    cfg = validate_config(min_base_cfg)
    prepare_plugins(cfg)
    plugin_set_cfg(cfg)
    merge.assert_not_called()
    assert not cfg.plugins


@pytest.mark.parametrize("heads,tp,local", [(8, 4, 2), (8, 2, 4), (2, 4, 1), (8, 1, 8)])
def test_tp_local_heads_drive_ulysses_selection(heads, tp, local):
    from types import SimpleNamespace

    rm = pytest.importorskip("ringmaster")

    model = SimpleNamespace(config=SimpleNamespace(num_key_value_heads=heads))
    actual = ContextParallelPlugin._num_kv_heads(model, tp_size=tp)
    assert actual == local
    config = rm.RingmasterConfig(size=4)
    config.normalize(num_kv_heads=actual, intra_node_size=4)
    assert config.ulysses_size <= local
    assert local % config.ulysses_size == 0


@pytest.mark.parametrize(
    "backend,u,r,glm,expected",
    [
        ("ulysses", 4, 1, False, ("none", "all_to_all")),
        ("ring", 1, 4, False, ("head_tail", "p2p")),
        ("auto", 1, 4, True, ("none", "glm_dsa")),
    ],
)
@pytest.mark.parametrize("rotate", [None, "allgather"])
def test_explicit_options_survive_config_lifecycle(
    min_base_cfg, backend, u, r, glm, expected, rotate
):
    pytest.importorskip("ringmaster")
    from axolotl.integrations.context_parallel.settings import resolve_settings

    block = {"size": 4, "backend": backend}
    if rotate is not None:
        block["rotate_method"] = rotate
    cfg = min_base_cfg | DictDefault(context_parallel=block, attn_implementation="sdpa")
    for iteration in range(2):
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        if iteration == 1:
            normalize_config(cfg)
        cp = ContextParallelPlugin._cp_cfg(cfg)
        assert ("rotate_method" in cp.model_fields_set) is (rotate is not None)
        resolved = SimpleNamespace(ulysses_size=u, ring_size=r)
        if rotate is not None and (glm or r == 1):
            with pytest.raises(ValueError, match="only apply|GLM DSA owns"):
                resolve_settings(cp, resolved, num_kv_heads=8, glm_dsa=glm)
        else:
            communication = resolve_settings(cp, resolved, num_kv_heads=8, glm_dsa=glm)
            assert (resolved.load_balance.value, communication) == (
                ("none", "allgather") if rotate is not None else expected
            )


@pytest.mark.parametrize(
    "builtin,explicit", [(True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize("fails", [False, True])
def test_cli_plugin_cleanup_guard(monkeypatch, builtin, explicit, fails):
    from unittest.mock import Mock

    from axolotl.cli import checks, train as cli_train
    from axolotl.integrations import base

    monkeypatch.setattr(base, "BUILTIN_PLUGINS", (PLUGIN_PATH,) if builtin else ())
    manager = Mock()
    manager.load_datasets.return_value = None
    get_manager = Mock(return_value=manager)
    monkeypatch.setattr(base.PluginManager, "get_instance", get_manager)
    monkeypatch.setattr(checks, "check_accelerate_default_config", Mock())
    monkeypatch.setattr(checks, "check_user_token", Mock())
    monkeypatch.setattr(cli_train, "load_datasets", Mock(return_value="dataset"))
    monkeypatch.setattr(cli_train, "load_preference_datasets", Mock())
    train = Mock(return_value=(object(), object(), object()))
    monkeypatch.setattr(cli_train, "train", train)
    cfg = DictDefault(plugins=[PLUGIN_PATH] if explicit else None)
    if fails:
        train.side_effect = RuntimeError("training failed")
        with pytest.raises(RuntimeError, match="training failed"):
            cli_train.do_train(cfg, SimpleNamespace())
    else:
        cli_train.do_train(cfg, SimpleNamespace())
    train.assert_called_once_with(cfg=cfg, dataset_meta="dataset")
    if builtin or explicit:
        manager.post_train_unload.assert_called_once_with(cfg)
    else:
        get_manager.assert_not_called()


def test_fla_cp_requires_companion_adapter_before_setup(monkeypatch):
    import builtins
    from unittest.mock import Mock

    rm = pytest.importorskip("ringmaster")
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "ringmaster.fla_mamba":
            raise ImportError("adapter unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    setup = Mock()
    monkeypatch.setattr(rm, "setup", setup)
    model = SimpleNamespace(
        config=SimpleNamespace(mamba_backend="fla", num_key_value_heads=4)
    )
    trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace())
    cfg = _cfg(size=2, backend="ulysses")
    with pytest.raises(ImportError, match="FLA Mamba adapters"):
        ContextParallelPlugin().post_trainer_create(cfg, trainer)
    setup.assert_not_called()
