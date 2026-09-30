"""Installed training plugins are discovered from package entry points."""

from importlib.metadata import EntryPoint

import pytest
from pydantic import BaseModel

from axolotl.integrations import base
from axolotl.integrations.base import BasePlugin
from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

EXPERT_PLUGIN_PATH = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"


class EntryPointPluginArgs(BaseModel):
    entry_point_plugin_enabled: bool = False


class EntryPointPlugin(BasePlugin):
    registered_configs: list[DictDefault] = []

    def get_input_args(self):
        return "tests.integrations.test_plugin_entry_points.EntryPointPluginArgs"

    def register(self, cfg):
        self.registered_configs.append(cfg)


def test_installed_plugin_is_loaded_and_registered_without_yaml_selection(
    min_base_cfg, monkeypatch
):
    point = EntryPoint(
        name="entry-point-plugin",
        value="tests.integrations.test_plugin_entry_points:EntryPointPlugin",
        group=base.PLUGIN_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(base, "BUILTIN_PLUGINS", ())
    monkeypatch.setattr(base, "entry_points", lambda **kwargs: [point])
    monkeypatch.setattr(base.PluginManager, "_instance", None)
    EntryPointPlugin.registered_configs = []

    cfg = validate_config(min_base_cfg | DictDefault(entry_point_plugin_enabled=True))
    prepare_plugins(cfg)

    assert cfg.entry_point_plugin_enabled is True
    assert list(base.PluginManager.get_instance().plugins) == [
        "tests.integrations.test_plugin_entry_points.EntryPointPlugin"
    ]
    assert EntryPointPlugin.registered_configs == [cfg]


def test_plugin_entry_points_are_deduplicated(monkeypatch):
    point = EntryPoint(
        name="context-parallel",
        value=base.BUILTIN_PLUGINS[0],
        group=base.PLUGIN_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(base, "entry_points", lambda **kwargs: [point])

    assert base.get_builtin_plugins() == base.BUILTIN_PLUGINS


def test_plugin_entry_points_are_cached_until_reset(monkeypatch):
    point = EntryPoint(
        name="entry-point-plugin",
        value="tests.integrations.test_plugin_entry_points:EntryPointPlugin",
        group=base.PLUGIN_ENTRY_POINT_GROUP,
    )
    calls = 0

    def discover(**_kwargs):
        nonlocal calls
        calls += 1
        return [point]

    monkeypatch.setattr(base, "BUILTIN_PLUGINS", ())
    monkeypatch.setattr(base, "entry_points", discover)
    base.reset_plugin_entry_points_cache()

    assert base.get_builtin_plugins() == (
        "tests.integrations.test_plugin_entry_points.EntryPointPlugin",
    )
    assert base.get_builtin_plugins() == (
        "tests.integrations.test_plugin_entry_points.EntryPointPlugin",
    )
    assert calls == 1

    base.reset_plugin_entry_points_cache()
    base.get_builtin_plugins()
    assert calls == 2


def test_expert_parallel_plugin_is_loaded_with_safe_defaults(min_base_cfg, monkeypatch):
    monkeypatch.setattr(base, "entry_points", lambda **kwargs: [])
    monkeypatch.setattr(base.PluginManager, "_instance", None)

    cfg = validate_config(min_base_cfg)
    prepare_plugins(cfg)

    assert EXPERT_PLUGIN_PATH in base.BUILTIN_PLUGINS
    assert cfg.expert_parallel_size == 1
    assert EXPERT_PLUGIN_PATH in base.PluginManager.get_instance().plugins


def test_configured_entry_point_plugin_is_registered_once(monkeypatch):
    point = EntryPoint(
        name="entry-point-plugin",
        value="tests.integrations.test_plugin_entry_points:EntryPointPlugin",
        group=base.PLUGIN_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(base, "BUILTIN_PLUGINS", ())
    monkeypatch.setattr(base, "entry_points", lambda **kwargs: [point])
    monkeypatch.setattr(base.PluginManager, "_instance", None)
    EntryPointPlugin.registered_configs = []

    cfg = DictDefault(plugins=[point.value])
    prepare_plugins(cfg)

    assert list(base.PluginManager.get_instance().plugins) == [
        "tests.integrations.test_plugin_entry_points.EntryPointPlugin"
    ]
    assert EntryPointPlugin.registered_configs == [cfg]


def test_configured_plugin_load_failure_is_raised(monkeypatch):
    monkeypatch.setattr(base, "BUILTIN_PLUGINS", ())
    monkeypatch.setattr(base, "entry_points", lambda **kwargs: [])
    monkeypatch.setattr(base.PluginManager, "_instance", None)

    with pytest.raises(ModuleNotFoundError, match="missing_plugin"):
        prepare_plugins(DictDefault(plugins=["missing_plugin:Plugin"]))


def test_broken_plugin_entry_point_is_logged_and_skipped(monkeypatch):
    from unittest.mock import Mock

    point = EntryPoint(
        name="broken-plugin",
        value="missing_plugin:Plugin",
        group=base.PLUGIN_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(base, "BUILTIN_PLUGINS", ())
    monkeypatch.setattr(base, "entry_points", lambda **kwargs: [point])
    monkeypatch.setattr(base.PluginManager, "_instance", None)
    warning = Mock()
    monkeypatch.setattr(base.LOG, "warning", warning)

    manager = base.PluginManager.get_instance()

    assert not manager.plugins
    warning.assert_called_once()
    assert "Could not load plugin" in warning.call_args.args[0]
    assert warning.call_args.kwargs["exc_info"] is True
