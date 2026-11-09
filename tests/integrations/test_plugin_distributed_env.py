"""Plugin env vars must be in os.environ before the distributed state is created."""

import collections
import logging
import os
from pathlib import Path

import pytest

from axolotl.cli import train as train_cli
from axolotl.integrations import base as plugin_base
from axolotl.integrations.base import BasePlugin, PluginManager
from axolotl.utils import trainer as trainer_module
from axolotl.utils.dict import DictDefault

ENV_KEY = "X_TEST_ENV"


class EnvPlugin(BasePlugin):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def get_distributed_env(self, cfg):
        return dict(self.env)


@pytest.fixture
def plugins(monkeypatch):
    """Isolated plugin registry and process environment."""
    manager = PluginManager.get_instance()
    monkeypatch.setattr(manager, "plugins", collections.OrderedDict())
    monkeypatch.setattr(plugin_base, "get_builtin_plugins", lambda: ())
    saved_env = dict(os.environ)
    monkeypatch.delenv(ENV_KEY, raising=False)
    yield manager.plugins
    os.environ.clear()
    os.environ.update(saved_env)


def _cfg(**kwargs):
    return DictDefault(plugins=["env"], **kwargs)


def test_env_is_set_before_init_distributed_state(plugins, monkeypatch):
    plugins["env"] = EnvPlugin({ENV_KEY: "1"})
    monkeypatch.setenv("AXOLOTL_IS_PREPROCESS", "1")
    monkeypatch.setattr(
        "transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig",
        lambda *a, **k: None,
    )
    seen = []

    def fake_init_distributed_state():
        seen.append(os.environ.get(ENV_KEY))

    monkeypatch.setattr(
        trainer_module, "init_distributed_state", fake_init_distributed_state
    )
    cfg = _cfg(deepspeed=DictDefault(zero_optimization={"stage": 2}))
    try:
        trainer_module.prepare_optim_env(cfg)
    finally:
        if isinstance(cfg.deepspeed, str):
            Path(cfg.deepspeed).unlink(missing_ok=True)

    assert seen == ["1"]


def test_user_value_is_not_overwritten(plugins, monkeypatch):
    plugins["env"] = EnvPlugin({ENV_KEY: "1"})
    monkeypatch.setenv(ENV_KEY, "user")

    trainer_module.prepare_optim_env(_cfg())

    assert os.environ[ENV_KEY] == "user"


def test_conflicting_values_raise(plugins):
    plugins["a"] = EnvPlugin({ENV_KEY: "1"})
    plugins["b"] = EnvPlugin({ENV_KEY: "2"})

    with pytest.raises(ValueError, match=rf"'a' and 'b'.*{ENV_KEY}"):
        PluginManager.get_instance().get_distributed_env(_cfg())


def test_identical_values_merge(plugins):
    plugins["a"] = EnvPlugin({ENV_KEY: "1"})
    plugins["b"] = EnvPlugin({ENV_KEY: "1", "OTHER": "x"})

    merged = PluginManager.get_instance().get_distributed_env(_cfg())

    assert merged == {ENV_KEY: "1", "OTHER": "x"}


def test_warns_when_process_group_already_exists(plugins, monkeypatch, caplog):
    plugins["env"] = EnvPlugin({ENV_KEY: "1"})
    monkeypatch.setattr(
        trainer_module.torch.distributed, "is_initialized", lambda: True
    )

    with caplog.at_level(logging.WARNING, logger=trainer_module.__name__):
        trainer_module.apply_plugin_distributed_env(_cfg())

    assert os.environ[ENV_KEY] == "1"
    assert any(
        ENV_KEY in record.getMessage() and "may not take effect" in record.getMessage()
        for record in caplog.records
    )


def test_no_warning_when_env_already_forwarded(plugins, monkeypatch, caplog):
    plugins["env"] = EnvPlugin({ENV_KEY: "1"})
    monkeypatch.setenv(ENV_KEY, "1")
    monkeypatch.setattr(
        trainer_module.torch.distributed, "is_initialized", lambda: True
    )

    with caplog.at_level(logging.WARNING, logger=trainer_module.__name__):
        trainer_module.apply_plugin_distributed_env(_cfg())

    assert not any("may not take effect" in r.getMessage() for r in caplog.records)


def test_ray_runtime_env_forwards_plugin_env(plugins):
    plugins["env"] = EnvPlugin({ENV_KEY: "1"})

    assert train_cli.build_ray_runtime_env(_cfg()) == {"env_vars": {ENV_KEY: "1"}}


def test_ray_runtime_env_is_none_without_plugin_env(plugins):
    plugins["env"] = EnvPlugin({})

    assert train_cli.build_ray_runtime_env(_cfg()) is None
