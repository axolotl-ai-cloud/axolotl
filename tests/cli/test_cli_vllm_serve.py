"""Tests for axolotl vllm-serve CLI/config boolean precedence."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from axolotl.cli.main import cli
from axolotl.utils.dict import DictDefault


@pytest.fixture
def stub_serve(monkeypatch):
    """Stub the serve module (no real vLLM server) and route load_cfg to a
    config with the vLLM booleans enabled."""
    name = "axolotl_stub_serve"
    module = types.ModuleType(name)
    module.main = MagicMock()
    monkeypatch.setitem(sys.modules, name, module)

    cfg = DictDefault(
        {
            "base_model": "dummy-model",
            "revision_of_model": "test-revision",
            "trust_remote_code": True,
            "vllm": {
                "serve_module": name,
                "enable_prefix_caching": True,
                "enable_reasoning": True,
            },
        }
    )
    monkeypatch.setattr("axolotl.cli.vllm_serve.load_cfg", lambda *_, **__: cfg)
    return module


@pytest.mark.parametrize(
    "flags, expected",
    [
        (["--no-enable-prefix-caching", "--no-enable-reasoning"], False),
        ([], True),
    ],
)
def test_vllm_serve_bool_precedence(cli_runner, tmp_path, stub_serve, flags, expected):
    """`--no-<flag>` overrides a config-enabled option; omitting it keeps the config value.

    Drives the real CLI so Click's ``--flag/--no-flag`` parsing and
    ``filter_none_kwargs`` run: an omitted flag reaches ``do_vllm_serve`` as
    ``None`` (config wins), ``--no-<flag>`` as ``False`` (overrides). Regression
    for the ``cli or cfg`` bug where ``False or True`` -> ``True``.
    """
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")

    result = cli_runner.invoke(cli, ["vllm-serve", str(config), *flags])
    assert result.exit_code == 0, result.output

    stub_serve.main.assert_called_once()
    script_args = stub_serve.main.call_args.args[0]
    assert script_args.enable_prefix_caching is expected
    assert script_args.enable_reasoning is expected


def test_vllm_serve_forwards_model_revision(cli_runner, tmp_path, stub_serve):
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")

    result = cli_runner.invoke(cli, ["vllm-serve", str(config)])
    assert result.exit_code == 0, result.output

    stub_serve.main.assert_called_once()
    script_args = stub_serve.main.call_args.args[0]
    assert script_args.revision == "test-revision"


def test_vllm_serve_forwards_trust_remote_code(cli_runner, tmp_path, stub_serve):
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")

    result = cli_runner.invoke(cli, ["vllm-serve", str(config)])
    assert result.exit_code == 0, result.output

    stub_serve.main.assert_called_once()
    script_args = stub_serve.main.call_args.args[0]
    assert script_args.trust_remote_code is True
