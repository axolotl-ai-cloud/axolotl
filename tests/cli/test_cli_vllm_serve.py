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


@pytest.fixture
def native_serve(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "axolotl.cli.vllm_serve.os.execve",
        lambda exe, cmd, env: calls.append((exe, cmd, env)),
    )
    monkeypatch.setattr("axolotl.cli.vllm_serve._vllm_at_least", lambda _v: False)

    def configure(vllm=None, trl=None, **extra):
        cfg = DictDefault(
            {
                "base_model": "dummy-model",
                "revision_of_model": None,
                "trust_remote_code": False,
                "vllm": vllm or {},
                "trl": trl or {},
                **extra,
            }
        )
        monkeypatch.setattr("axolotl.cli.vllm_serve.load_cfg", lambda *_, **__: cfg)
        return calls

    return configure


def _run(cli_runner, tmp_path, flags=()):
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")
    result = cli_runner.invoke(cli, ["vllm-serve", str(config), *flags])
    assert result.exit_code == 0, result.output


def test_native_default_command(cli_runner, tmp_path, native_serve):
    calls = native_serve(vllm={"host": "127.0.0.1", "port": 8123})
    _run(cli_runner, tmp_path)
    _, cmd, env = calls[0]
    assert cmd[1:5] == ["-m", "vllm.entrypoints.cli.main", "serve", "dummy-model"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert cmd[cmd.index("--weight-transfer-config") + 1] == '{"backend": "nccl"}'
    assert "--enable-lora" not in cmd
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in env


def test_native_lora_sync_enables_lora(cli_runner, tmp_path, native_serve, monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    calls = native_serve(trl={"vllm_lora_sync": True}, lora_r=32)
    _run(cli_runner, tmp_path)
    _, cmd, env = calls[0]
    assert "--enable-lora" in cmd
    assert cmd[cmd.index("--max-lora-rank") + 1] == "32"
    assert cmd[cmd.index("--api-server-count") + 1] == "1"
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"


def test_native_lora_sync_off(cli_runner, tmp_path, native_serve, monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    calls = native_serve(trl={"vllm_lora_sync": False}, lora_r=32)
    _run(cli_runner, tmp_path)
    _, cmd, env = calls[0]
    assert "--enable-lora" not in cmd
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in env


def test_legacy_serve_module_routes_to_native(cli_runner, tmp_path, native_serve):
    calls = native_serve(vllm={"serve_module": "axolotl.scripts.vllm_serve_lora"})
    _run(cli_runner, tmp_path)
    assert len(calls) == 1
    assert calls[0][1][3] == "serve"


def test_native_reasoning_parser_forwarded(cli_runner, tmp_path, native_serve):
    calls = native_serve(vllm={"reasoning_parser": "qwen3", "enable_reasoning": True})
    _run(cli_runner, tmp_path)
    cmd = calls[0][1]
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"


def test_native_no_prefix_caching_flag(cli_runner, tmp_path, native_serve):
    calls = native_serve(vllm={"enable_prefix_caching": True})
    _run(cli_runner, tmp_path, ["--no-enable-prefix-caching"])
    cmd = calls[0][1]
    assert "--no-enable-prefix-caching" in cmd
    assert "--enable-prefix-caching" not in cmd


@pytest.mark.parametrize(
    "lora_r,expected", [(4, "8"), (12, "16"), (96, "128"), (64, "64")]
)
def test_native_lora_rank_rounded_up(
    cli_runner, tmp_path, native_serve, monkeypatch, lora_r, expected
):
    calls = native_serve(trl={"vllm_lora_sync": True}, lora_r=lora_r)
    _run(cli_runner, tmp_path)
    cmd = calls[0][1]
    assert cmd[cmd.index("--max-lora-rank") + 1] == expected


def test_lora_rank_above_vllm_limit_rejected():
    from axolotl.cli.vllm_serve import round_max_lora_rank

    with pytest.raises(ValueError):
        round_max_lora_rank(513)


def test_reasoning_parser_without_enable_flag(cli_runner, tmp_path, native_serve):
    calls = native_serve(vllm={"reasoning_parser": "deepseek_r1"})
    _run(cli_runner, tmp_path)
    cmd = calls[0][1]
    assert cmd[cmd.index("--reasoning-parser") + 1] == "deepseek_r1"
