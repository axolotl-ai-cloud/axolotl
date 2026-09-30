"""Tests for axolotl vllm-serve: config/CLI precedence and the `vllm serve` command."""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from axolotl.cli.main import cli
from axolotl.cli.vllm_serve import (
    VllmServeArgs,
    build_vllm_serve_command,
    build_vllm_serve_env,
    round_up_lora_rank,
)
from axolotl.utils.dict import DictDefault


def _cfg(**overrides):
    cfg = {
        "base_model": "dummy-model",
        "revision_of_model": "test-revision",
        "trust_remote_code": True,
        "vllm": {"enable_prefix_caching": True},
    }
    cfg.update(overrides)
    return DictDefault(cfg)


@pytest.fixture
def stub_serve(monkeypatch):
    """Stub a custom serve module (no real vLLM server)."""
    name = "axolotl_stub_serve"
    module = types.ModuleType(name)
    module.main = MagicMock()
    monkeypatch.setitem(sys.modules, name, module)

    cfg = _cfg()
    cfg.vllm.serve_module = name
    monkeypatch.setattr("axolotl.cli.vllm_serve.load_cfg", lambda *_, **__: cfg)
    return module


@pytest.fixture
def exec_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "axolotl.cli.vllm_serve.os.execve",
        lambda path, argv, env: calls.append((path, argv, env)),
    )
    return calls


def _flag_value(command, flag):
    return command[command.index(flag) + 1]


@pytest.mark.parametrize(
    "flags, expected",
    [
        (["--no-enable-prefix-caching"], False),
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
    args = stub_serve.main.call_args.args[0]
    assert args.enable_prefix_caching is expected


def test_vllm_serve_forwards_model_revision_and_remote_code(
    cli_runner, tmp_path, stub_serve
):
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")

    result = cli_runner.invoke(cli, ["vllm-serve", str(config)])
    assert result.exit_code == 0, result.output

    args = stub_serve.main.call_args.args[0]
    assert args.revision == "test-revision"
    assert args.trust_remote_code is True


def test_vllm_serve_execs_native_server(cli_runner, tmp_path, monkeypatch, exec_calls):
    cfg = _cfg()
    monkeypatch.setattr("axolotl.cli.vllm_serve.load_cfg", lambda *_, **__: cfg)
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")

    result = cli_runner.invoke(cli, ["vllm-serve", str(config)])
    assert result.exit_code == 0, result.output

    (path, argv, env), *_ = exec_calls
    assert path == sys.executable
    assert argv[1:5] == ["-m", "vllm.entrypoints.cli.main", "serve", "dummy-model"]
    assert _flag_value(argv, "--revision") == "test-revision"
    assert "--trust-remote-code" in argv
    assert "--enable-lora" not in argv
    assert env["VLLM_SERVER_DEV_MODE"] == "1"


def test_removed_serve_module_falls_back_to_native_server(
    cli_runner, tmp_path, monkeypatch, exec_calls
):
    cfg = _cfg()
    cfg.vllm.serve_module = "axolotl.scripts.vllm_serve_lora"
    monkeypatch.setattr("axolotl.cli.vllm_serve.load_cfg", lambda *_, **__: cfg)
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")

    result = cli_runner.invoke(cli, ["vllm-serve", str(config)])
    assert result.exit_code == 0, result.output
    assert len(exec_calls) == 1


def test_lora_sync_enables_runtime_lora_loading(
    cli_runner, tmp_path, monkeypatch, exec_calls
):
    cfg = _cfg(lora_r=24, trl={"vllm_lora_sync": True})
    monkeypatch.setattr("axolotl.cli.vllm_serve.load_cfg", lambda *_, **__: cfg)
    config = tmp_path / "config.yml"
    config.write_text("base_model: dummy-model\n")

    result = cli_runner.invoke(cli, ["vllm-serve", str(config)])
    assert result.exit_code == 0, result.output

    (_, argv, env), *_ = exec_calls
    assert "--enable-lora" in argv
    assert _flag_value(argv, "--max-lora-rank") == "32"
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "1"


def test_command_always_enables_weight_transfer():
    command = build_vllm_serve_command(VllmServeArgs(model="m"))
    assert json.loads(_flag_value(command, "--weight-transfer-config")) == {
        "backend": "nccl"
    }
    assert _flag_value(command, "--logprobs-mode") == "processed_logprobs"
    assert _flag_value(command, "--max-logprobs") == "-1"


def test_command_optional_flags():
    args = VllmServeArgs(
        model="m",
        max_model_len=2048,
        enable_prefix_caching=False,
        enforce_eager=True,
        reasoning_parser="qwen3",
        worker_extension_cls="my.Ext",
        extra_args=["--seed", "1"],
    )
    command = build_vllm_serve_command(args)
    assert _flag_value(command, "--max-model-len") == "2048"
    assert "--no-enable-prefix-caching" in command
    assert "--enforce-eager" in command
    assert _flag_value(command, "--reasoning-parser") == "qwen3"
    assert _flag_value(command, "--worker-extension-cls") == "my.Ext"
    assert command[-2:] == ["--seed", "1"]
    assert "--revision" not in command


def test_env_without_lora_leaves_runtime_lora_unset(monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    env = build_vllm_serve_env(VllmServeArgs(model="m"))
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in env
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


@pytest.mark.parametrize(
    "rank, expected", [(1, 1), (4, 8), (16, 16), (17, 32), (300, 320)]
)
def test_round_up_lora_rank(rank, expected):
    assert round_up_lora_rank(rank) == expected


def test_round_up_lora_rank_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds"):
        round_up_lora_rank(1024)
