"""Tests for axolotl vLLM serve argument construction."""

from dataclasses import dataclass
import sys
import types
from types import SimpleNamespace


def test_vllm_serve_cli_false_overrides_true_config(monkeypatch, tmp_path):
    captured = {}
    serve_module = types.ModuleType("test_vllm_serve_stub")

    def main(script_args):
        captured["args"] = script_args

    serve_module.main = main
    monkeypatch.setitem(sys.modules, "test_vllm_serve_stub", serve_module)
    trl_module = types.ModuleType("trl")
    trl_scripts_module = types.ModuleType("trl.scripts")
    trl_vllm_module = types.ModuleType("trl.scripts.vllm_serve")

    @dataclass
    class ScriptArguments:
        model: str
        tensor_parallel_size: int = 1
        data_parallel_size: int = 1
        host: str | None = None
        port: int | None = None
        gpu_memory_utilization: float | None = None
        dtype: str | None = None
        max_model_len: int | None = None
        enable_prefix_caching: bool | None = None
        enforce_eager: bool = False

    trl_vllm_module.ScriptArguments = ScriptArguments
    monkeypatch.setitem(sys.modules, "trl", trl_module)
    monkeypatch.setitem(sys.modules, "trl.scripts", trl_scripts_module)
    monkeypatch.setitem(sys.modules, "trl.scripts.vllm_serve", trl_vllm_module)

    cfg = SimpleNamespace(
        base_model="test-model",
        vllm=SimpleNamespace(
            serve_module="test_vllm_serve_stub",
            tensor_parallel_size=None,
            data_parallel_size=None,
            host="127.0.0.1",
            port=8000,
            gpu_memory_utilization=0.9,
            dtype="auto",
            max_model_len=None,
            enable_prefix_caching=True,
            enable_reasoning=True,
            reasoning_parser="deepseek_r1",
            enforce_eager=True,
        ),
    )
    config_module = types.ModuleType("axolotl.cli.config")
    config_module.load_cfg = lambda _: cfg
    monkeypatch.setitem(sys.modules, "axolotl.cli.config", config_module)

    from axolotl.cli.vllm_serve import do_vllm_serve

    do_vllm_serve(
        tmp_path / "config.yml",
        {
            "enable_prefix_caching": False,
            "enable_reasoning": False,
            "enforce_eager": False,
        },
    )

    assert captured["args"].enable_prefix_caching is False
    assert captured["args"].enable_reasoning is False
    assert captured["args"].enforce_eager is False
