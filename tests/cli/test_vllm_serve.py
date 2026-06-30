"""pytest tests for axolotl CLI vllm_serve argument precedence."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from axolotl.cli.vllm_serve import do_vllm_serve


def _make_cfg(serve_module):
    """Build a minimal cfg whose vLLM booleans are enabled in the config."""
    return SimpleNamespace(
        base_model="dummy-model",
        vllm=SimpleNamespace(
            serve_module=serve_module,
            tensor_parallel_size=1,
            data_parallel_size=1,
            host="0.0.0.0",
            port=8000,
            gpu_memory_utilization=0.9,
            dtype="auto",
            max_model_len=2048,
            enable_prefix_caching=True,
            reasoning_parser="",
            enable_reasoning=True,
            enforce_eager=None,
            worker_extension_cls=None,
        ),
        trl=SimpleNamespace(vllm_lora_sync=False),
        lora_r=None,
    )


def test_cli_false_overrides_config_true_booleans():
    """Explicit CLI ``False`` must override config-enabled boolean options.

    Regression test for the ``cli or cfg`` precedence bug where a config that
    enabled ``enable_prefix_caching`` / ``enable_reasoning`` could not be
    disabled from the CLI (``False or True`` -> ``True``).
    """
    # Use a stub serve module so we hit the AxolotlScriptArguments path, which
    # carries both enable_prefix_caching (via base kwargs) and enable_reasoning.
    serve_module_name = "axolotl_test_stub_serve"
    stub = types.ModuleType(serve_module_name)
    stub.main = MagicMock()
    sys.modules[serve_module_name] = stub

    cfg = _make_cfg(serve_module=None)
    cli_args = {
        "serve_module": serve_module_name,
        "enable_prefix_caching": False,
        "enable_reasoning": False,
    }

    try:
        with patch("axolotl.cli.vllm_serve.load_cfg", return_value=cfg):
            do_vllm_serve("config.yml", cli_args)
    finally:
        del sys.modules[serve_module_name]

    stub.main.assert_called_once()
    script_args = stub.main.call_args.args[0]
    assert script_args.enable_prefix_caching is False
    assert script_args.enable_reasoning is False


def test_cli_none_falls_back_to_config_booleans():
    """When the CLI value is omitted, the config value is preserved."""
    serve_module_name = "axolotl_test_stub_serve_2"
    stub = types.ModuleType(serve_module_name)
    stub.main = MagicMock()
    sys.modules[serve_module_name] = stub

    cfg = _make_cfg(serve_module=None)
    cli_args = {
        "serve_module": serve_module_name,
        "enable_prefix_caching": None,
        "enable_reasoning": None,
    }

    try:
        with patch("axolotl.cli.vllm_serve.load_cfg", return_value=cfg):
            do_vllm_serve("config.yml", cli_args)
    finally:
        del sys.modules[serve_module_name]

    stub.main.assert_called_once()
    script_args = stub.main.call_args.args[0]
    assert script_args.enable_prefix_caching is True
    assert script_args.enable_reasoning is True
