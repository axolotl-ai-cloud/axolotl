"""Tests for axolotl vllm_serve CLI/config boolean precedence."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from axolotl.cli.vllm_serve import do_vllm_serve


def _run_serve(module_name, cli_bools):
    """Run do_vllm_serve against a stub serve module with config booleans on."""
    stub = types.ModuleType(module_name)
    stub.main = MagicMock()
    sys.modules[module_name] = stub
    cfg = SimpleNamespace(
        base_model="dummy-model",
        vllm=SimpleNamespace(
            serve_module=None,
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
    try:
        with patch("axolotl.cli.vllm_serve.load_cfg", return_value=cfg):
            do_vllm_serve("config.yml", {"serve_module": module_name, **cli_bools})
    finally:
        del sys.modules[module_name]
    stub.main.assert_called_once()
    return stub.main.call_args.args[0]


@pytest.mark.parametrize("cli_value, expected", [(False, False), (None, True)])
def test_vllm_serve_bool_precedence(cli_value, expected):
    """CLI False overrides config True; omitted CLI (None) keeps the config value.

    Regression test for the ``cli or cfg`` bug where ``False or True`` -> ``True``
    made a config-enabled option impossible to disable from the CLI.
    """
    args = _run_serve(
        f"axolotl_stub_serve_{expected}",
        {"enable_prefix_caching": cli_value, "enable_reasoning": cli_value},
    )
    assert args.enable_prefix_caching is expected
    assert args.enable_reasoning is expected
