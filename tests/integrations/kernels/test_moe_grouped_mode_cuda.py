"""``moe_grouped_mode`` on a real GPU: each mode's backend resolves, and auto on a block-FP8
checkpoint picks fp8_block with that backend."""

import logging

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def default_runtime():
    from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import RUNTIME

    RUNTIME.reset()
    yield
    RUNTIME.reset()


@pytest.mark.parametrize(
    "mode,backends",
    [
        ("fp8_block", {"deepgemm_fp8", "blockfp8_triton"}),
        ("mxfp8", {"mxfp8_mma", "mxfp8_triton"}),
    ],
)
def test_backend_resolves(mode, backends):
    from axolotl.integrations.kernels.grouped_mode import grouped_backend

    assert grouped_backend(mode) in backends


def test_nvfp4_backend_matches_availability():
    from axolotl.integrations.kernels.grouped_mode import grouped_backend
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
    )

    backend = grouped_backend("nvfp4")
    assert (backend is not None) == grouped_fp4_available("nvfp4")
    assert backend in (None, "marlin", "cutlass", "deepgemm")


def test_auto_on_block_fp8_checkpoint(tmp_path, caplog):
    from axolotl.integrations.kernels.grouped_mode import configure_moe_grouped_mode

    from tests.integrations.kernels_cpu.test_moe_grouped_mode import _checkpoint

    cfg = _checkpoint(tmp_path, ["block_fp8", "block_fp8"])
    with caplog.at_level(logging.INFO, logger="axolotl"):
        choice = configure_moe_grouped_mode(cfg)
    assert choice.mode == "fp8_block" and not choice.explicit
    lines = [
        r.getMessage() for r in caplog.records if "moe_grouped_mode=" in r.getMessage()
    ]
    gpu = "sm%d%d" % torch.cuda.get_device_capability()
    assert len(lines) == 1 and f"gpu={gpu} | backend=" in lines[0]
    assert "backend=-" not in lines[0]
