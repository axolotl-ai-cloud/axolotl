# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The NVFP4 fp8-read path is gated to sm_120 (workstation Blackwell, the validated win).

fp8-read wins only in the weight-bandwidth-bound regime; on datacenter Blackwell (sm_100 /
sm_103, HBM, compute-bound) it regresses, so the path must auto-fall-back to bf16-read there.
`_fp8_read_arch_ok` is the gate (added to the fp8 branch of `_prepare_weights_and_lora`). This
test mocks the device capability, so it needs no GPU.
"""

from unittest import mock

import pytest

ex = pytest.importorskip(
    "axolotl.integrations.kernels.libs.scattermoe_lora.experts",
    reason="triton/torchao required to import the experts module",
)


def test_arch_gate_requires_cuda():
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert ex._fp8_read_arch_ok() is False


def test_arch_gate_enables_only_sm120():
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)):
            assert ex._fp8_read_arch_ok() is True  # sm_120 workstation Blackwell
        # datacenter Blackwell + every other arch must fall back to bf16-read
        for cap in [(10, 0), (10, 3), (9, 0), (12, 1), (8, 6), (8, 0)]:
            with mock.patch("torch.cuda.get_device_capability", return_value=cap):
                assert ex._fp8_read_arch_ok() is False, f"sm_{cap[0]}{cap[1]} should disable fp8-read"
