# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""No-GPU coverage for the load-bearing safety guards on the grouped NVFP4 MoE path.

Each guard turns silent corruption or a Blackwell SIGSEGV into a clear RuntimeError. They were
shipping untested: the GPU oracle (test_grouped_fp4_train.py) never enters them, because on the sm89
CI a fused backend is available, so the C1 cutlass-only raise and the A4/B2 fall-throughs do not
fire. Here each guard is driven directly with mocked arch/availability probes - no CUDA needed (the
scattermoe_lora package imports triton, so the module skips on a triton-less host).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")


def _mock_experts(num_experts):
    """Minimal stand-in for a scattermoe experts module. The guards under test fire before any
    weight/LoRA shape is read, so the proj params and LoRA tuples only need to be present + non-None.
    """
    lora = (torch.zeros(1), torch.zeros(1), 1.0)
    return SimpleNamespace(
        num_experts=num_experts,
        gate_up_proj=torch.zeros(1),
        down_proj=torch.zeros(1),
        _scattermoe_lora={"gate_up_proj": lora, "down_proj": lora},
    )


def test_c1_cutlass_nonunit_weight_scale_2_without_alt_backend_raises(monkeypatch):
    # C1: cutlass cannot fold a non-unit per_tensor_scale (weight_scale_2). With cutlass resolved,
    # a non-unit pt present, and neither marlin nor deepgemm available to fold it, the path must
    # hard-error rather than silently produce a wrong forward + grad mismatch.
    from axolotl.integrations.kernels.libs.scattermoe_lora import grouped_train as gt

    monkeypatch.setattr(gt, "_train_backend", lambda mode: "cutlass")
    monkeypatch.setattr(gt, "_has_nonunit_pt", lambda *nvs: True)
    monkeypatch.setattr(gt, "_backend_available", lambda name: False)

    E, twoI, packed_k, H, packed_i = 4, 16, 8, 8, 4
    gu = SimpleNamespace(qdata=torch.zeros(E, twoI, packed_k, dtype=torch.uint8))
    dn = SimpleNamespace(qdata=torch.zeros(E, H, packed_i, dtype=torch.uint8))
    hidden = torch.zeros(2, H)

    with pytest.raises(RuntimeError, match="weight_scale_2"):
        gt.grouped_fp4_moe_train(
            hidden, None, None, gu, dn, None, None, None, "nvfp4", mxfp4_cache={}
        )


def test_a4_nvfp4_lora_grouped_unavailable_raises(monkeypatch):
    # A4: NVFP4 experts + LoRA select the grouped path (dsv4_fp4_grouped_mode set), but no fused
    # grouped backend resolves -> must hard-error instead of falling through to the SIGSEGV MX kernel.
    from axolotl.integrations.kernels.libs.scattermoe_lora import (
        experts as ex,
        grouped_train as gt,
    )

    monkeypatch.setattr(ex, "is_nvfp4_param", lambda p: True)
    monkeypatch.setattr(gt, "grouped_fp4_available", lambda mode: False)
    monkeypatch.setattr(ex.RUNTIME, "fp4_grouped_mode", "nvfp4")

    E, H, topk, N = 4, 8, 2, 6
    self_ = _mock_experts(E)
    top_k_index = torch.randint(0, E, (N, topk))
    top_k_weights = torch.rand(N, topk)
    hidden = torch.randn(N, H)

    with pytest.raises(RuntimeError, match="scatter2scatter_lora_mx"):
        ex.scattermoe_experts_forward(self_, hidden, top_k_index, top_k_weights)


def test_b2_nvfp4_lora_without_grouped_mode_raises_on_blackwell(monkeypatch):
    # B2: NVFP4 experts + LoRA WITHOUT dsv4_fp4_grouped_mode would fall through to the legacy MX
    # kernel, which SIGSEGVs on Blackwell. On sm100/sm120 it must hard-error with guidance instead.
    from axolotl.integrations.kernels.libs.scattermoe_lora import experts as ex

    monkeypatch.setattr(ex, "is_nvfp4_param", lambda p: True)
    monkeypatch.setattr(ex.RUNTIME, "fp4_grouped_mode", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))

    E, H, topk, N = 4, 8, 2, 6
    self_ = _mock_experts(E)
    top_k_index = torch.randint(0, E, (N, topk))
    top_k_weights = torch.rand(N, topk)
    hidden = torch.randn(N, H)

    with pytest.raises(RuntimeError, match="grouped fp4 MoE path on Blackwell"):
        ex.scattermoe_experts_forward(self_, hidden, top_k_index, top_k_weights)


def test_b2_nvfp4_lora_without_grouped_mode_ok_off_blackwell(monkeypatch):
    # The B2 guard must NOT fire on non-Blackwell archs (the legacy MX path is left intact there);
    # it should fall through to _prepare_weights_and_lora rather than raising the B2 error.
    from axolotl.integrations.kernels.libs.scattermoe_lora import experts as ex

    monkeypatch.setattr(ex, "is_nvfp4_param", lambda p: True)
    monkeypatch.setattr(ex.RUNTIME, "fp4_grouped_mode", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))

    E, H, topk, N = 4, 8, 2, 6
    self_ = _mock_experts(E)
    top_k_index = torch.randint(0, E, (N, topk))
    top_k_weights = torch.rand(N, topk)
    hidden = torch.randn(N, H)

    # The B2 guard is skipped (sm89); execution proceeds past it. It will fail later in the real MX
    # path on CPU, but NOT with the B2 Blackwell message.
    with pytest.raises(Exception) as exc:
        ex.scattermoe_experts_forward(self_, hidden, top_k_index, top_k_weights)
    assert "grouped fp4 MoE path on Blackwell" not in str(exc.value)
