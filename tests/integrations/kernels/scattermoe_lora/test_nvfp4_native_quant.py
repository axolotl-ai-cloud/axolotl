# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Byte-parity gate: native NVFP4 to_nvfp4 must be 0-mismatch vs torchao across all producer profiles + fallbacks."""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor

import axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_native_quant as nq  # noqa: E402


def _blackwell_device() -> int | None:
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major == 12:
            return i
    # Also accept sm100 datacenter Blackwell.
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major == 10:
            return i
    return None


_BW = _blackwell_device()
requires_blackwell = pytest.mark.skipif(
    _BW is None, reason="Blackwell (cc major 10/12) GPU required"
)
DEV = f"cuda:{_BW}" if _BW is not None else "cuda:0"


def _torchao_ge_018() -> bool:
    """torchao >= 0.18. The byte-parity oracle below is torchao's *reference*
    nvfp4 quant, which only matches the production (reciprocal-multiply / MSLK)
    path the native kernel targets from 0.18 on. On 0.17 the reference (a) does
    bf16 single-level via true-division — ≤1 ULP off the kernel on grid
    boundaries — and (b) raises on 3D [E,N,K] per-expert input (fixed in 0.18).
    The native kernel itself is version-independent; only the *oracle* moved."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import parse

    try:
        return parse(version("torchao")) >= parse("0.18.0")
    except PackageNotFoundError:
        return False


# Versioned gate: the kernel-vs-torchao byte-parity oracle is only valid on
# torchao>=0.18. axolotl currently pins 0.17, so these skip until that bump (and
# they are Blackwell-gated, so CI never runs them regardless); they auto-activate
# at 0.18 with no further changes. The fallback/idempotent tests below are
# version-robust (same torchao on both sides) and stay ungated.
requires_torchao_018 = pytest.mark.skipif(
    not _torchao_ge_018(),
    reason="kernel-vs-torchao byte-parity oracle requires torchao>=0.18 "
    "(0.17 reference uses true-division for bf16 and raises on 3D per-expert)",
)


@pytest.fixture(autouse=True)
def _restore_patch():
    """Ensure every test starts/ends with torchao's original to_nvfp4."""
    nq.uninstall_native_nvfp4()
    yield
    nq.uninstall_native_nvfp4()


def _orig_to_nvfp4(data, **kw):
    """Call the genuine torchao implementation (patch guaranteed uninstalled)."""
    assert nq._ORIG_TO_NVFP4 is None, "patch leaked into reference call"
    return NVFP4Tensor.to_nvfp4(data, **kw)


def _native_to_nvfp4(data, **kw):
    installed = nq.install_native_nvfp4()
    assert installed
    try:
        return NVFP4Tensor.to_nvfp4(data, **kw)
    finally:
        nq.uninstall_native_nvfp4()


_MismatchValue = int | tuple[tuple[int, ...], tuple[int, ...]]


def _mismatch_counts(native: Any, ref: Any) -> dict[str, _MismatchValue]:
    """Return dict of byte-mismatch counts for qdata, scale, per_tensor_scale."""
    out: dict[str, _MismatchValue] = {}

    q_n = native.qdata.view(torch.uint8)
    q_r = ref.qdata.view(torch.uint8)
    out["qdata_shape"] = (tuple(q_n.shape), tuple(q_r.shape))
    out["qdata"] = int((q_n != q_r).sum().item()) if q_n.shape == q_r.shape else -1

    s_n = native.scale.view(torch.uint8)
    s_r = ref.scale.view(torch.uint8)
    out["scale_shape"] = (tuple(s_n.shape), tuple(s_r.shape))
    out["scale"] = int((s_n != s_r).sum().item()) if s_n.shape == s_r.shape else -1

    pn, pr = native.per_tensor_scale, ref.per_tensor_scale
    if pn is None and pr is None:
        out["per_tensor_scale"] = 0
    elif pn is None or pr is None:
        out["per_tensor_scale"] = -1
    else:
        out["per_tensor_scale"] = int((pn != pr).sum().item())
    return out


def _assert_byte_identical(native, ref, label):
    mm = _mismatch_counts(native, ref)
    assert mm["qdata"] == 0, f"[{label}] qdata mismatch: {mm}"
    assert mm["scale"] == 0, f"[{label}] scale mismatch: {mm}"
    assert mm["per_tensor_scale"] == 0, f"[{label}] per_tensor_scale mismatch: {mm}"
    # Constructor parity beyond raw bytes:
    assert native.is_swizzled_scales == ref.is_swizzled_scales, label
    assert native.block_size == ref.block_size, label
    assert native.orig_dtype == ref.orig_dtype, label
    assert tuple(native.shape) == tuple(ref.shape), label
    return mm


_SHAPES_2D = [(512, 512), (256, 2048)]
_SHAPE_3D = (8, 256, 512)
_DTYPES = [torch.float32, torch.bfloat16]


def _mk(shape, dtype, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(*shape, device=DEV, dtype=dtype, generator=g).contiguous()


@requires_torchao_018
@requires_blackwell
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES_2D + [_SHAPE_3D])
def test_single_level_non_swizzled(shape, dtype):
    """Profile (a): MoE single-level, non-swizzled."""
    x = _mk(shape, dtype)
    ref = _orig_to_nvfp4(x, block_size=16, is_swizzled_scales=False)
    nat = _native_to_nvfp4(x, block_size=16, is_swizzled_scales=False)
    _assert_byte_identical(nat, ref, f"single/non-swizzled {shape} {dtype}")


@requires_torchao_018
@requires_blackwell
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES_2D + [_SHAPE_3D])
def test_two_level_swizzled_scalar(shape, dtype):
    """Profile (b): PTQ two-level swizzled with a scalar per_tensor_scale."""
    x = _mk(shape, dtype)
    pts = torch.tensor(0.37, device=DEV, dtype=torch.float32)  # scalar, ndim 0
    ref = _orig_to_nvfp4(
        x, block_size=16, per_tensor_scale=pts, is_swizzled_scales=True
    )
    nat = _native_to_nvfp4(
        x, block_size=16, per_tensor_scale=pts, is_swizzled_scales=True
    )
    _assert_byte_identical(nat, ref, f"two/swizzled-scalar {shape} {dtype}")


@requires_torchao_018
@requires_blackwell
@pytest.mark.parametrize("dtype", _DTYPES)
def test_two_level_per_expert(dtype):
    """Profile (c): two-level with per-expert [E,1,1] per_tensor_scale (3D)."""
    E, N, K = _SHAPE_3D
    x = _mk((E, N, K), dtype)
    pts = torch.rand(E, 1, 1, device=DEV, dtype=torch.float32) + 0.1
    # 3D per-expert scale: torchao path is non-swizzled here (MoE convert).
    ref = _orig_to_nvfp4(
        x, block_size=16, per_tensor_scale=pts, is_swizzled_scales=False
    )
    nat = _native_to_nvfp4(
        x, block_size=16, per_tensor_scale=pts, is_swizzled_scales=False
    )
    _assert_byte_identical(nat, ref, f"two/per-expert {dtype}")


@requires_torchao_018
@requires_blackwell
@pytest.mark.parametrize("dtype", _DTYPES)
def test_two_level_per_expert_swizzled(dtype):
    """Profile (c'): per-expert [E,1,1] per_tensor_scale, swizzled."""
    E, N, K = _SHAPE_3D
    x = _mk((E, N, K), dtype, seed=7)
    pts = torch.rand(E, 1, 1, device=DEV, dtype=torch.float32) + 0.1
    ref = _orig_to_nvfp4(
        x, block_size=16, per_tensor_scale=pts, is_swizzled_scales=True
    )
    nat = _native_to_nvfp4(
        x, block_size=16, per_tensor_scale=pts, is_swizzled_scales=True
    )
    _assert_byte_identical(nat, ref, f"two/per-expert-swizzled {dtype}")


@requires_blackwell
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fallback_non_power_of_2_K(dtype):
    """K=192 (mult of 16 but not power-of-two) must fall back to torchao, byte-identical."""
    x = _mk((256, 192), dtype)
    assert not nq._native_k_supported(192)
    ref = _orig_to_nvfp4(x, block_size=16, is_swizzled_scales=True)
    nat = _native_to_nvfp4(x, block_size=16, is_swizzled_scales=True)
    _assert_byte_identical(nat, ref, f"fallback K=192 {dtype}")


@pytest.mark.parametrize("dtype", _DTYPES)
def test_fallback_forced_non_blackwell(monkeypatch, dtype):
    """Gate forced False -> patched to_nvfp4 delegates to torchao, identical tensor (any CUDA GPU)."""
    monkeypatch.setattr(
        nq, "is_blackwell_native_nvfp4_available", lambda device=None: False
    )
    x = torch.randn(256, 512, device="cuda:0", dtype=dtype).contiguous()
    ref = _orig_to_nvfp4(x, block_size=16, is_swizzled_scales=True)
    nat = _native_to_nvfp4(x, block_size=16, is_swizzled_scales=True)
    _assert_byte_identical(nat, ref, f"forced-non-blackwell {dtype}")


def test_install_uninstall_idempotent_and_restoring():
    """install/uninstall round-trips restore the exact original function."""
    orig = NVFP4Tensor.to_nvfp4
    assert nq.install_native_nvfp4()
    assert nq.install_native_nvfp4()  # idempotent
    assert NVFP4Tensor.to_nvfp4 is not orig
    assert nq.uninstall_native_nvfp4()
    assert NVFP4Tensor.to_nvfp4 is orig
    assert not nq.uninstall_native_nvfp4()  # already uninstalled


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
