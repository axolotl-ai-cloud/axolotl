# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The cutlass-dsl 4.8 ``nvvm.mbarrier_arrive`` shim for the quack vendored in sonic-moe."""

import importlib.metadata
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from axolotl.integrations.kernels import plugin


def _nvvm(fn):
    names = ("cutlass", "cutlass._mlir", "cutlass._mlir.dialects")
    modules = {name: ModuleType(name) for name in names}
    nvvm = ModuleType("cutlass._mlir.dialects.nvvm")
    nvvm.mbarrier_arrive = fn
    modules["cutlass._mlir.dialects"].nvvm = nvvm
    modules[nvvm.__name__] = nvvm
    return modules, nvvm


def _recorder_48():
    calls = []

    def mbarrier_arrive(
        addr,
        *,
        count=None,
        multicast_mask=None,
        scope=None,
        relaxed=None,
        results=None,
        loc=None,
        ip=None,
    ):
        calls.append(
            {
                "addr": addr,
                "count": count,
                "scope": scope,
                "relaxed": relaxed,
                "results": results,
                "loc": loc,
                "ip": ip,
            }
        )
        return "token"

    return mbarrier_arrive, calls


def test_48_binding_is_shimmed_and_forwards_both_call_forms():
    fn, calls = _recorder_48()
    modules, nvvm = _nvvm(fn)
    with patch.dict(sys.modules, modules):
        plugin._shim_nvvm_mbarrier_arrive()
        shimmed = nvvm.mbarrier_arrive
        assert shimmed is not fn and shimmed._axolotl_legacy_shim
        # quack 0.6.1: leading None result type, then the address
        assert (
            shimmed(None, "ptr", count=1, scope="cta", relaxed=True, loc="L", ip="I")
            == "token"
        )
        assert calls[-1] == {
            "addr": "ptr",
            "count": 1,
            "scope": "cta",
            "relaxed": True,
            "results": None,
            "loc": "L",
            "ip": "I",
        }
        shimmed("ptr", count=2)
        assert calls[-1]["addr"] == "ptr" and calls[-1]["results"] is None
        shimmed("i64", "ptr")
        assert calls[-1]["addr"] == "ptr" and calls[-1]["results"] == ["i64"]
        plugin._shim_nvvm_mbarrier_arrive()
        assert nvvm.mbarrier_arrive is shimmed


def test_46_binding_with_a_positional_result_is_left_alone():
    def mbarrier_arrive(res, addr, *, count=None, loc=None, ip=None):
        return res, addr

    modules, nvvm = _nvvm(mbarrier_arrive)
    with patch.dict(sys.modules, modules):
        plugin._shim_nvvm_mbarrier_arrive()
        assert nvvm.mbarrier_arrive is mbarrier_arrive


def test_missing_cutlass_is_a_no_op():
    with patch.dict(sys.modules, {"cutlass": None}):
        plugin._shim_nvvm_mbarrier_arrive()


def _metadata(monkeypatch, versions, quack_requires):
    def version(name):
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    def requires(name):
        if name == "quack-kernels" and name in versions:
            return quack_requires
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib.metadata, "requires", requires)


def test_loaded_vendored_quack_replaces_the_installed_pin(monkeypatch):
    _metadata(
        monkeypatch,
        {"nvidia-cutlass-dsl": "4.8.0", "quack-kernels": "0.6.1"},
        ["nvidia-cutlass-dsl==4.6.0"],
    )
    with pytest.raises(RuntimeError):
        plugin._check_sonicmoe_cutlass_dsl()
    vendored = ModuleType("_sonic_moe_cuda_89d4054.quack")
    vendored.__version__ = "0.6.1"  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {vendored.__name__: vendored}):
        assert plugin._loaded_vendored_quack() == {vendored.__name__: "0.6.1"}
        plugin._check_sonicmoe_cutlass_dsl()
