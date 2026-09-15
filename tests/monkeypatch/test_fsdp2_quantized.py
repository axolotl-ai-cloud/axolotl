"""CPU-only tests for the FSDP2 quantized capability helpers (#2)."""

import pytest
import torch
import torch.nn as nn

from axolotl.monkeypatch.accelerate import fsdp2_quantized as fq


def test_model_has_nonfloat_params():
    float_only = nn.Linear(4, 4)
    assert not fq.model_has_nonfloat_params(float_only)

    class Quant(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(
                torch.zeros(4, 4, dtype=torch.uint8), requires_grad=False
            )

    assert fq.model_has_nonfloat_params(Quant())


def test_nonfloat_param_guard_restores_on_success():
    orig = nn.Parameter.__new__
    model = nn.Linear(2, 2)
    with fq.nonfloat_param_guard(model):
        assert nn.Parameter.__new__ is not orig  # patched inside
    assert nn.Parameter.__new__ is orig  # restored after


def test_nonfloat_param_guard_restores_on_exception():
    orig = nn.Parameter.__new__
    model = nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="boom"):
        with fq.nonfloat_param_guard(model):
            assert nn.Parameter.__new__ is not orig
            raise RuntimeError("boom during fully_shard")
    # the process-global patch must be restored even though the body raised
    assert nn.Parameter.__new__ is orig


def test_nonfloat_param_guard_defaults_new_nonfloat_to_no_grad():
    # torch normally forbids constructing a non-float Parameter with the default requires_grad=True;
    # inside the guard the default flips to False for non-float data, so it succeeds.
    with pytest.raises(RuntimeError):
        nn.Parameter(torch.zeros(2, dtype=torch.uint8))  # default True -> torch rejects

    model = nn.Linear(2, 2)
    with fq.nonfloat_param_guard(model):
        p = nn.Parameter(
            torch.zeros(2, dtype=torch.uint8)
        )  # default True -> guard makes it False
        assert p.requires_grad is False
        assert nn.Parameter(torch.zeros(2)).requires_grad is True  # float keeps True

    # after restore, the normal torch behavior returns
    with pytest.raises(RuntimeError):
        nn.Parameter(torch.zeros(2, dtype=torch.uint8))


def test_nonfloat_param_guard_freezes_existing_nonfloat():
    class Quant(nn.Module):
        def __init__(self):
            super().__init__()
            # non-float params must be created frozen (torch forbids requires_grad=True here)
            self.q = nn.Parameter(
                torch.zeros(2, 2, dtype=torch.uint8), requires_grad=False
            )
            self.f = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

    m = Quant()
    with fq.nonfloat_param_guard(m):
        assert m.q.requires_grad is False  # non-float stays frozen
        assert m.f.requires_grad is True  # float untouched


def test_register_fp32_shard_classes():
    saved = set(fq._FP32_SHARD_CLASS_NAMES)
    try:
        fq.register_fp32_shard_classes(["FooBarModule"])
        assert "FooBarModule" in fq._FP32_SHARD_CLASS_NAMES
    finally:  # restore the global registry so tests stay order-independent
        fq._FP32_SHARD_CLASS_NAMES.clear()
        fq._FP32_SHARD_CLASS_NAMES.update(saved)


def test_quantized_param_detection_float_logical_subclass():
    # torchao NVFP4Tensor/Float8Tensor report a logical FLOAT dtype, so the nonfloat check misses
    # them; the quantized check must still catch them by tensor-subclass name.
    saved = set(fq._QUANT_TENSOR_CLASS_NAMES)
    try:

        class FakeNVFP4Tensor(torch.Tensor):
            pass

        t = torch.zeros(4, 4, dtype=torch.bfloat16).as_subclass(FakeNVFP4Tensor)
        assert torch.is_floating_point(
            t
        )  # float-logical -> invisible to the nonfloat check

        fq.register_quantized_tensor_classes(["FakeNVFP4Tensor"])
        assert fq._is_quantized_param(t)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # torchao wraps the subclass directly in the Parameter (preserves the subclass type)
                self.w = nn.Parameter(
                    torch.zeros(4, 4, dtype=torch.bfloat16).as_subclass(
                        FakeNVFP4Tensor
                    ),
                    requires_grad=False,
                )

        m = M()
        assert fq.model_has_quantized_params(m)  # detected via the registry
        assert not fq.model_has_nonfloat_params(m)  # but NOT a plain non-float param

        # built-in torchao names are detected out of the box
        assert "NVFP4Tensor" in fq._QUANT_TENSOR_CLASS_NAMES
        assert "Float8Tensor" in fq._QUANT_TENSOR_CLASS_NAMES
    finally:  # restore the global registry so tests stay order-independent
        fq._QUANT_TENSOR_CLASS_NAMES.clear()
        fq._QUANT_TENSOR_CLASS_NAMES.update(saved)
