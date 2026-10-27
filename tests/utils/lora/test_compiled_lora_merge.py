"""Compiled expert arithmetic preserves PEFT and NVFP4 merge outputs."""

import math

import pytest
import torch

from axolotl.cli.utils.compiled_lora_merge import (
    CompiledExpertMerge,
    merge_expert_chunk,
)
from axolotl.cli.utils.lora_merge import _build_peft_layer_and_get_delta


@pytest.fixture(autouse=True)
def reset_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _reference(base, a, b, config):
    delta = _build_peft_layer_and_get_delta(
        a,
        b,
        config,
        base,
        is_param_wrapper=True,
        is_transposed=False,
        canonical_fp32=True,
    )
    return (base.float() + delta.float()).to(torch.bfloat16)


@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("use_rslora", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_compiled_chunk_matches_peft_and_nvfp4(experts, use_rslora, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("torchao")
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_merge,
    )

    torch.manual_seed(42)
    rank = 4
    base = torch.randn(experts, 48, 32, dtype=torch.bfloat16, device=device)
    a = torch.randn(experts * rank, 32, dtype=torch.bfloat16, device=device)
    b = torch.randn(48, experts * rank, dtype=torch.bfloat16, device=device)
    config = {"r": rank, "lora_alpha": 7, "use_rslora": use_rslora}
    scaling = 7 / (math.sqrt(rank) if use_rslora else rank)
    compiler = CompiledExpertMerge()
    for _ in range(2):
        expected = _reference(base, a, b, config)
        merged = compiler(base, a, b, scaling, lambda expected=expected: expected)
        assert not compiler.disabled
        assert len(compiler.validated) == 1
        assert torch.equal(merged, expected)
        pts = torch.linspace(0.02, 0.04, experts, device=device)
        for actual, want in zip(
            quantize_nvfp4_merge(merged.float(), pts),
            quantize_nvfp4_merge(expected.float(), pts),
            strict=True,
        ):
            assert torch.equal(actual.view(torch.uint8), want.view(torch.uint8))
        a = a + 0.125


def test_chunk_preserves_bf16_rounding():
    base = torch.ones(2, 16, 16, dtype=torch.bfloat16)
    a = torch.ones(2, 16)
    b = torch.tensor([1 / 256, 3 / 256]).repeat(16, 1)
    expected = _reference(base, a, b, {"r": 1, "lora_alpha": 1})
    compiler = CompiledExpertMerge()
    result = compiler(base, a, b, 1.0, lambda: expected)
    assert not compiler.disabled
    assert torch.equal(result, expected)
    assert result[0, 0, 0] == 1.0
    assert result[1, 0, 0] == 1.015625


def test_relaxed_compile_is_explicit_and_skips_parity_check(monkeypatch, caplog):
    def compile_fn(fn, **kwargs):
        assert kwargs["options"] == {}

        def run(*args):
            assert torch.get_float32_matmul_precision() == "high"
            return fn(*args) + 1

        return run

    def reference():
        pytest.fail("Relaxed compilation must not run strict parity validation")

    monkeypatch.setattr(torch, "compile", compile_fn)
    base = torch.zeros(2, 16, 16, dtype=torch.bfloat16)
    a, b = torch.ones(2, 16), torch.ones(16, 2)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        compiler = CompiledExpertMerge("relaxed")
        result = compiler(base, a, b, 1.0, reference)
        assert torch.equal(result, merge_expert_chunk(base, a, b, 1.0) + 1)
        assert not compiler.validated and not compiler.disabled
        assert torch.get_float32_matmul_precision() == "high"
        assert "NVFP4 bytes may differ" in caplog.text
    finally:
        torch.set_float32_matmul_precision(previous)


def test_invalid_compile_precision_rejected():
    with pytest.raises(ValueError, match="Unsupported merge compile precision"):
        CompiledExpertMerge("fastest")


@pytest.mark.parametrize("failure", ["compile", "execute", "mismatch"])
def test_compiler_falls_back_and_restores_precision(monkeypatch, failure):
    calls = []

    def compile_fn(fn, **kwargs):
        calls.append(kwargs)
        assert kwargs["options"]["emulate_precision_casts"]
        if failure == "compile":
            raise RuntimeError("unsupported compiler")

        def run(*args):
            assert torch.get_float32_matmul_precision() == "highest"
            if failure == "execute":
                raise RuntimeError("unsupported operation")
            return fn(*args) + 1

        return run

    monkeypatch.setattr(torch, "compile", compile_fn)
    base = torch.zeros(2, 16, 16, dtype=torch.bfloat16)
    a, b = torch.ones(2, 16), torch.ones(16, 2)
    expected = merge_expert_chunk(base, a, b, 1.0)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        compiler = CompiledExpertMerge()
        result = compiler(base, a, b, 1.0, lambda: expected)
        assert torch.equal(result, expected)
        assert compiler.disabled
        assert torch.equal(compiler(base, a, b, 1.0, lambda: expected), expected)
        assert len(calls) == 1
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
