"""Tests for the λ == 1 int8 (W2A8) forward path.

The eager `TernaryLinear` (fused=False) is the oracle: the int8 path must reproduce
its forward and its straight-through gradients, and — because integer accumulation is
exact — land *closer* to the real-valued result than the bf16 GEMM it replaces.
"""

from __future__ import annotations

import pytest
import torch

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.kernels import int8_gemm as int8
from axolotl.integrations.ternary.modules import TernaryLinear

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="int8 tensor-core path requires CUDA"
)


def ternary_linear(
    in_features: int,
    out_features: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    baked: bool = False,
    **kwargs,
) -> TernaryLinear:
    """A randomly initialized eager `TernaryLinear`, optionally baked to `code · s16`."""
    module = TernaryLinear(
        in_features, out_features, fused=False, device=device, dtype=dtype, **kwargs
    )
    with torch.no_grad():
        module.weight.normal_(0.0, 0.02)
    if baked:
        module._post_training(None, "")  # type: ignore[arg-type]
    return module


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.double() - want.double()).norm() / want.double().norm()).item()


def ideal_forward(module: TernaryLinear, x: torch.Tensor) -> torch.Tensor:
    """The real-valued `(x_codes · s_x) @ (w_codes · s16).T` both paths approximate."""
    w_codes, w_scale = int8._derive_weight_codes(module)
    x_codes, x_scale = quant.act_quant_int8(x)
    return (x_codes.double() * x_scale.double()) @ (
        w_codes.double() * w_scale.double()
    ).t()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "tokens,in_features,out_features",
    [(1, 64, 64), (16, 128, 64), (17, 256, 128), (33, 512, 256), (512, 512, 512)],
)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_int8_gemm_matches_integer_oracle(
    device, tokens, in_features, out_features, out_dtype
):
    torch.manual_seed(0)
    x_codes = torch.randint(
        -127, 128, (tokens, in_features), dtype=torch.int8, device=device
    )
    w_codes = torch.randint(
        -1, 2, (out_features, in_features), dtype=torch.int8, device=device
    )
    x_scale = torch.rand(tokens, 1, device=device) * 0.01 + 1e-4
    w_scale = torch.tensor(0.0123, device=device)

    # int32 accumulation is exact, so the only float op is the pinned epilogue
    exact = (x_codes.double() @ w_codes.double().t()).float()
    expected = (exact * (x_scale * w_scale)).to(out_dtype)
    got = int8.int8_gemm(x_codes, w_codes, x_scale, w_scale, out_dtype)

    assert got.shape == (tokens, out_features)
    assert got.dtype == out_dtype
    assert torch.equal(got, expected)


@requires_cuda
@pytest.mark.parametrize(
    "tokens,in_features,out_features",
    [(1, 64, 64), (5, 64, 64), (16, 64, 64), (17, 24, 64), (32, 64, 12), (3, 60, 12)],
)
def test_int8_gemm_pads_around_int_mm_constraints(tokens, in_features, out_features):
    torch.manual_seed(0)
    x_codes = torch.randint(
        -127, 128, (tokens, in_features), dtype=torch.int8, device="cuda"
    )
    w_codes = torch.randint(
        -1, 2, (out_features, in_features), dtype=torch.int8, device="cuda"
    )
    x_scale = torch.rand(tokens, 1, device="cuda") * 0.01 + 1e-4
    w_scale = torch.tensor(0.5, device="cuda")

    with pytest.raises(RuntimeError):
        torch._int_mm(x_codes, w_codes.t())

    expected = ((x_codes.double() @ w_codes.double().t()).float()) * (x_scale * w_scale)
    got = int8.int8_gemm(x_codes, w_codes, x_scale, w_scale, torch.float32)
    assert torch.equal(got, expected)
    # padding, not the float fallback, is what made the shape runnable
    assert not int8._INT_MM_REJECTED


@pytest.mark.parametrize("device", DEVICES)
def test_int8_gemm_falls_back_when_int_mm_is_rejected(device, monkeypatch):
    calls: list[int] = []

    def rejected(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")

    monkeypatch.setattr(torch, "_int_mm", rejected)
    monkeypatch.setattr(int8, "_INT_MM_REJECTED", set())

    torch.manual_seed(0)
    x_codes = torch.randint(-127, 128, (64, 512), dtype=torch.int8, device=device)
    w_codes = torch.randint(-1, 2, (256, 512), dtype=torch.int8, device=device)
    x_scale = torch.rand(64, 1, device=device) * 0.01 + 1e-4
    w_scale = torch.tensor(0.0123, device=device)

    expected = ((x_codes.double() @ w_codes.double().t()).float()) * (x_scale * w_scale)
    got = int8.int8_gemm(x_codes, w_codes, x_scale, w_scale, torch.float32)
    assert torch.equal(got, expected)

    int8.int8_gemm(x_codes, w_codes, x_scale, w_scale, torch.float32)
    # the rejected shape is memoized, so a training loop pays the exception once
    assert len(calls) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_int8_gemm_rejects_bad_operands(device):
    x_codes = torch.zeros(8, 16, dtype=torch.int8, device=device)
    w_codes = torch.zeros(16, 16, dtype=torch.int8, device=device)
    x_scale = torch.ones(8, 1, device=device)
    w_scale = torch.ones((), device=device)

    with pytest.raises(ValueError, match="2-D operands"):
        int8.int8_gemm(x_codes.unsqueeze(0), w_codes, x_scale, w_scale)
    with pytest.raises(ValueError, match="contraction dim"):
        int8.int8_gemm(
            x_codes,
            torch.zeros(16, 8, dtype=torch.int8, device=device),
            x_scale,
            w_scale,
        )
    with pytest.raises(ValueError, match="one activation scale per token"):
        int8.int8_gemm(x_codes, w_codes, torch.ones(4, 1, device=device), w_scale)
    with pytest.raises(ValueError, match="per-tensor weight scale"):
        int8.int8_gemm(x_codes, w_codes, x_scale, torch.ones(2, device=device))


@requires_cuda
@pytest.mark.parametrize("shape", [(32, 4096), (7, 4099), (4, 8, 1024), (1, 5)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_act_quant_int8_matches_oracle(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0
    got_codes, got_scale = int8.act_quant_int8(x)
    want_codes, want_scale = quant.act_quant_int8(x)

    assert torch.equal(got_scale, want_scale.reshape(-1, 1))
    assert torch.equal(got_codes, want_codes.reshape(-1, shape[-1]))


@requires_cuda
def test_act_quant_int8_zero_row_and_noncontiguous():
    x = torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16)
    codes, scale = int8.act_quant_int8(x)
    assert torch.equal(codes, torch.zeros_like(codes))
    assert torch.allclose(scale, torch.full_like(scale, quant.SCALE_EPS))

    strided = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
    got_codes, got_scale = int8.act_quant_int8(strided)
    want_codes, want_scale = quant.act_quant_int8(strided)
    assert torch.equal(got_codes, want_codes)
    assert torch.equal(got_scale, want_scale)


@requires_cuda
@pytest.mark.parametrize("shape", [(512, 512), (4096, 2048), (7, 13), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_ternary_codes_matches_oracle(shape, dtype):
    torch.manual_seed(0)
    weight = torch.randn(*shape, device="cuda", dtype=dtype) * 0.02
    scale = quant.absmean_scale(weight)
    assert torch.equal(
        int8.ternary_codes(weight, scale), quant.ternary_codes(weight, scale)
    )


@requires_cuda
def test_ternary_codes_rounds_ties_to_even_and_defers_group_scales():
    weight = torch.tensor(
        [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], device="cuda", dtype=torch.float32
    )
    scale = torch.tensor(1.0, device="cuda")
    got = int8.ternary_codes(weight, scale)
    assert got.tolist() == [-1, -1, 0, 0, 0, 1, 1]
    assert torch.equal(got, quant.ternary_codes(weight, scale))

    grouped = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    group_scale = quant.absmean_scale(grouped, 32)
    assert torch.equal(
        int8.ternary_codes(grouped, group_scale),
        quant.ternary_codes(grouped, group_scale),
    )


@requires_cuda
@pytest.mark.parametrize("tokens,out_features", [(17, 512), (1024, 4096), (3, 4097)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_triton_rescale_matches_eager(tokens, out_features, out_dtype):
    torch.manual_seed(0)
    acc = torch.randint(
        -100_000, 100_000, (tokens, out_features), dtype=torch.int32, device="cuda"
    )
    x_scale = torch.rand(tokens, 1, device="cuda") * 0.01 + 1e-4
    w_scale = torch.tensor(0.0123, device="cuda")

    got = int8._rescale(acc, x_scale, w_scale, out_dtype)
    want = (acc * (x_scale * w_scale)).to(out_dtype)
    assert torch.equal(got, want)


@requires_cuda
@pytest.mark.parametrize("baked", [False, True])
@pytest.mark.parametrize(
    "tokens,in_features,out_features",
    [(1, 2048, 512), (17, 2048, 2048), (512, 1024, 3072)],
)
def test_int8_linear_forward_beats_eager_bf16_accuracy(
    baked, tokens, in_features, out_features
):
    torch.manual_seed(0)
    module = ternary_linear(in_features, out_features, baked=baked)
    x = torch.randn(tokens, in_features, device="cuda", dtype=torch.bfloat16)

    got = int8.int8_linear_forward(module, x)
    eager = module(x)
    ideal = ideal_forward(module, x)

    assert rel_l2(got, eager) < 5e-3
    # exact integer accumulation: the int8 path is closer to the real-valued result
    assert rel_l2(got, ideal) <= rel_l2(eager, ideal)


@pytest.fixture(name="exact_fp32_matmul")
def fixture_exact_fp32_matmul():
    """Pin true fp32 matmuls; another test enabling TF32 globally would fake a mismatch."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@requires_cuda
@pytest.mark.parametrize("baked", [False, True])
def test_int8_linear_forward_matches_eager_fp32(baked, exact_fp32_matmul):
    torch.manual_seed(0)
    module = ternary_linear(1024, 2048, dtype=torch.float32, baked=baked)
    x = torch.randn(64, 1024, device="cuda")

    got = int8.int8_linear_forward(module, x)
    eager = module(x)

    assert rel_l2(got, eager) < 1e-5
    assert torch.allclose(got, eager, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_int8_linear_forward_keeps_leading_dims():
    torch.manual_seed(0)
    module = ternary_linear(1024, 512)
    x = torch.randn(2, 9, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = int8.int8_linear_forward(module, x)
    assert out.shape == (2, 9, 512)
    assert rel_l2(out, module(x)) < 5e-3

    out.backward(torch.randn_like(out))
    assert x.grad.shape == x.shape
    assert module.weight.grad.shape == module.weight.shape


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_int8_linear_forward_gradients_match_ste(dtype):
    torch.manual_seed(0)
    module = ternary_linear(1024, 1024, dtype=dtype)
    x = torch.randn(64, 1024, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn(64, 1024, device="cuda", dtype=dtype)

    out = int8.int8_linear_forward(module, x)
    assert out.requires_grad
    out.backward(grad_out)
    got_w, got_x = module.weight.grad.clone(), x.grad.clone()

    module.weight.grad = None
    x.grad = None
    module(x).backward(grad_out)

    assert rel_l2(got_w, module.weight.grad) < 1e-6
    assert rel_l2(got_x, x.grad) < 1e-2


@requires_cuda
def test_eval_cache_is_reused_and_invalidated():
    torch.manual_seed(0)
    module = ternary_linear(1024, 1024, baked=True).eval()
    x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        int8.int8_linear_forward(module, x)
        cached = getattr(module, int8._EVAL_CACHE_ATTR)
        int8.int8_linear_forward(module, x)
        assert getattr(module, int8._EVAL_CACHE_ATTR).codes is cached.codes

        module.weight.mul_(-1)
        int8.int8_linear_forward(module, x)
        refreshed = getattr(module, int8._EVAL_CACHE_ATTR)
        assert refreshed.codes is not cached.codes
        assert torch.equal(refreshed.codes, -cached.codes)


@requires_cuda
def test_training_mode_holds_no_cache():
    module = ternary_linear(1024, 1024).eval()
    x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        int8.int8_linear_forward(module, x)
        assert hasattr(module, int8._EVAL_CACHE_ATTR)

        module.train()
        int8.int8_linear_forward(module, x)
        assert not hasattr(module, int8._EVAL_CACHE_ATTR)


@requires_cuda
def test_replaced_weight_is_never_served_from_a_stale_cache():
    torch.manual_seed(0)
    module = ternary_linear(1024, 1024).eval()
    x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        int8.int8_linear_forward(module, x)
        module.weight.copy_(torch.randn_like(module.weight) * 0.05)
        got = int8.int8_linear_forward(module, x)
        assert rel_l2(got, module(x)) < 5e-3

        module._post_training(None, "")  # type: ignore[arg-type]
        baked = int8.int8_linear_forward(module, x)
        assert rel_l2(baked, module(x)) < 5e-3


@requires_cuda
@pytest.mark.parametrize(
    "kwargs,lambda_",
    [
        ({}, 0.5),
        ({}, 0.999),
        ({"activation_bits": None}, 1.0),
        ({"weight_scale": "group", "group_size": 128}, 1.0),
        ({"weight_scale": "learnable"}, 1.0),
    ],
)
def test_int8_linear_forward_declines_unsupported_modules(kwargs, lambda_):
    module = ternary_linear(1024, 1024, **kwargs)
    module.set_lambda(lambda_)
    x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    assert int8.int8_linear_forward(module, x) is None


@requires_cuda
@pytest.mark.parametrize(
    "in_features,out_features", [(256, 1024), (1024, 256), (1028, 1024)]
)
def test_int8_linear_forward_declines_unsupported_shapes(in_features, out_features):
    module = ternary_linear(in_features, out_features)
    x = torch.randn(32, in_features, device="cuda", dtype=torch.bfloat16)
    assert int8.int8_linear_forward(module, x) is None


def test_int8_linear_forward_declines_on_cpu():
    module = ternary_linear(1024, 1024, device="cpu")
    x = torch.randn(32, 1024, dtype=torch.bfloat16)
    assert int8.int8_linear_forward(module, x) is None


@requires_cuda
def test_int8_linear_forward_declines_dtype_mismatch():
    module = ternary_linear(1024, 1024)
    x = torch.randn(32, 1024, device="cuda", dtype=torch.float32)
    assert int8.int8_linear_forward(module, x) is None


@requires_cuda
@pytest.mark.parametrize(
    "in_features,out_features,dtype,expected",
    [
        (4096, 4096, torch.bfloat16, True),
        (4096, 4096, torch.float32, True),
        (512, 512, torch.bfloat16, True),
        (256, 4096, torch.bfloat16, False),
        (4096, 4100, torch.bfloat16, False),
        (4096, 4096, torch.int8, False),
        (4096, 4096, torch.float64, False),
    ],
)
def test_int8_forward_supported(in_features, out_features, dtype, expected):
    assert int8.int8_forward_supported(in_features, out_features, dtype) is expected


def test_int8_forward_supported_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not int8.int8_forward_supported(4096, 4096, torch.bfloat16)
