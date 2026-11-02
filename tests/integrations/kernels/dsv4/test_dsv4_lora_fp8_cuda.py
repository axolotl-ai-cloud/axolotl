"""The DeepSeek-V4 native-fp8 attention LoRA runs transformers' real ``fp8_linear`` (DeepGEMM on
sm90/sm100, the finegrained-fp8 Triton kernel elsewhere) and matches a bf16 dequantized
reference forward and backward."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason="CUDA with FP8 support (sm89+) required",
    ),
]

DEV = "cuda"
BLOCK = [128, 128]


def _rel(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-12)


def _quantize(out_features, in_features):
    w = torch.randn(out_features, in_features, device=DEV) * 0.05
    rows, cols = out_features // BLOCK[0], in_features // BLOCK[1]
    blocks = w.reshape(rows, BLOCK[0], cols, BLOCK[1])
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-8) / 448.0
    qdata = (blocks / scale[:, None, :, None]).reshape(w.shape)
    return qdata.to(torch.float8_e4m3fn), scale.float()


@pytest.mark.parametrize("tokens", [64, 300])
def test_native_fp8_lora_matches_dequant_reference(tokens):
    from axolotl.integrations.kernels.libs.dsv4.lora_fp8 import _NativeFp8Lora
    from axolotl.kernels.quantize import dequantize_fp8

    torch.manual_seed(0)
    dt = torch.bfloat16
    qdata, scale_inv = _quantize(1024, 512)
    x = torch.randn(2, tokens, 512, device=DEV, dtype=dt, requires_grad=True)
    A = (torch.randn(16, 512, device=DEV, dtype=dt) * 0.05).requires_grad_()
    B = (torch.randn(1024, 16, device=DEV, dtype=dt) * 0.05).requires_grad_()
    grad = torch.randn(2, tokens, 1024, device=DEV, dtype=dt)

    out = _NativeFp8Lora.apply(x, qdata, scale_inv, BLOCK, A, B, 2.0, None)
    out.backward(grad)

    x_ref, A_ref, B_ref = (t.detach().clone().requires_grad_() for t in (x, A, B))
    w = dequantize_fp8(qdata, scale_inv, dt)
    ref = x_ref @ w.t() + 2.0 * (x_ref @ A_ref.t()) @ B_ref.t()
    ref.backward(grad)

    assert out.dtype == dt and out.shape == ref.shape
    # the activation is quantized to fp8 (1x128) in the forward only
    assert _rel(out, ref) < 5e-2
    for got, want in ((x, x_ref), (A, A_ref), (B, B_ref)):
        assert _rel(got.grad, want.grad) < 1e-2
