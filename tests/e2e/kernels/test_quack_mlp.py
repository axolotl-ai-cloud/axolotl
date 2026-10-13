"""GPU numeric gate for the quack fused gated-MLP kernel.

Requires a Hopper+ GPU and `quack-kernels` installed. Excluded from the CPU suite
(tests/e2e is ignored there). This is the pod-validation gate for the fused-MLP
path: it checks the fused forward/backward against a plain PyTorch SwiGLU/GeGLU
reference.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("quack")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
    pytest.skip(
        "quack fused MLP requires a Hopper+ (SM90+) GPU", allow_module_level=True
    )

from axolotl.integrations.quack_kernels.mlp import quack_gated_mlp_forward  # noqa: E402

# quack gated activation -> the matching PyTorch gate function.
GATE_FNS = {
    "swiglu": F.silu,
    "geglu": lambda g: F.gelu(g, approximate="tanh"),
}


def _make_mlp(hidden, inter, dtype=torch.bfloat16, dev="cuda"):
    mlp = nn.Module()
    mlp.gate_proj = nn.Linear(hidden, inter, bias=False, device=dev, dtype=dtype)
    mlp.up_proj = nn.Linear(hidden, inter, bias=False, device=dev, dtype=dtype)
    mlp.down_proj = nn.Linear(inter, hidden, bias=False, device=dev, dtype=dtype)
    return mlp


def _ref(mlp, gate_fn, x):
    return F.linear(
        gate_fn(F.linear(x, mlp.gate_proj.weight)) * F.linear(x, mlp.up_proj.weight),
        mlp.down_proj.weight,
    )


@pytest.mark.parametrize("activation", ["swiglu", "geglu"])
@pytest.mark.parametrize("tokens", [128, 512])
@pytest.mark.parametrize("hidden,inter", [(1024, 2816), (2048, 8192)])
def test_quack_gated_mlp_matches_reference(activation, tokens, hidden, inter):
    torch.manual_seed(0)
    gate_fn = GATE_FNS[activation]
    mlp = _make_mlp(hidden, inter)

    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_(True)

    out = quack_gated_mlp_forward(mlp, activation, x)
    ref = _ref(mlp, gate_fn, x_ref)

    # bf16 matmul + fused-accumulation order differs from the reference; compare loosely.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None


def test_quack_mlp_3d_input_roundtrip():
    torch.manual_seed(0)
    hidden, inter = 1024, 2816
    mlp = _make_mlp(hidden, inter)

    x = torch.randn(2, 64, hidden, device="cuda", dtype=torch.bfloat16)
    out = quack_gated_mlp_forward(mlp, "swiglu", x)
    assert out.shape == (2, 64, hidden)
    torch.testing.assert_close(out, _ref(mlp, F.silu, x), rtol=2e-2, atol=2e-2)


def test_quack_mlp_noncontiguous_input():
    torch.manual_seed(0)
    hidden, inter = 1024, 2816
    mlp = _make_mlp(hidden, inter)

    # Strided view -> last-dim stride != 1, exercising the contiguous() branch.
    x = torch.randn(256, hidden, 2, device="cuda", dtype=torch.bfloat16)[..., 0]
    assert x.stride(-1) != 1
    out = quack_gated_mlp_forward(mlp, "swiglu", x)
    torch.testing.assert_close(out, _ref(mlp, F.silu, x), rtol=2e-2, atol=2e-2)
