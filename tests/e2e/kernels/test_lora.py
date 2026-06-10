"""Tests for LoRA custom autograd."""

import pytest
import torch
from bitsandbytes.functional import QuantState
from torch import nn

from axolotl.kernels.geglu import geglu_backward, geglu_forward
from axolotl.kernels.lora import (
    LoRA_MLP,
    LoRA_O,
    LoRA_QKV,
    apply_lora_mlp_geglu,
    apply_lora_mlp_swiglu,
    get_lora_parameters,
    matmul_lora,
)
from axolotl.kernels.swiglu import swiglu_backward, swiglu_forward


@pytest.fixture
def mock_quantstate():
    """Creates a mock QuantState for testing"""
    shape = (64, 64)
    n_blocks = shape[0]  # Assuming blockwise quantization along first dimension

    # Create nested state first
    nested_state = QuantState(
        absmax=torch.ones(n_blocks, device="cuda"),  # One value per block
        shape=shape,
        code=torch.randint(0, 15, shape, device="cuda"),  # NF4 range is 0-15
        dtype=torch.float16,
        blocksize=64,
        quant_type="nf4",
        offset=None,
        state2=None,
    )

    # Create main state with nested state
    return QuantState(
        absmax=torch.ones(n_blocks, device="cuda"),
        shape=shape,
        code=torch.randint(0, 15, shape, device="cuda"),
        dtype=torch.float16,
        blocksize=64,
        quant_type="nf4",
        offset=torch.zeros(n_blocks, dtype=torch.int32, device="cuda"),
        state2=nested_state,
    )


@pytest.fixture
def sample_tensors():
    """Creates sample tensors for testing"""
    torch.manual_seed(42)
    batch_size, seq_len, hidden_dim = 2, 3, 64
    rank = 8
    out_dim = hidden_dim

    return {
        "X": torch.randn(
            batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16
        ),
        "W": torch.randn(out_dim, hidden_dim, device="cuda", dtype=torch.float16),
        "b": torch.randn(out_dim, device="cuda", dtype=torch.float16),
        "scale": 0.5,
        "shapes": {
            "batch": batch_size,
            "seq": seq_len,
            "hidden": hidden_dim,
            "out": out_dim,
            "rank": rank,
        },
    }


@pytest.fixture
def mock_proj():
    """Creates a mock projection module for testing."""

    class MockProj(nn.Module):
        """Mock projection class."""

        def __init__(self, in_features=64, out_features=128, rank=8):
            super().__init__()
            self.base_layer = nn.Linear(in_features, out_features)
            self.base_layer.to("cuda")
            self.lora_A = nn.ModuleDict(
                {"default": nn.Linear(in_features, rank, bias=False).to("cuda")}
            )
            self.lora_B = nn.ModuleDict(
                {"default": nn.Linear(rank, out_features, bias=False).to("cuda")}
            )
            self.scaling = {"default": 0.5}
            self.active_adapter = "default"
            self.disable_adapters = False
            self.merged = False

    return MockProj()


def test_get_lora_parameters(mock_proj):
    """Tests get_lora_parameters function"""
    # Test with LoRA enabled
    W, b, _, A, B, s, *_ = get_lora_parameters(mock_proj)

    assert isinstance(W, torch.Tensor)
    assert W.shape == (128, 64)
    assert b.shape == (128,)
    assert A.shape == (8, 64)
    assert B.shape == (128, 8)
    assert s == 0.5

    # Test with LoRA disabled
    mock_proj.disable_adapters = True
    W, b, _, A, B, s, *_ = get_lora_parameters(mock_proj)
    assert A is None and B is None and s is None

    # Test with merged state
    mock_proj.disable_adapters = False
    mock_proj.merged = True
    W, b, _, A, B, s, *_ = get_lora_parameters(mock_proj)
    assert A is None and B is None and s is None


def test_matmul_lora(sample_tensors):
    """Tests matmul_lora function"""
    X = sample_tensors["X"]
    W = sample_tensors["W"]
    b = sample_tensors["b"]
    scale = sample_tensors["scale"]

    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

    # Test base matmul
    out1 = matmul_lora(X, W, b, None, None, None, None)
    matmul = torch.matmul(X, W.t())
    expected1 = matmul + b
    assert torch.allclose(out1, expected1, rtol=1e-3)

    # Test with LoRA. matmul_lora fuses the add via in-place addmm_ (fp32 accumulate,
    # no [M, out] LoRA temp), so compare against an fp32 reference rather than a fp16
    # one whose intermediate rounding the kernel intentionally no longer mirrors.
    out2 = matmul_lora(X, W, b, None, A, B, scale)
    expected2 = (
        X.float() @ W.float().t()
        + scale * (X.float() @ A.float().t()) @ B.float().t()
        + b.float()
    )
    assert torch.allclose(out2.float(), expected2, rtol=2e-3, atol=2e-2)

    # Test 3D input reshaping
    X_3d = X.clone()
    out3 = matmul_lora(X_3d, W, b, None, A, B, scale)
    assert out3.shape == (X.shape[0], X.shape[1], W.shape[0])


@pytest.mark.parametrize(
    "activation_forward,activation_backward",
    [(swiglu_forward, swiglu_backward), (geglu_forward, geglu_backward)],
)
def test_lora_mlp_direct(sample_tensors, activation_forward, activation_backward):
    """Tests LoRA_MLP directly with different activation functions"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]

    # Create linear layers
    gate_proj = nn.Linear(hidden_dim, out_dim).to(device="cuda", dtype=torch.float16)
    up_proj = nn.Linear(hidden_dim, out_dim).to(device="cuda", dtype=torch.float16)
    down_proj = nn.Linear(out_dim, hidden_dim).to(device="cuda", dtype=torch.float16)

    # Test SwiGLU path
    X.requires_grad = True
    output = LoRA_MLP.apply(
        X,
        None,  # X_drop
        gate_proj.weight,
        gate_proj.bias,
        None,  # gate_quant
        None,  # gate_A
        None,  # gate_B
        None,  # gate_scale
        None,  # gate_lora_bias
        None,  # gate_magnitude
        up_proj.weight,
        up_proj.bias,
        None,  # up_quant
        None,  # up_A
        None,  # up_B
        None,  # up_scale
        None,  # up_lora_bias
        None,  # up_magnitude
        down_proj.weight,
        down_proj.bias,
        None,  # down_quant
        None,  # down_A
        None,  # down_B
        None,  # down_scale
        None,  # down_lora_bias
        None,  # down_magnitude
        activation_forward,
        activation_backward,
        True,  # inplace
    )

    assert output.shape == X.shape
    assert not torch.isnan(output).any()

    # Test backward pass
    loss = output.sum()
    loss.backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()


@pytest.mark.parametrize(
    "activation_forward,activation_backward",
    [(swiglu_forward, swiglu_backward), (geglu_forward, geglu_backward)],
)
def test_lora_mlp_with_adapters(
    sample_tensors, activation_forward, activation_backward
):
    """Tests LoRA_MLP with LoRA adapters"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    # Create LoRA components
    gate_A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    gate_B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)
    up_A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    up_B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)
    down_A = torch.randn(rank, out_dim, device="cuda", dtype=torch.float16)
    down_B = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float16)
    scale = 0.5

    gate_proj = nn.Linear(hidden_dim, out_dim).to(device="cuda", dtype=torch.float16)
    up_proj = nn.Linear(hidden_dim, out_dim).to(device="cuda", dtype=torch.float16)
    down_proj = nn.Linear(out_dim, hidden_dim).to(device="cuda", dtype=torch.float16)

    X.requires_grad = True
    gate_A.requires_grad = True
    gate_B.requires_grad = True
    up_A.requires_grad = True
    up_B.requires_grad = True
    down_A.requires_grad = True
    down_B.requires_grad = True

    # Forward pass with adapters
    output = LoRA_MLP.apply(
        X,
        None,  # X_drop
        gate_proj.weight,
        gate_proj.bias,
        None,
        gate_A,
        gate_B,
        scale,
        None,  # gate_lora_bias
        None,  # gate_magnitude
        up_proj.weight,
        up_proj.bias,
        None,
        up_A,
        up_B,
        scale,
        None,  # up_lora_bias
        None,  # up_magnitude
        down_proj.weight,
        down_proj.bias,
        None,
        down_A,
        down_B,
        scale,
        None,  # down_lora_bias
        None,  # down_magnitude
        activation_forward,
        activation_backward,
        True,
    )

    assert output.shape == X.shape
    assert not torch.isnan(output).any()

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check all gradients
    assert X.grad is not None
    assert gate_A.grad is not None
    assert gate_B.grad is not None
    assert up_A.grad is not None
    assert up_B.grad is not None
    assert down_A.grad is not None
    assert down_B.grad is not None

    assert not torch.isnan(X.grad).any()
    assert not torch.isnan(gate_A.grad).any()
    assert not torch.isnan(gate_B.grad).any()
    assert not torch.isnan(up_A.grad).any()
    assert not torch.isnan(up_B.grad).any()
    assert not torch.isnan(down_A.grad).any()
    assert not torch.isnan(down_B.grad).any()


def test_lora_qkv(sample_tensors):
    """Tests LoRA QKV implementation with and without adapters"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    rank = shapes["rank"]

    # Create base weights
    q_weight = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=torch.float16)
    k_weight = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=torch.float16)
    v_weight = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=torch.float16)

    # Create LoRA matrices
    q_A = torch.randn(
        rank, hidden_dim, device="cuda", dtype=torch.float16, requires_grad=True
    )
    q_B = torch.randn(
        hidden_dim, rank, device="cuda", dtype=torch.float16, requires_grad=True
    )
    k_A = torch.randn(
        rank, hidden_dim, device="cuda", dtype=torch.float16, requires_grad=True
    )
    k_B = torch.randn(
        hidden_dim, rank, device="cuda", dtype=torch.float16, requires_grad=True
    )
    v_A = torch.randn(
        rank, hidden_dim, device="cuda", dtype=torch.float16, requires_grad=True
    )
    v_B = torch.randn(
        hidden_dim, rank, device="cuda", dtype=torch.float16, requires_grad=True
    )
    scale = 0.5

    X.requires_grad = True

    # Test without LoRA adapters

    Q1, K1, V1 = LoRA_QKV.apply(
        X,
        None,  # X_drop
        q_weight,
        None,
        None,
        None,
        None,
        None,
        None,
        None,  # Q: weight, bias, quant, A, B, scale, lora_bias, magnitude
        k_weight,
        None,
        None,
        None,
        None,
        None,
        None,
        None,  # K
        v_weight,
        None,
        None,
        None,
        None,
        None,
        None,
        None,  # V
        True,  # inplace
    )

    assert Q1.shape == K1.shape == V1.shape == X.shape
    loss1 = (Q1 + K1 + V1).sum()
    loss1.backward()
    assert X.grad is not None

    # Clear gradients
    X.grad = None

    # Test with LoRA adapters
    Q2, K2, V2 = LoRA_QKV.apply(
        X,
        None,  # X_drop
        q_weight,
        None,
        None,
        q_A,
        q_B,
        scale,
        None,
        None,  # Q
        k_weight,
        None,
        None,
        k_A,
        k_B,
        scale,
        None,
        None,  # K
        v_weight,
        None,
        None,
        v_A,
        v_B,
        scale,
        None,
        None,  # V
        True,  # inplace
    )

    assert Q2.shape == K2.shape == V2.shape == X.shape
    loss2 = (Q2 + K2 + V2).sum()
    loss2.backward()

    # Check gradients
    assert X.grad is not None
    assert q_A.grad is not None
    assert q_B.grad is not None
    assert k_A.grad is not None
    assert k_B.grad is not None
    assert v_A.grad is not None
    assert v_B.grad is not None

    # Check for NaN values
    assert not torch.isnan(X.grad).any()
    assert not torch.isnan(q_A.grad).any()
    assert not torch.isnan(q_B.grad).any()
    assert not torch.isnan(k_A.grad).any()
    assert not torch.isnan(k_B.grad).any()
    assert not torch.isnan(v_A.grad).any()
    assert not torch.isnan(v_B.grad).any()


def test_lora_o(sample_tensors):
    """Tests LoRA output projection"""
    X = sample_tensors["X"]
    W = sample_tensors["W"]
    b = sample_tensors["b"]
    scale = sample_tensors["scale"]

    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

    # Test forward pass
    X.requires_grad = True
    output = LoRA_O.apply(
        X, None, W, b, None, A, B, scale, None, None
    )  # X_drop, ..., lora_bias, magnitude

    assert output.shape == (X.shape[0], X.shape[1], W.shape[0])

    # Test backward pass
    loss = output.sum()
    loss.backward()
    assert X.grad is not None


def test_with_quantization(sample_tensors, mock_quantstate):
    """Tests LoRA with quantized weights"""
    X = sample_tensors["X"]  # [batch, seq, hidden]
    W = sample_tensors["W"]  # [out, hidden]
    b = sample_tensors["b"]  # [out]
    scale = 0.5

    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

    # Test matmul with quantization
    out = matmul_lora(X, W, b, mock_quantstate, A, B, scale)
    assert out.shape == (X.shape[0], X.shape[1], W.shape[0])
    assert not torch.isnan(out).any()

    # Test with different batch sizes
    X2 = torch.randn(4, 6, hidden_dim, device="cuda", dtype=torch.float16)
    out2 = matmul_lora(X2, W, b, mock_quantstate, A, B, scale)
    assert out2.shape == (4, 6, W.shape[0])
    assert not torch.isnan(out2).any()


@pytest.mark.parametrize(
    "batch,seq,hidden,rank,out",
    [
        (1, 1, 32, 4, 64),
        (2, 3, 64, 8, 128),
        (4, 5, 128, 16, 256),
    ],
)
def test_shapes_and_dimensions(batch, seq, hidden, rank, out):
    """Tests various input shapes and dimensions"""
    X = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    W = torch.randn(out, hidden, device="cuda", dtype=torch.float16)
    b = torch.randn(out, device="cuda", dtype=torch.float16)
    A = torch.randn(rank, hidden, device="cuda", dtype=torch.float16)
    B = torch.randn(out, rank, device="cuda", dtype=torch.float16)
    scale = 0.5

    result = matmul_lora(X, W, b, None, A, B, scale)
    assert result.shape == (batch, seq, out)


def test_gradient_flow(sample_tensors):
    """Tests gradient flow through LoRA layers"""
    X = sample_tensors["X"].clone()
    W = sample_tensors["W"].clone()
    b = sample_tensors["b"].clone()
    scale = sample_tensors["scale"]

    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

    X.requires_grad = True
    A.requires_grad = True
    B.requires_grad = True

    # Forward pass
    out = matmul_lora(X, W, b, None, A, B, scale)
    loss = out.sum()

    # Backward pass
    loss.backward()

    assert X.grad is not None
    assert A.grad is not None
    assert B.grad is not None
    assert not torch.isnan(X.grad).any()
    assert not torch.isnan(A.grad).any()
    assert not torch.isnan(B.grad).any()


@pytest.mark.parametrize(
    "apply_function",
    [apply_lora_mlp_swiglu, apply_lora_mlp_geglu],
)
def test_inplace_operations(sample_tensors, apply_function):
    """Tests inplace operation behavior"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]

    # Create MLP with both inplace=True and inplace=False
    mlp = type(
        "MLPModule",
        (),
        {
            "gate_proj": nn.Linear(shapes["hidden"], shapes["out"]).to(
                device="cuda", dtype=torch.float16
            ),
            "up_proj": nn.Linear(shapes["hidden"], shapes["out"]).to(
                device="cuda", dtype=torch.float16
            ),
            "down_proj": nn.Linear(shapes["out"], shapes["hidden"]).to(
                device="cuda", dtype=torch.float16
            ),
            "training": False,
        },
    )

    out1 = apply_function(mlp, X.clone(), inplace=True)
    out2 = apply_function(mlp, X.clone(), inplace=False)

    assert torch.allclose(out1, out2, rtol=1e-3)


def _lora_pair(out_f, in_f, rank, device="cuda"):
    A = (torch.randn(rank, in_f, device=device) * 0.02).requires_grad_(True)
    B = (torch.randn(out_f, rank, device=device) * 0.02).requires_grad_(True)
    return A, B


@pytest.mark.parametrize("inplace", [True, False])
def test_batched_qkv_matches_per_module(inplace):
    """Batched LoRA QKV must match the per-module path bit-for-bit (same math)."""
    torch.manual_seed(0)
    bsz, seq, in_f, rank = 2, 7, 256, 16
    dt = torch.bfloat16
    qo, ko, vo = 256, 128, 128

    def run(batched):
        torch.manual_seed(1)
        X = torch.randn(bsz, seq, in_f, device="cuda", dtype=dt, requires_grad=True)
        weights = [
            torch.randn(o, in_f, device="cuda", dtype=dt) * 0.02 for o in (qo, ko, vo)
        ]
        qA, qB = _lora_pair(qo, in_f, rank)
        kA, kB = _lora_pair(ko, in_f, rank)
        vA, vB = _lora_pair(vo, in_f, rank)
        q, k, v = LoRA_QKV.apply(
            X,
            None,
            weights[0],
            None,
            None,
            qA,
            qB,
            2.0,
            None,
            None,
            weights[1],
            None,
            None,
            kA,
            kB,
            2.0,
            None,
            None,
            weights[2],
            None,
            None,
            vA,
            vB,
            2.0,
            None,
            None,
            inplace,
            batched,
        )
        loss = (q.float() ** 2).sum() + (k.float() ** 2).sum() + (v.float() ** 2).sum()
        loss.backward()
        return (
            torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)]).detach(),
            X.grad.detach().clone(),
            [g.grad.detach().clone() for g in (qA, qB, kA, kB, vA, vB)],
        )

    o1, x1, g1 = run(False)
    o2, x2, g2 = run(True)
    assert torch.isfinite(o2).all() and torch.isfinite(x2).all()
    assert torch.equal(o1, o2)
    assert torch.equal(x1, x2)
    for a, b in zip(g1, g2, strict=False):
        assert torch.equal(a, b)


@pytest.mark.parametrize("inplace", [True, False])
def test_batched_mlp_matches_per_module(inplace):
    """Batched LoRA MLP (gate/up fused) must match per-module path bit-for-bit."""
    torch.manual_seed(0)
    bsz, seq, in_f, inter, rank = 2, 7, 256, 512, 16
    dt = torch.bfloat16

    def run(batched):
        torch.manual_seed(2)
        X = torch.randn(bsz, seq, in_f, device="cuda", dtype=dt, requires_grad=True)
        gW = torch.randn(inter, in_f, device="cuda", dtype=dt) * 0.02
        uW = torch.randn(inter, in_f, device="cuda", dtype=dt) * 0.02
        dW = torch.randn(in_f, inter, device="cuda", dtype=dt) * 0.02
        gA, gB = _lora_pair(inter, in_f, rank)
        uA, uB = _lora_pair(inter, in_f, rank)
        dA, dB = _lora_pair(in_f, inter, rank)
        out = LoRA_MLP.apply(
            X,
            None,
            gW,
            None,
            None,
            gA,
            gB,
            2.0,
            None,
            None,
            uW,
            None,
            None,
            uA,
            uB,
            2.0,
            None,
            None,
            dW,
            None,
            None,
            dA,
            dB,
            2.0,
            None,
            None,
            swiglu_forward,
            swiglu_backward,
            inplace,
            batched,
        )
        (out.float() ** 2).sum().backward()
        return (
            out.detach().reshape(-1),
            X.grad.detach().clone(),
            [g.grad.detach().clone() for g in (gA, gB, uA, uB, dA, dB)],
        )

    o1, x1, g1 = run(False)
    o2, x2, g2 = run(True)
    assert torch.isfinite(o2).all() and torch.isfinite(x2).all()
    assert torch.equal(o1, o2)
    assert torch.equal(x1, x2)
    for a, b in zip(g1, g2, strict=False):
        assert torch.equal(a, b)
