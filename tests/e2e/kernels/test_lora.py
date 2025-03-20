"""Tests for LoRA custom autograd."""

# pylint: disable=invalid-name,redefined-outer-name

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
    W, _, A, B, s = get_lora_parameters(mock_proj)

    assert isinstance(W, torch.Tensor)
    assert W.shape == (128, 64)
    assert A.shape == (8, 64)
    assert B.shape == (128, 8)
    assert s == 0.5

    # Test with LoRA disabled
    mock_proj.disable_adapters = True
    W, _, A, B, s = get_lora_parameters(mock_proj)
    assert A is None and B is None and s is None

    # Test with merged state
    mock_proj.disable_adapters = False
    mock_proj.merged = True
    W, _, A, B, s = get_lora_parameters(mock_proj)
    assert A is None and B is None and s is None


def test_matmul_lora(sample_tensors):
    """Tests matmul_lora function"""
    X = sample_tensors["X"]
    W = sample_tensors["W"]
    scale = sample_tensors["scale"]

    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

    # Test base matmul
    out1 = matmul_lora(X, W, None, None, None, None)
    expected1 = torch.matmul(X, W.t())
    assert torch.allclose(out1, expected1, rtol=1e-3)

    # Test with LoRA
    out2 = matmul_lora(X, W, None, A, B, scale)
    lora_term = scale * torch.matmul(torch.matmul(X, A.t()), B.t())
    expected2 = expected1 + lora_term
    assert torch.allclose(out2, expected2, rtol=1e-3)

    # Test 3D input reshaping
    X_3d = X.clone()
    out3 = matmul_lora(X_3d, W, None, A, B, scale)
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
        gate_proj.weight,
        None,  # gate_quant
        None,  # gate_A
        None,  # gate_B
        None,  # gate_scale
        up_proj.weight,
        None,  # up_quant
        None,  # up_A
        None,  # up_B
        None,  # up_scale
        down_proj.weight,
        None,  # down_quant
        None,  # down_A
        None,  # down_B
        None,  # down_scale
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
        gate_proj.weight,
        None,
        gate_A,
        gate_B,
        scale,
        up_proj.weight,
        None,
        up_A,
        up_B,
        scale,
        down_proj.weight,
        None,
        down_A,
        down_B,
        scale,
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
        q_weight,
        None,
        None,
        None,
        None,
        k_weight,
        None,
        None,
        None,
        None,
        v_weight,
        None,
        None,
        None,
        None,
        True,
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
        q_weight,
        None,
        q_A,
        q_B,
        scale,
        k_weight,
        None,
        k_A,
        k_B,
        scale,
        v_weight,
        None,
        v_A,
        v_B,
        scale,
        True,
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
    scale = sample_tensors["scale"]

    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

    # Test forward pass
    X.requires_grad = True
    output = LoRA_O.apply(X, W, None, A, B, scale)

    assert output.shape == (X.shape[0], X.shape[1], W.shape[0])

    # Test backward pass
    loss = output.sum()
    loss.backward()
    assert X.grad is not None


def test_with_quantization(sample_tensors, mock_quantstate):
    """Tests LoRA with quantized weights"""
    X = sample_tensors["X"]  # [batch, seq, hidden]
    W = sample_tensors["W"]  # [out, hidden]
    scale = 0.5

    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    A = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float16)
    B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

    # Test matmul with quantization
    out = matmul_lora(X, W, mock_quantstate, A, B, scale)
    assert out.shape == (X.shape[0], X.shape[1], W.shape[0])
    assert not torch.isnan(out).any()

    # Test with different batch sizes
    X2 = torch.randn(4, 6, hidden_dim, device="cuda", dtype=torch.float16)
    out2 = matmul_lora(X2, W, mock_quantstate, A, B, scale)
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
    A = torch.randn(rank, hidden, device="cuda", dtype=torch.float16)
    B = torch.randn(out, rank, device="cuda", dtype=torch.float16)
    scale = 0.5

    result = matmul_lora(X, W, None, A, B, scale)
    assert result.shape == (batch, seq, out)


def test_gradient_flow(sample_tensors):
    """Tests gradient flow through LoRA layers"""
    X = sample_tensors["X"].clone()
    W = sample_tensors["W"].clone()
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
    out = matmul_lora(X, W, None, A, B, scale)
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
        },
    )

    out1 = apply_function(mlp, X.clone(), inplace=True)
    out2 = apply_function(mlp, X.clone(), inplace=False)

    assert torch.allclose(out1, out2, rtol=1e-3)
