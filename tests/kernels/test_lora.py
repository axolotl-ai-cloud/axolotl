import pytest
import torch
import torch.nn as nn
from bitsandbytes.functional import QuantState

from axolotl.kernels.lora import (
    LoRA_MLP,
    LoRA_O,
    LoRA_QKV,
    apply_lora_mlp_geglu,
    apply_lora_mlp_swiglu,
    get_lora_parameters,
    matmul_lora,
)


@pytest.fixture
def device():
    """Returns the appropriate device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mock_quantstate(device):
    """Creates a mock QuantState for testing"""
    shape = (64, 64)
    n_blocks = shape[0]  # Assuming blockwise quantization along first dimension

    # Create nested state first
    nested_state = QuantState(
        absmax=torch.ones(n_blocks, device=device),  # One value per block
        shape=shape,
        code=torch.randint(-127, 127, shape, device=device),
        dtype=torch.float32,
        blocksize=64,
        quant_type="fp4",
        offset=None,
        state2=None,
    )

    # Create main state with nested state
    return QuantState(
        absmax=torch.ones(n_blocks, device=device),  # One value per block
        shape=shape,
        code=torch.randint(-127, 127, shape, device=device),
        dtype=torch.float32,
        blocksize=64,
        quant_type="fp4",
        offset=torch.zeros(
            n_blocks, dtype=torch.int32, device=device
        ),  # Match absmax shape
        state2=nested_state,
    )


@pytest.fixture
def sample_tensors(device):
    """Creates sample tensors for testing"""
    torch.manual_seed(42)
    batch_size, seq_len, hidden_dim = 2, 3, 64
    rank = 8
    out_dim = hidden_dim  # Make output dimension match input for QKV testing

    return {
        "X": torch.randn(batch_size, seq_len, hidden_dim, device=device),
        "W": torch.randn(out_dim, hidden_dim, device=device),
        # Note: A and B shapes will be created as needed in specific tests
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
def mock_proj(device):
    """Creates a mock projection module for testing"""

    class MockProj(nn.Module):
        def __init__(self, in_features=64, out_features=128, rank=8):
            super().__init__()
            self.base_layer = nn.Linear(in_features, out_features)
            self.base_layer.to(device)
            self.lora_A = nn.ModuleDict(
                {"default": nn.Linear(in_features, rank, bias=False).to(device)}
            )
            self.lora_B = nn.ModuleDict(
                {"default": nn.Linear(rank, out_features, bias=False).to(device)}
            )
            self.scaling = {"default": 0.5}
            self.active_adapter = "default"
            self.disable_adapters = False
            self.merged = False

    return MockProj()


def test_get_lora_parameters(mock_proj, mock_quantstate):
    """Tests get_lora_parameters function"""
    # Test with LoRA enabled
    W, quant_state, A, B, s = get_lora_parameters(mock_proj)

    assert isinstance(W, torch.Tensor)
    assert W.shape == (128, 64)
    assert A.shape == (8, 64)
    assert B.shape == (128, 8)
    assert s == 0.5

    # Test with LoRA disabled
    mock_proj.disable_adapters = True
    W, quant_state, A, B, s = get_lora_parameters(mock_proj)
    assert A is None and B is None and s is None

    # Test with merged state
    mock_proj.disable_adapters = False
    mock_proj.merged = True
    W, quant_state, A, B, s = get_lora_parameters(mock_proj)
    assert A is None and B is None and s is None


def test_matmul_lora(sample_tensors):
    """Tests matmul_lora function"""
    X = sample_tensors["X"]
    W = sample_tensors["W"]
    A = sample_tensors["A"]
    B = sample_tensors["B"]
    scale = sample_tensors["scale"]

    # Test base matmul
    out1 = matmul_lora(X, W, None, None, None, None)
    expected1 = torch.matmul(X, W.t())
    assert torch.allclose(out1, expected1, rtol=1e-4)

    # Test with LoRA
    out2 = matmul_lora(X, W, None, A, B, scale)
    lora_term = scale * torch.matmul(torch.matmul(X, A.t()), B.t())
    expected2 = expected1 + lora_term
    assert torch.allclose(out2, expected2, rtol=1e-4)

    # Test 3D input reshaping
    X_3d = X.clone()
    out3 = matmul_lora(X_3d, W, None, A, B, scale)
    assert out3.shape == (X.shape[0], X.shape[1], W.shape[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_lora_mlp_direct(sample_tensors, device):
    """Tests LoRA_MLP directly with different activation functions"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]

    # Create linear layers
    gate_proj = nn.Linear(hidden_dim, out_dim).to(device)
    up_proj = nn.Linear(hidden_dim, out_dim).to(device)
    down_proj = nn.Linear(out_dim, hidden_dim).to(device)

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
        lambda x, y: torch.nn.functional.silu(x) * y,  # swiglu forward
        lambda dw, x, y: (dw * torch.nn.functional.silu(x), dw * y),  # swiglu backward
        True,  # inplace
    )

    assert output.shape == X.shape
    assert not torch.isnan(output).any()

    # Test backward pass
    loss = output.sum()
    loss.backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()

    # Clear gradients
    X.grad = None

    # Test GEGLU path
    output = LoRA_MLP.apply(
        X,
        gate_proj.weight,
        None,
        None,
        None,
        None,
        up_proj.weight,
        None,
        None,
        None,
        None,
        down_proj.weight,
        None,
        None,
        None,
        None,
        lambda x, y: torch.nn.functional.gelu(x) * y,  # geglu forward
        lambda dw, x, y: (dw * torch.nn.functional.gelu(x), dw * y),  # geglu backward
        True,
    )

    assert output.shape == X.shape
    assert not torch.isnan(output).any()

    # Test backward pass
    loss = output.sum()
    loss.backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_lora_mlp_with_adapters(sample_tensors, device):
    """Tests LoRA_MLP with LoRA adapters"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    # Create LoRA components
    gate_A = torch.randn(rank, hidden_dim, device=device)
    gate_B = torch.randn(out_dim, rank, device=device)
    up_A = torch.randn(rank, hidden_dim, device=device)
    up_B = torch.randn(out_dim, rank, device=device)
    down_A = torch.randn(rank, out_dim, device=device)
    down_B = torch.randn(hidden_dim, rank, device=device)
    scale = 0.5

    gate_proj = nn.Linear(hidden_dim, out_dim).to(device)
    up_proj = nn.Linear(hidden_dim, out_dim).to(device)
    down_proj = nn.Linear(out_dim, hidden_dim).to(device)

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
        lambda x, y: torch.nn.functional.silu(x) * y,
        lambda dw, x, y: (dw * torch.nn.functional.silu(x), dw * y),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_lora_qkv(sample_tensors, device):
    """Tests LoRA QKV implementation with and without adapters"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    rank = shapes["rank"]

    # Create base weights
    q_weight = torch.randn(hidden_dim, hidden_dim, device=device)
    k_weight = torch.randn(hidden_dim, hidden_dim, device=device)
    v_weight = torch.randn(hidden_dim, hidden_dim, device=device)

    # Create LoRA matrices
    q_A = torch.randn(rank, hidden_dim, device=device, requires_grad=True)
    q_B = torch.randn(hidden_dim, rank, device=device, requires_grad=True)
    k_A = torch.randn(rank, hidden_dim, device=device, requires_grad=True)
    k_B = torch.randn(hidden_dim, rank, device=device, requires_grad=True)
    v_A = torch.randn(rank, hidden_dim, device=device, requires_grad=True)
    v_B = torch.randn(hidden_dim, rank, device=device, requires_grad=True)
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
    A = sample_tensors["A"]
    B = sample_tensors["B"]
    scale = sample_tensors["scale"]

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
    shapes = sample_tensors["shapes"]
    hidden_dim = shapes["hidden"]
    out_dim = shapes["out"]
    rank = shapes["rank"]

    # Create LoRA matrices with correct shapes
    A = torch.randn(rank, hidden_dim, device=X.device)  # [rank, hidden]
    B = torch.randn(out_dim, rank, device=X.device)  # [out, rank]
    scale = 0.5

    # Test matmul with quantization
    out = matmul_lora(X, W, mock_quantstate, A, B, scale)
    assert out.shape == (X.shape[0], X.shape[1], W.shape[0])
    assert not torch.isnan(out).any()

    # Test with different batch sizes
    X2 = torch.randn(4, 6, hidden_dim, device=X.device)
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
def test_shapes_and_dimensions(batch, seq, hidden, rank, out, device):
    """Tests various input shapes and dimensions"""
    X = torch.randn(batch, seq, hidden, device=device)
    W = torch.randn(out, hidden, device=device)
    A = torch.randn(rank, hidden, device=device)
    B = torch.randn(out, rank, device=device)
    scale = 0.5

    result = matmul_lora(X, W, None, A, B, scale)
    assert result.shape == (batch, seq, out)


def test_gradient_flow(sample_tensors):
    """Tests gradient flow through LoRA layers"""
    X = sample_tensors["X"].clone()
    W = sample_tensors["W"].clone()
    A = sample_tensors["A"].clone()
    B = sample_tensors["B"].clone()
    scale = sample_tensors["scale"]

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_inplace_operations(sample_tensors, device):
    """Tests inplace operation behavior"""
    X = sample_tensors["X"]
    shapes = sample_tensors["shapes"]

    # Create MLP with both inplace=True and inplace=False
    mlp = type(
        "MLPModule",
        (),
        {
            "gate_proj": nn.Linear(shapes["hidden"], shapes["out"]).to(device),
            "up_proj": nn.Linear(shapes["hidden"], shapes["out"]).to(device),
            "down_proj": nn.Linear(shapes["out"], shapes["hidden"]).to(device),
        },
    )

    out1 = apply_lora_mlp_swiglu(mlp, X.clone(), inplace=True)
    out2 = apply_lora_mlp_swiglu(mlp, X.clone(), inplace=False)

    assert torch.allclose(out1, out2, rtol=1e-4)
