"""CPU-only embedding LoRA autocast regression, without CCE or GPU dependencies.

Run independently of the shared model-download fixtures with:
    pytest --noconftest tests/kernels/test_embedding_lora_autocast.py
"""

import ast
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def embedding_kernel():
    """Load the real embedding class with CPU AMP, bypassing GPU-only imports."""
    path = Path(__file__).resolve().parents[2] / "src/axolotl/kernels/lora.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    embedding = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LoRA_Embedding"
    )
    namespace = {
        "torch": torch,
        "torch_amp_custom_fwd": torch.amp.custom_fwd(device_type="cpu"),
        "torch_amp_custom_bwd": torch.amp.custom_bwd(device_type="cpu"),
    }
    exec(
        compile(ast.Module(body=[embedding], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["LoRA_Embedding"]


@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padding_idx", [None, 0])
@pytest.mark.parametrize("scale_grad_by_freq", [False, True])
def test_embedding_backward_autocast(
    embedding_kernel, autocast_dtype, weight_dtype, padding_idx, scale_grad_by_freq
):
    torch.manual_seed(42)
    vocab, hidden, rank = 16, 32, 4
    W = torch.randn(vocab, hidden, device="cpu", dtype=weight_dtype)
    A = torch.randn(rank, vocab, device="cpu", dtype=torch.bfloat16).float()
    B = torch.randn(hidden, rank, device="cpu", dtype=torch.bfloat16).float()
    A.requires_grad_()
    B.requires_grad_()
    x = torch.tensor([[0, 1, 1, 2], [3, 0, 1, 3]], device="cpu")
    s = 0.7

    with torch.autocast("cpu", dtype=autocast_dtype):
        out = embedding_kernel.apply(
            x, W, A, B, s, None, padding_idx, None, 2.0, scale_grad_by_freq, False
        )
    grad = torch.randn_like(out)
    d_A, d_B = torch.autograd.grad(out, (A, B), grad)

    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    after_A = torch.nn.functional.embedding(
        x,
        A_ref.t(),
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
    )
    ref = torch.nn.functional.embedding(x, W.float()) + s * (after_A @ B_ref.t())
    expected = torch.autograd.grad(ref, (A_ref, B_ref), grad.float())

    for actual, reference in zip((d_A, d_B), expected, strict=True):
        assert actual.dtype == torch.float32
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
