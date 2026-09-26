"""AMP decorators in the gradient-checkpointing offload paths."""

import pytest
import torch

from axolotl.kernels.utils import torch_amp_custom_fwd
from axolotl.monkeypatch.gradient_checkpointing import offload_disk
from axolotl.monkeypatch.gradient_checkpointing.offload_cpu import (
    CPU_Offloaded_Gradient_Checkpointer,
)


@pytest.mark.parametrize(
    "custom_fwd",
    [torch_amp_custom_fwd, offload_disk.torch_cuda_amp_custom_fwd],
)
def test_amp_decorators_resolve_a_valid_device_type(custom_fwd):
    """The decorators must not end up with the invalid device type "None".

    ``str(torch.accelerator.current_accelerator())`` is the string ``"None"``
    on builds without an accelerator (e.g. CPU-only wheels), and every call to
    a decorated function then raises a RuntimeError.
    """

    @custom_fwd
    def double(x):
        return x * 2

    torch.testing.assert_close(double(torch.ones(3)), torch.full((3,), 2.0))


def test_cpu_offloaded_checkpointer_matches_baseline():
    """Forward/backward roundtrip on whichever device this test runs on."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward_function(hidden_states, weight):
        return hidden_states @ weight

    weight = torch.randn(3, 2, device=device)
    reference = torch.randn(4, 3, device=device, requires_grad=True)
    hidden_states = reference.detach().clone().requires_grad_(True)

    output = CPU_Offloaded_Gradient_Checkpointer.apply(
        forward_function, hidden_states, weight
    )
    output.sum().backward()
    (reference @ weight).sum().backward()

    torch.testing.assert_close(output, reference @ weight)
    torch.testing.assert_close(hidden_states.grad, reference.grad)
