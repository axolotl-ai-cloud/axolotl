"""Unit tests for memory pinning monkeypatch."""

import unittest

import torch
from torch.distributed.fsdp import CPUOffloadPolicy

from axolotl.monkeypatch.memory_pinning import apply_memory_pinning_patch


class TestMemoryPinning(unittest.TestCase):
    """
    Unit test class for memory pinning monkeypatch
    """

    def setUp(self):
        """Set up test fixtures."""
        # Store original values to restore after tests
        self.original_pin_memory = torch.Tensor.pin_memory
        self.original_cpu_offload_pin_memory = CPUOffloadPolicy.pin_memory

        # Reset state - remove any previous patches
        if hasattr(torch.Tensor, "_original_pin_memory"):
            delattr(torch.Tensor, "_original_pin_memory")
        torch.Tensor.pin_memory = self.original_pin_memory
        CPUOffloadPolicy.pin_memory = self.original_cpu_offload_pin_memory

    def tearDown(self):
        """Restore original state after tests."""
        # Restore original implementations
        torch.Tensor.pin_memory = self.original_pin_memory
        CPUOffloadPolicy.pin_memory = self.original_cpu_offload_pin_memory
        if hasattr(torch.Tensor, "_original_pin_memory"):
            delattr(torch.Tensor, "_original_pin_memory")

    def test_torch_methods_are_patchable(self):
        """Test that the upstream torch methods we patch still exist."""
        assert hasattr(torch.Tensor, "pin_memory")
        assert callable(torch.Tensor.pin_memory)
        assert hasattr(CPUOffloadPolicy, "pin_memory")

    def test_memory_pinning_patch(self):
        """Test that the patch correctly disables memory pinning."""
        # Apply the patch
        apply_memory_pinning_patch()

        # Verify torch.Tensor.pin_memory was patched
        assert hasattr(torch.Tensor, "_original_pin_memory")
        assert torch.Tensor._original_pin_memory == self.original_pin_memory

        # Verify CPUOffloadPolicy.pin_memory was set to False
        self.assertFalse(CPUOffloadPolicy.pin_memory)

        # Test that pin_memory now returns self (no-op)
        t = torch.randn(10, 10)
        t_pinned = t.pin_memory()
        assert t.data_ptr() == t_pinned.data_ptr()

    def test_patch_idempotency(self):
        """Test that applying the patch twice is safe."""
        # Apply patch first time
        apply_memory_pinning_patch()

        # Store first patch result
        first_patch_fn = torch.Tensor.pin_memory

        # Apply patch second time (should be no-op)
        apply_memory_pinning_patch()

        # Verify it's still the same function
        assert torch.Tensor.pin_memory == first_patch_fn
        assert hasattr(torch.Tensor, "_original_pin_memory")
        assert torch.Tensor._original_pin_memory == self.original_pin_memory


if __name__ == "__main__":
    unittest.main()
