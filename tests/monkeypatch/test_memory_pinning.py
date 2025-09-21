"""Unit tests for memory pinning monkeypatch for FSDP v1."""

import unittest

import torch

from axolotl.monkeypatch.memory_pinning import patch_memory_pinning


class TestMemoryPinning(unittest.TestCase):
    """
    Unit test class for memory pinning monkeypatch
    """

    def setUp(self):
        """Set up test fixtures."""
        self.original_pin_memory = torch.Tensor.pin_memory

        if hasattr(torch.Tensor, "_original_pin_memory"):
            delattr(torch.Tensor, "_original_pin_memory")
        torch.Tensor.pin_memory = self.original_pin_memory

    def tearDown(self):
        """Restore original state after tests."""
        torch.Tensor.pin_memory = self.original_pin_memory
        if hasattr(torch.Tensor, "_original_pin_memory"):
            delattr(torch.Tensor, "_original_pin_memory")

    def test_torch_methods_are_patchable(self):
        """Test that the upstream torch methods we patch still exist."""
        assert hasattr(torch.Tensor, "pin_memory")
        assert callable(torch.Tensor.pin_memory)

    def test_memory_pinning_patch(self):
        """Test that the patch correctly disables tensor memory pinning for FSDP v1."""
        patch_memory_pinning()

        assert hasattr(torch.Tensor, "_original_pin_memory")
        assert torch.Tensor._original_pin_memory == self.original_pin_memory

        t = torch.randn(10, 10)
        t_pinned = t.pin_memory()
        assert t.data_ptr() == t_pinned.data_ptr()

        assert not t_pinned.is_pinned()

    def test_patch_idempotency(self):
        """Test that applying the patch twice is safe."""
        patch_memory_pinning()

        first_patch_fn = torch.Tensor.pin_memory

        patch_memory_pinning()

        assert torch.Tensor.pin_memory == first_patch_fn
        assert hasattr(torch.Tensor, "_original_pin_memory")
        assert torch.Tensor._original_pin_memory == self.original_pin_memory


if __name__ == "__main__":
    unittest.main()
