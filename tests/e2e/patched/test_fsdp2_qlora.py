"""Integration tests for FSDP Params4bit patches."""

from unittest.mock import Mock, patch

import bitsandbytes as bnb
import pytest
import torch
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

from axolotl.monkeypatch.fsdp2_qlora import (
    apply_bnb_torch_function_patch,
    patched_torch_function,
)


@pytest.fixture
def mock_params4bit():
    """Create a mock Params4bit instance with test attributes."""
    mock_instance = Mock()
    mock_instance.requires_grad = True
    mock_instance.quant_state = "test_state"
    mock_instance.blocksize = 128
    mock_instance.compress_statistics = True
    mock_instance.quant_type = "fp4"
    mock_instance.quant_storage = "test_storage"
    mock_instance.module = "test_module"
    mock_instance.bnb_quantized = True
    return mock_instance


class TestBnbTorchFunctionPatch:
    """Test the Params4bit.__torch_function__ patch."""

    def test_apply_patch(self):
        """Test that the patch can be applied."""
        with patch("bitsandbytes.nn.modules.Params4bit") as mock_cls:
            apply_bnb_torch_function_patch()
            assert hasattr(mock_cls, "__torch_function__")
            assert isinstance(mock_cls.__torch_function__, classmethod)

    # pylint: disable=redefined-outer-name
    def test_torch_chunk_preserves_attributes(self, mock_params4bit):
        """Test that torch.chunk preserves Params4bit attributes."""
        mock_cls = Mock()
        chunks = (torch.tensor([1, 2]), torch.tensor([3, 4]))

        with patch("torch.nn.Parameter.__torch_function__", return_value=chunks):
            result = patched_torch_function(
                mock_cls,
                torch.chunk,
                (type(mock_params4bit),),
                args=(mock_params4bit, 2),
            )

            assert isinstance(result, tuple)
            assert len(result) == 2

            # Check that Params4bit constructor was called with preserved attributes
            assert mock_cls.call_count == 2
            for call in mock_cls.call_args_list:
                kwargs = call[1]
                assert kwargs["requires_grad"] == mock_params4bit.requires_grad
                assert kwargs["quant_state"] == mock_params4bit.quant_state
                assert kwargs["blocksize"] == mock_params4bit.blocksize

    # pylint: disable=redefined-outer-name
    def test_other_functions_fallback(self, mock_params4bit):
        """Test that non-chunk/split functions use Parameter fallback."""
        mock_cls = Mock()
        fallback_result = torch.tensor([5, 6, 7])

        with patch(
            "torch.nn.Parameter.__torch_function__", return_value=fallback_result
        ) as mock_fallback:
            result = patched_torch_function(
                mock_cls, torch.add, (type(mock_params4bit),), args=(mock_params4bit, 1)
            )

            # Should call Parameter.__torch_function__ and return its result
            mock_fallback.assert_called_once()
            assert result is fallback_result
            mock_cls.assert_not_called()


class TestFSDPPatchIntegration:
    """Test FSDP patch integration."""

    @pytest.mark.integration
    def test_all_patches_together(self):
        """Test that all patches can be applied together."""
        from axolotl.monkeypatch.fsdp2_qlora import (
            apply_init_sharded_param_patch,
            apply_init_unsharded_param_patch,
        )

        # Store original methods before patching
        original_torch_function = getattr(
            bnb.nn.modules.Params4bit, "__torch_function__", None
        )

        # pylint: disable=protected-access
        original_init_sharded = FSDPParam._init_sharded_param
        original_init_unsharded = FSDPParam.init_unsharded_param

        # Apply patches
        apply_bnb_torch_function_patch()
        apply_init_sharded_param_patch()
        apply_init_unsharded_param_patch()

        # Verify patches were applied
        current_torch_function = getattr(
            bnb.nn.modules.Params4bit, "__torch_function__", None
        )
        if original_torch_function is not None:
            assert (
                current_torch_function != original_torch_function
            ), "Params4bit.__torch_function__ was not patched"
        else:
            assert (
                current_torch_function is not None
            ), "Params4bit.__torch_function__ was not added"

        # Check that FSDP methods were patched
        assert (
            # pylint: disable=protected-access
            FSDPParam._init_sharded_param
            != original_init_sharded
        ), "_init_sharded_param was not patched"
        assert (
            FSDPParam.init_unsharded_param != original_init_unsharded
        ), "init_unsharded_param was not patched"
