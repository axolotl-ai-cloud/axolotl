"""Integration tests for FSDP2 Params4bit patches."""

import pytest
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam


class TestFSDPPatchIntegration:
    """Test FSDP patch integration."""

    @pytest.mark.integration
    def test_fsdp2_init_patches(self):
        """Test that all patches can be applied together."""
        from axolotl.monkeypatch.fsdp2_qlora import (
            apply_init_sharded_param_patch,
            apply_init_unsharded_param_patch,
        )

        original_init_sharded = FSDPParam._init_sharded_param
        original_init_unsharded = FSDPParam.init_unsharded_param

        # Apply patches
        apply_init_sharded_param_patch()
        apply_init_unsharded_param_patch()

        assert FSDPParam._init_sharded_param != original_init_sharded, (
            "_init_sharded_param was not patched"
        )
        assert FSDPParam.init_unsharded_param != original_init_unsharded, (
            "init_unsharded_param was not patched"
        )
