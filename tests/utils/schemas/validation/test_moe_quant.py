"""Tests for MoE expert quantization config validation and PEFT patch idempotency."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture()
def gpu_caps():
    return {"compute_capability": "sm_89", "bf16": True, "n_gpu": 1, "n_node": 1}


@pytest.fixture()
def env_caps():
    return {"torch_version": "2.7.0"}


class TestQuantizeMoeExpertsValidation:
    """Test suite for quantize_moe_experts config validator."""

    def test_requires_adapter(self, min_base_cfg, gpu_caps, env_caps):
        """quantize_moe_experts without adapter should fail."""
        cfg = (
            DictDefault(
                quantize_moe_experts=True,
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="requires adapter"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_requires_quantization(self, min_base_cfg, gpu_caps, env_caps):
        """quantize_moe_experts without load_in_4bit/8bit should fail."""
        cfg = (
            DictDefault(
                quantize_moe_experts=True,
                adapter="lora",
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="requires load_in_4bit or load_in_8bit"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_valid_qlora_4bit(self, min_base_cfg, gpu_caps, env_caps):
        """quantize_moe_experts with qlora + 4bit should pass."""
        cfg = (
            DictDefault(
                quantize_moe_experts=True,
                adapter="qlora",
                load_in_4bit=True,
            )
            | min_base_cfg
        )
        result = validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert result["quantize_moe_experts"] is True

    def test_valid_lora_8bit(self, min_base_cfg, gpu_caps, env_caps):
        """quantize_moe_experts with lora + 8bit should pass."""
        cfg = (
            DictDefault(
                quantize_moe_experts=True,
                adapter="lora",
                load_in_8bit=True,
            )
            | min_base_cfg
        )
        result = validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert result["quantize_moe_experts"] is True

    def test_false_skips_validation(self, min_base_cfg, gpu_caps, env_caps):
        """quantize_moe_experts=false should not check adapter/quantization."""
        cfg = (
            DictDefault(
                quantize_moe_experts=False,
            )
            | min_base_cfg
        )
        result = validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert result["quantize_moe_experts"] is False

    def test_rejects_lora_target_linear(self, min_base_cfg, gpu_caps, env_caps):
        """quantize_moe_experts with lora_target_linear should fail."""
        cfg = (
            DictDefault(
                quantize_moe_experts=True,
                adapter="qlora",
                load_in_4bit=True,
                lora_target_linear=True,
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="lora_target_linear is not compatible"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_default_is_false(self, min_base_cfg, gpu_caps, env_caps):
        """quantize_moe_experts should default to false."""
        cfg = DictDefault({}) | min_base_cfg
        result = validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert result["quantize_moe_experts"] is False


class TestLoraTargetParametersDropout:
    """Test that lora_dropout must be 0 when lora_target_parameters is set."""

    def test_rejects_nonzero_dropout(self, min_base_cfg):
        """lora_dropout > 0 with lora_target_parameters should fail."""
        cfg = (
            DictDefault(
                adapter="lora",
                lora_target_parameters=["mlp.experts.gate_up_proj"],
                lora_dropout=0.1,
                load_in_8bit=True,
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="lora_dropout must be 0"):
            validate_config(cfg)

    def test_zero_dropout_passes(self, min_base_cfg):
        """lora_dropout=0 with lora_target_parameters should pass."""
        cfg = (
            DictDefault(
                adapter="lora",
                lora_target_parameters=["mlp.experts.gate_up_proj"],
                lora_dropout=0.0,
                load_in_8bit=True,
            )
            | min_base_cfg
        )
        result = validate_config(cfg)
        assert result["lora_dropout"] == 0.0


class TestPeftPatchIdempotency:
    """Test that patch_peft_target_parameters_matching is idempotent."""

    def test_double_call_does_not_stack_wrappers(self):
        """Calling patch twice should not double-wrap _inject_parameters."""
        from peft.tuners.tuners_utils import BaseTuner

        from axolotl.monkeypatch.moe_quant import (
            patch_peft_target_parameters_matching,
        )

        original = BaseTuner._inject_parameters
        try:
            patch_peft_target_parameters_matching()
            first_patched = BaseTuner._inject_parameters
            patch_peft_target_parameters_matching()
            second_patched = BaseTuner._inject_parameters
            # Should be same function, not double-wrapped
            assert first_patched is second_patched
        finally:
            BaseTuner._inject_parameters = original
            patch_peft_target_parameters_matching._axolotl_patched = False
