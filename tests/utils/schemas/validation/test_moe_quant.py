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


class TestMoeAdapterTrainMergeRoundtrip:
    """E2E: train adapter on fake quantized MoE experts, then merge without error.

    Reproduces the GLM-4.5-Air bug where weight-loading order (which determines
    module.parametrizations insertion order during training) differs from model-
    definition order (which determines named_parameters order during merge onto
    a non-quantized base model).  Without sorted-order in _inject_parameters,
    the ParamWrapper nesting is reversed between training and merge → size mismatch.
    """

    @staticmethod
    def _make_classes():
        """Return FakeExperts and FakeModel classes shared by both model builders."""
        import torch
        import torch.nn as nn

        class FakeExperts(nn.Module):
            def __init__(self):
                super().__init__()
                # Model definition order: gate_up_proj first, then down_proj.
                self.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
                self.down_proj = nn.Parameter(torch.randn(4, 8, 16))

            def forward(self, x):
                x = torch.matmul(x, self.gate_up_proj[0].T)  # (batch, 16)
                x = torch.matmul(x, self.down_proj[0].T)  # (batch, 8)
                return x

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)
                self.experts = FakeExperts()

            def forward(self, x):
                return self.linear(x) + self.experts(x)

        return FakeExperts, FakeModel

    @staticmethod
    def _make_quantized_model():
        """Training model: experts have parametrizations registered in *reversed* order.

        Simulates weight-loading order (down_proj before gate_up_proj) differing
        from model-definition order (gate_up_proj before down_proj).
        """
        import torch.nn as nn
        import torch.nn.utils.parametrize as P

        _, FakeModel = TestMoeAdapterTrainMergeRoundtrip._make_classes()

        class PassthroughParametrization(nn.Module):
            def forward(self, x):
                return x

        model = FakeModel()
        # Deliberately reversed vs __init__ order to expose the insertion-order bug.
        P.register_parametrization(
            model.experts, "down_proj", PassthroughParametrization(), unsafe=True
        )
        P.register_parametrization(
            model.experts, "gate_up_proj", PassthroughParametrization(), unsafe=True
        )
        return model

    @staticmethod
    def _make_plain_model():
        """Merge model: no parametrizations — standard branch uses definition order."""
        _, FakeModel = TestMoeAdapterTrainMergeRoundtrip._make_classes()
        return FakeModel()

    def test_train_save_merge_no_size_mismatch(self, tmp_path):
        """Train on quantized experts, merge onto plain model — must not raise."""
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from peft.tuners.tuners_utils import BaseTuner

        from axolotl.monkeypatch.moe_quant import patch_peft_target_parameters_matching

        adapter_dir = tmp_path / "adapter"
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        original_inject = BaseTuner._inject_parameters

        # Training phase: quantized model (parametrized branch).
        patch_peft_target_parameters_matching()
        try:
            peft_model = get_peft_model(self._make_quantized_model(), lora_cfg)
        finally:
            BaseTuner._inject_parameters = original_inject
            patch_peft_target_parameters_matching._axolotl_patched = False

        optimizer = torch.optim.SGD(peft_model.parameters(), lr=1e-3)
        for _ in range(3):
            peft_model(torch.randn(2, 8)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        peft_model.save_pretrained(str(adapter_dir))

        # Merge phase: plain (non-quantized) model → standard branch.
        # Without the sorted-order fix, the ParamWrapper nesting would be reversed
        # relative to training, causing RuntimeError: size mismatch here.
        patch_peft_target_parameters_matching()
        try:
            loaded = PeftModel.from_pretrained(
                self._make_plain_model(), str(adapter_dir)
            )
            merged = loaded.merge_and_unload()
        finally:
            BaseTuner._inject_parameters = original_inject
            patch_peft_target_parameters_matching._axolotl_patched = False

        assert merged is not None
