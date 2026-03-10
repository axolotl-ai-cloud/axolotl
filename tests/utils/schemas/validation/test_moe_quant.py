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


class TestConsistentParamWrapperNesting:
    """Test that ParamWrapper nesting order is consistent between training and merge.

    The bug: when multiple target_parameters land on the same module, PEFT creates
    nested ParamWrappers. The nesting order (and thus saved adapter key structure)
    differs between the parametrized branch (training with quantize_moe_experts) and the
    standard branch (merge without quantize_moe_experts) because the former uses sorted
    target_names while the latter follows named_parameters insertion order.

    The fix: _sorted_named_params_ctx wraps named_parameters(recurse=False) to return
    alphabetically sorted results, ensuring the standard branch uses the same order.
    """

    def test_sorted_named_params_ctx_sorts_when_recurse_false(self):
        """_sorted_named_params_ctx should sort named_parameters when recurse=False."""
        import torch
        import torch.nn as nn

        from axolotl.monkeypatch.moe_quant import _sorted_named_params_ctx

        class FakeExperts(nn.Module):
            def __init__(self):
                super().__init__()
                # Register in reverse alphabetical order to detect if sorting is applied
                self.z_proj = nn.Parameter(torch.zeros(2, 3))
                self.a_proj = nn.Parameter(torch.zeros(3, 2))

        module = FakeExperts()

        # Without patch: insertion order (z_proj first, a_proj second)
        names_before = [n for n, _ in module.named_parameters(recurse=False)]
        assert names_before[0] == "z_proj", "Expected insertion order before patch"

        # With patch: alphabetical order (a_proj first, z_proj second)
        with _sorted_named_params_ctx():
            names_during = [n for n, _ in module.named_parameters(recurse=False)]
        assert names_during[0] == "a_proj", "Expected sorted order during patch"

        # After context: restored to insertion order
        names_after = [n for n, _ in module.named_parameters(recurse=False)]
        assert names_after[0] == "z_proj", (
            "Expected insertion order restored after patch"
        )

    def test_sorted_named_params_ctx_does_not_affect_recurse_true(self):
        """_sorted_named_params_ctx should not change behavior for recurse=True."""
        import torch.nn as nn

        from axolotl.monkeypatch.moe_quant import _sorted_named_params_ctx

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = nn.Linear(3, 2)

        module = Parent()

        names_before = [n for n, _ in module.named_parameters(recurse=True)]
        with _sorted_named_params_ctx():
            names_during = [n for n, _ in module.named_parameters(recurse=True)]

        # recurse=True result should be unchanged (same set)
        assert set(names_before) == set(names_during)

    def test_param_wrapper_nesting_order_is_consistent(self):
        """ParamWrapper nesting order must match between parametrized and standard branch.

        Simulates the core bug: two target_parameters on the same module should produce
        the same outer/inner wrapper assignment regardless of named_parameters order.
        """
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora.layer import ParamWrapper
        from peft.tuners.tuners_utils import BaseTuner

        from axolotl.monkeypatch.moe_quant import (
            patch_peft_target_parameters_matching,
        )

        class FakeExperts(nn.Module):
            def __init__(self):
                super().__init__()
                # gate_up_proj registered BEFORE down_proj (typical GLM order)
                self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16))
                self.down_proj = nn.Parameter(torch.zeros(4, 4, 8))

        class FakeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = FakeExperts()

            def forward(self, x):
                return x

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = FakeMLP()

            def forward(self, x):
                return self.mlp(x)

        def build_peft_model(apply_patch: bool):
            model = FakeModel()
            if apply_patch:
                original = BaseTuner._inject_parameters
                try:
                    patch_peft_target_parameters_matching()
                    lora_config = LoraConfig(
                        r=4,
                        lora_alpha=8,
                        target_modules=[],
                        target_parameters=[
                            "mlp.experts.gate_up_proj",
                            "mlp.experts.down_proj",
                        ],
                        lora_dropout=0.0,
                    )
                    peft_model = get_peft_model(model, lora_config)
                finally:
                    BaseTuner._inject_parameters = original
                    patch_peft_target_parameters_matching._axolotl_patched = False
            else:
                lora_config = LoraConfig(
                    r=4,
                    lora_alpha=8,
                    target_modules=[],
                    target_parameters=[
                        "mlp.experts.gate_up_proj",
                        "mlp.experts.down_proj",
                    ],
                    lora_dropout=0.0,
                )
                peft_model = get_peft_model(model, lora_config)
            return peft_model

        # Build with patch (simulates merge path with our fix)
        patched_model = build_peft_model(apply_patch=True)
        experts_module = patched_model.base_model.model.mlp.experts

        # The outer wrapper (at .experts directly) should be gate_up_proj (g > d alphabetically)
        assert isinstance(experts_module, ParamWrapper), (
            "experts should be a ParamWrapper"
        )
        assert experts_module.parameter_name == "gate_up_proj", (
            f"Outer ParamWrapper should wrap gate_up_proj (last alphabetically), "
            f"got: {experts_module.parameter_name}"
        )
        inner = experts_module.base_layer
        assert isinstance(inner, ParamWrapper), (
            "base_layer should also be a ParamWrapper"
        )
        assert inner.parameter_name == "down_proj", (
            f"Inner ParamWrapper should wrap down_proj (first alphabetically), "
            f"got: {inner.parameter_name}"
        )

    def test_adapter_save_load_roundtrip_no_size_mismatch(self, tmp_path):
        """Adapter saved during training (parametrized) must load cleanly during merge (no parametrizations).

        Regression test for the bug zerofata hit: training with quantize_moe_experts uses
        PEFT's parametrized branch (sorted order) while merge uses the standard branch
        (named_parameters insertion order). Without the fix the two branches produce opposite
        ParamWrapper nesting → the outer/inner LoRA weights are swapped → size mismatch on load.
        """
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
        from peft.tuners.tuners_utils import BaseTuner

        from axolotl.monkeypatch.moe_quant import (
            _sorted_named_params_ctx,
            patch_peft_target_parameters_matching,
        )

        # gate_up_proj (shape [4,8,16]) registered BEFORE down_proj (shape [4,4,8])
        # This insertion order is what causes the bug: without sorting the merge path
        # processes gate_up_proj first → inner, down_proj second → outer, which is the
        # opposite of the parametrized training path.
        class FakeExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16))
                self.down_proj = nn.Parameter(torch.zeros(4, 4, 8))

        class FakeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = FakeExperts()

            def forward(self, x):
                return x

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = FakeMLP()

            def forward(self, x):
                return self.mlp(x)

        target_params = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]

        # --- TRAINING PATH (parametrized branch) ---
        # During training with quantize_moe_experts the expert weights are parametrized via
        # bitsandbytes.  PEFT's parametrized branch uses sorted(target_names) internally,
        # so down_proj is processed first (inner) and gate_up_proj second (outer).
        # We simulate this with _sorted_named_params_ctx to get the same ordering without
        # needing actual 4-bit parametrizations.
        train_model = FakeModel()
        train_cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[],
            target_parameters=target_params,
            lora_dropout=0.0,
        )
        with _sorted_named_params_ctx():
            train_peft = get_peft_model(train_model, train_cfg)

        adapter_dir = tmp_path / "adapter"
        train_peft.save_pretrained(str(adapter_dir))

        # Sanity-check saved keys: both outer and inner keys must be present
        from safetensors.torch import load_file

        saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
        outer_lora_a_keys = [
            k for k in saved if "experts.lora_A" in k and "base_layer" not in k
        ]
        inner_lora_a_keys = [k for k in saved if "experts.base_layer.lora_A" in k]
        assert outer_lora_a_keys, "Expected outer experts.lora_A keys in saved adapter"
        assert inner_lora_a_keys, (
            "Expected inner experts.base_layer.lora_A keys in saved adapter"
        )
        # The two params have different sizes; shapes must differ between outer/inner
        outer_shape = next(v for k, v in saved.items() if k in outer_lora_a_keys).shape
        inner_shape = next(v for k, v in saved.items() if k in inner_lora_a_keys).shape
        assert outer_shape != inner_shape, (
            "Outer and inner lora_A should have different shapes (gate_up_proj vs down_proj)"
        )

        # --- MERGE PATH (standard branch, no parametrizations) ---
        # Load the saved adapter into a fresh unparametrized model.
        # Without our fix this raises RuntimeError: size mismatch.
        original_inject = BaseTuner._inject_parameters
        patch_peft_target_parameters_matching()
        try:
            merge_model = FakeModel()
            from peft import PeftModel

            loaded = PeftModel.from_pretrained(merge_model, str(adapter_dir))
            # If we get here without RuntimeError the fix works
            loaded_experts = loaded.base_model.model.mlp.experts
            from peft.tuners.lora.layer import ParamWrapper

            assert isinstance(loaded_experts, ParamWrapper)
            assert loaded_experts.parameter_name == "gate_up_proj", (
                f"Outer ParamWrapper should be gate_up_proj after load, "
                f"got {loaded_experts.parameter_name}"
            )
        finally:
            BaseTuner._inject_parameters = original_inject
            patch_peft_target_parameters_matching._axolotl_patched = False
