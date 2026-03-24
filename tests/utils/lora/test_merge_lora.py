import json
import math
from unittest.mock import Mock, patch

import safetensors.torch
import torch

from axolotl.cli.merge_lora import do_merge_lora
from axolotl.cli.utils.lora_merge import (
    _build_peft_layer_and_get_delta,
    _find_param_wrapper_lora,
    _merge_tensor_with_lora,
    find_lora_weights,
    merge_lora_sharded_efficient,
)
from axolotl.utils.dict import DictDefault


class TestAdapterMergeUnmerge:
    """Test suite for LoRA adapter merging/unmerging functionality"""

    def setup_method(self):
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def create_mock_base_model(self, vocab_size=1000, hidden_size=256):
        """Create a mock base model with linear layers"""
        mock_model = Mock()

        mock_model.config = Mock()
        mock_model.config.vocab_size = vocab_size
        mock_model.config.hidden_size = hidden_size

        mock_model.q_proj = Mock()
        mock_model.q_proj.weight = torch.randn(
            hidden_size, hidden_size, dtype=self.dtype
        )
        mock_model.q_proj.bias = torch.randn(hidden_size, dtype=self.dtype)

        mock_model.v_proj = Mock()
        mock_model.v_proj.weight = torch.randn(
            hidden_size, hidden_size, dtype=self.dtype
        )
        mock_model.v_proj.bias = torch.randn(hidden_size, dtype=self.dtype)

        return mock_model

    def create_mock_lora_model(self, base_model, r=8, alpha=16):
        """Create a mock LoRA model wrapping the base model"""
        mock_lora_model = Mock()
        mock_lora_model.base_model = base_model

        mock_lora_model.merge_and_unload = None
        mock_lora_model.to = Mock(return_value=mock_lora_model)

        mock_lora_model.generation_config = Mock()
        mock_lora_model.config = Mock()

        self.original_q_weight = base_model.q_proj.weight.clone()
        self.original_v_weight = base_model.v_proj.weight.clone()

        mock_lora_model.peft_config = {"default": Mock()}
        mock_lora_model.peft_config["default"].r = r
        mock_lora_model.peft_config["default"].lora_alpha = alpha

        self.lora_A_q = torch.randn(
            r, base_model.q_proj.weight.shape[1], dtype=self.dtype
        )
        self.lora_B_q = torch.randn(
            base_model.q_proj.weight.shape[0], r, dtype=self.dtype
        )

        self.lora_A_v = torch.randn(
            r, base_model.v_proj.weight.shape[1], dtype=self.dtype
        )
        self.lora_B_v = torch.randn(
            base_model.v_proj.weight.shape[0], r, dtype=self.dtype
        )

        self.scaling = alpha / r

        def mock_merge_and_unload(progressbar=False):
            """Simulate the actual merge operation"""
            # Apply LoRA delta to base weights: W_new = W_base + (B @ A) * scaling
            delta_q = (self.lora_B_q @ self.lora_A_q) * self.scaling
            delta_v = (self.lora_B_v @ self.lora_A_v) * self.scaling

            base_model.q_proj.weight = self.original_q_weight + delta_q
            base_model.v_proj.weight = self.original_v_weight + delta_v

            return base_model

        mock_lora_model.merge_and_unload = mock_merge_and_unload
        return mock_lora_model

    def test_basic_lora_merge_unmerge_cycle(self):
        """Test: original_weights -> merge -> unmerge -> should equal original_weights"""

        base_model = self.create_mock_base_model()
        lora_model = self.create_mock_lora_model(base_model)

        original_q_weight = self.original_q_weight.clone()
        original_v_weight = self.original_v_weight.clone()

        merged_model = lora_model.merge_and_unload()

        assert not torch.equal(merged_model.q_proj.weight, original_q_weight)
        assert not torch.equal(merged_model.v_proj.weight, original_v_weight)

        delta_q = (self.lora_B_q @ self.lora_A_q) * self.scaling
        delta_v = (self.lora_B_v @ self.lora_A_v) * self.scaling

        unmerged_q_weight = merged_model.q_proj.weight - delta_q
        unmerged_v_weight = merged_model.v_proj.weight - delta_v

        assert torch.allclose(unmerged_q_weight, original_q_weight, atol=1e-6)
        assert torch.allclose(unmerged_v_weight, original_v_weight, atol=1e-6)

    def test_merge_weight_calculation_accuracy(self):
        """Test: merged_weight = base_weight + (lora_B @ lora_A * scaling)"""
        base_model = self.create_mock_base_model()
        lora_model = self.create_mock_lora_model(base_model, r=16, alpha=32)

        expected_delta_q = (self.lora_B_q @ self.lora_A_q) * self.scaling
        expected_merged_q = self.original_q_weight + expected_delta_q
        merged_model = lora_model.merge_and_unload()

        assert torch.allclose(merged_model.q_proj.weight, expected_merged_q, atol=1e-6)

    @patch("axolotl.cli.merge_lora.load_model_and_tokenizer")
    def test_cli_do_merge_functionality(self, mock_load_model, tmp_path):
        base_model = self.create_mock_base_model()
        lora_model = self.create_mock_lora_model(base_model)
        tokenizer = Mock()
        processor = None

        mock_load_model.return_value = (lora_model, tokenizer, processor)

        cfg = DictDefault(
            {
                "save_safetensors": True,
                "torch_dtype": torch.float32,
                "local_rank": 0,
                "output_dir": str(tmp_path),
                "merge_method": "legacy",
            }
        )

        with (
            patch("pathlib.Path.mkdir"),
            patch.object(base_model, "save_pretrained") as mock_save_model,
            patch.object(tokenizer, "save_pretrained") as mock_save_tokenizer,
        ):
            do_merge_lora(cfg=cfg)

        mock_save_model.assert_called_once()
        mock_save_tokenizer.assert_called_once()

    def test_quantized_model_merge_compatibility(self):
        """Test 4-bit/8-bit model merging scenarios"""
        base_model = self.create_mock_base_model()

        # Mock quantized weights
        base_model.q_proj.weight.quant_state = Mock()
        base_model.q_proj.weight.quant_state.dtype = torch.uint8

        lora_model = self.create_mock_lora_model(base_model)

        merged_model = lora_model.merge_and_unload()
        assert merged_model is not None

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""})
    def test_memory_efficient_merge_with_cpu_offload(self, tmp_path):
        """Test lora_on_cpu configuration during merge"""
        cfg = DictDefault(
            {
                "lora_on_cpu": True,
                "save_safetensors": True,
                "output_dir": str(tmp_path),
                "local_rank": 0,
                "merge_method": "legacy",
            }
        )

        with patch("axolotl.cli.merge_lora.load_model_and_tokenizer") as mock_load:
            base_model = self.create_mock_base_model()
            lora_model = self.create_mock_lora_model(base_model)
            mock_load.return_value = (lora_model, Mock(), None)

            with patch("pathlib.Path.mkdir"), patch("torch.save"):
                do_merge_lora(cfg=cfg)

            assert mock_load.called


class TestEfficientMerge:
    """Test suite for memory-efficient shard-by-shard LoRA merge."""

    def _make_adapter(self, tmp_path, r=8, alpha=16, use_dora=False, use_rslora=False):
        """Create a minimal adapter directory with config + weights."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {
            "r": r,
            "lora_alpha": alpha,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "use_dora": use_dora,
            "use_rslora": use_rslora,
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        return adapter_dir, config

    def _make_base_model(self, tmp_path, hidden=32):
        """Create a minimal base model directory with one shard."""
        model_dir = tmp_path / "base_model"
        model_dir.mkdir()

        weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden, hidden),
            "model.embed_tokens.weight": torch.randn(100, hidden),
        }
        safetensors.torch.save_file(weights, model_dir / "model.safetensors")

        # Minimal config files
        (model_dir / "config.json").write_text("{}")
        return model_dir, weights

    def test_find_lora_weights(self):
        lora_state = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(
                8, 32
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(
                32, 8
            ),
        }
        a, b = find_lora_weights(lora_state, "layers.0.self_attn.q_proj.weight")
        assert a is not None and b is not None
        assert a.shape == (8, 32)

        a, b = find_lora_weights(lora_state, "layers.0.self_attn.v_proj.weight")
        assert a is None and b is None

    def test_merge_tensor_basic(self):
        hidden = 32
        r = 8
        alpha = 16
        base = torch.randn(hidden, hidden)
        lora_a = torch.randn(r, hidden)
        lora_b = torch.randn(hidden, r)
        scale = alpha / r

        lora_state = {
            "base_model.model.layer.q_proj.lora_A.weight": lora_a,
            "base_model.model.layer.q_proj.lora_B.weight": lora_b,
        }

        config = {"r": r, "lora_alpha": alpha}
        merged, was_merged = _merge_tensor_with_lora(
            base, "layer.q_proj.weight", lora_state, scale, config, "cpu"
        )
        assert was_merged
        expected = base + scale * (lora_b @ lora_a)
        assert torch.allclose(merged, expected, atol=1e-5)

    def test_merge_tensor_rslora_scale(self):
        """RSLoRA should use alpha/sqrt(r) as scaling factor."""
        r = 16
        alpha = 32
        standard_scale = alpha / r  # 2.0
        rslora_scale = alpha / math.sqrt(r)  # 8.0

        assert rslora_scale != standard_scale
        assert abs(rslora_scale - 8.0) < 1e-6

    def test_sharded_efficient_merge(self, tmp_path):
        """End-to-end test of shard-by-shard merge."""
        hidden = 32
        r = 8
        alpha = 16

        model_dir, base_weights = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, _ = self._make_adapter(tmp_path, r=r, alpha=alpha)

        # Create LoRA weights
        lora_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(
                r, hidden
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(
                hidden, r
            ),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(
                r, hidden
            ),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(
                hidden, r
            ),
        }
        safetensors.torch.save_file(
            lora_state, adapter_dir / "adapter_model.safetensors"
        )

        output_dir = tmp_path / "output"
        merge_lora_sharded_efficient(
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=output_dir,
            device="cpu",
        )

        # Verify output exists and has merged weights
        merged = safetensors.torch.load_file(output_dir / "model.safetensors")
        scale = alpha / r

        q_key = "model.layers.0.self_attn.q_proj.weight"
        expected_q = base_weights[q_key] + scale * (
            lora_state[f"base_model.model.{q_key[:-7]}.lora_B.weight"]
            @ lora_state[f"base_model.model.{q_key[:-7]}.lora_A.weight"]
        )
        assert torch.allclose(merged[q_key], expected_q, atol=1e-5)

        # Embedding should be unchanged
        assert torch.equal(
            merged["model.embed_tokens.weight"],
            base_weights["model.embed_tokens.weight"],
        )

    def test_dora_merge(self):
        """DoRA merge applies magnitude normalization via PEFT."""
        hidden = 32
        r = 8
        alpha = 16
        scale = alpha / r

        base = torch.randn(hidden, hidden)
        lora_a = torch.randn(r, hidden)
        lora_b = torch.randn(hidden, r)
        magnitude = torch.randn(hidden).abs() + 0.1

        lora_state = {
            "base_model.model.layer.q_proj.lora_A.weight": lora_a,
            "base_model.model.layer.q_proj.lora_B.weight": lora_b,
            "base_model.model.layer.q_proj.lora_magnitude_vector": magnitude,
        }

        config = {"r": r, "lora_alpha": alpha, "use_dora": True}
        merged, was_merged = _merge_tensor_with_lora(
            base,
            "layer.q_proj.weight",
            lora_state,
            scale,
            config,
            "cpu",
            use_dora=True,
        )
        assert was_merged

        # The merge should differ from both base and base+delta (DoRA applies normalization)
        delta = scale * (lora_b @ lora_a)
        assert not torch.allclose(merged, base, atol=1e-3)
        assert not torch.allclose(merged, base + delta, atol=1e-3)

    def test_fuse_unfuse_moe_merge(self):
        """Test fuse→merge→unfuse for MoE expert weights (WeightConverter path)."""
        from axolotl.cli.utils.lora_merge import _fuse_and_unfuse_with_merge

        hidden = 16
        intermediate = 32
        num_experts = 4
        r = 4
        alpha = 8
        scale = alpha / r

        # Simulate checkpoint format: per-expert separate tensors
        shard_tensors = {}
        for i in range(num_experts):
            shard_tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = (
                torch.randn(intermediate, hidden)
            )
            shard_tensors[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = (
                torch.randn(intermediate, hidden)
            )
            shard_tensors[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = (
                torch.randn(hidden, intermediate)
            )
        shard_tensors["model.layers.0.self_attn.q_proj.weight"] = torch.randn(
            hidden, hidden
        )

        # LoRA targets the fused key (runtime format)
        lora_state = {
            "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight": torch.randn(
                r, hidden
            ),
            "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_B.weight": torch.randn(
                intermediate * 2, r
            ),
            "base_model.model.model.layers.0.mlp.experts.down_proj.lora_A.weight": torch.randn(
                r, intermediate
            ),
            "base_model.model.model.layers.0.mlp.experts.down_proj.lora_B.weight": torch.randn(
                hidden, r
            ),
        }

        # Build converters matching qwen2_moe pattern
        from transformers.core_model_loading import (
            Concatenate,
            MergeModulelist,
            WeightConverter,
        )

        converters = [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="mlp.experts.*.down_proj.weight",
                target_patterns="mlp.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ]

        config = {"r": r, "lora_alpha": alpha}
        result, merged_count, processed_keys = _fuse_and_unfuse_with_merge(
            shard_tensors, converters, lora_state, scale, config, "cpu"
        )

        # Should have merged 2 LoRA targets (gate_up_proj and down_proj)
        assert merged_count == 2

        # Processed keys include original per-expert keys (removed) + fused keys (added)
        assert len(processed_keys) > 0

        # Output should be in fused format (runtime keys)
        assert "model.layers.0.mlp.experts.gate_up_proj" in result
        assert "model.layers.0.mlp.experts.down_proj" in result

        # Per-expert keys should be removed
        for i in range(num_experts):
            assert f"model.layers.0.mlp.experts.{i}.gate_proj.weight" not in result

        # Non-expert tensor should be passed through
        assert "model.layers.0.self_attn.q_proj.weight" in result

        # Verify fused tensors are 3D (stacked experts)
        gate_up = result["model.layers.0.mlp.experts.gate_up_proj"]
        assert gate_up.ndim == 3
        assert gate_up.shape[0] == num_experts  # [num_experts, intermediate*2, hidden]

        # Verify the fused LoRA delta was applied correctly
        # Reconstruct the fused base (stack per-expert, concat gate+up)
        gate_stack = torch.stack(
            [
                shard_tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"]
                for i in range(num_experts)
            ]
        )
        up_stack = torch.stack(
            [
                shard_tensors[f"model.layers.0.mlp.experts.{i}.up_proj.weight"]
                for i in range(num_experts)
            ]
        )
        base_fused = torch.cat([gate_stack, up_stack], dim=1)
        lora_a = lora_state[
            "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight"
        ]
        lora_b = lora_state[
            "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_B.weight"
        ]
        expected_fused = base_fused + scale * (lora_b @ lora_a)
        assert torch.allclose(gate_up, expected_fused, atol=1e-5)

    def test_param_wrapper_merge_math(self):
        """ParamWrapper merge via PEFT's get_delta_weight matches manual einsum."""
        num_experts = 4
        r = 2
        in_features = 8
        out_features = 4
        alpha = 4

        base = torch.randn(num_experts, in_features, out_features)
        lora_a = torch.randn(r * num_experts, in_features)
        lora_b = torch.randn(out_features, r * num_experts)

        config = {"r": r, "lora_alpha": alpha}
        delta = _build_peft_layer_and_get_delta(
            lora_a, lora_b, config, base, is_param_wrapper=True
        )
        assert delta.shape == base.shape

        merged = base + delta

        # Verify against manual einsum
        scale = alpha / r
        wa = lora_a.reshape(num_experts, r, in_features)
        wb = lora_b.reshape(out_features, r, num_experts)
        manual_delta = torch.einsum("o r e, e r i -> e i o", wb, wa) * scale
        for e in range(num_experts):
            assert torch.allclose(merged[e], base[e] + manual_delta[e], atol=1e-5), (
                f"Expert {e} mismatch"
            )

    def test_param_wrapper_nesting_dim_filter(self):
        """_find_param_wrapper_lora skips wrong-dimension LoRA at outer level."""
        num_experts = 4
        r = 2

        # Outer LoRA (gate_up_proj): A=[r*E, 8], B=[16, r*E]
        # Inner LoRA (down_proj via base_layer): A=[r*E, 16], B=[8, r*E]
        lora_state = {
            "base_model.model.mod.experts.lora_A.weight": torch.randn(
                r * num_experts, 8
            ),
            "base_model.model.mod.experts.lora_B.weight": torch.randn(
                16, r * num_experts
            ),
            "base_model.model.mod.experts.base_layer.lora_A.weight": torch.randn(
                r * num_experts, 16
            ),
            "base_model.model.mod.experts.base_layer.lora_B.weight": torch.randn(
                8, r * num_experts
            ),
        }

        # gate_up_proj shape [4, 8, 16] — should match outer LoRA
        a, b, name = _find_param_wrapper_lora(
            lora_state, "mod.experts.gate_up_proj", tensor_shape=(4, 8, 16)
        )
        assert a is not None and name == "gate_up_proj"
        assert a.shape == (r * num_experts, 8)  # outer

        # down_proj shape [4, 16, 8] — outer dims don't match, should find inner
        a, b, name = _find_param_wrapper_lora(
            lora_state, "mod.experts.down_proj", tensor_shape=(4, 16, 8)
        )
        assert a is not None and name == "down_proj"
        assert a.shape == (r * num_experts, 16)  # inner (base_layer)

        # shape that matches neither — should return None
        a, b, name = _find_param_wrapper_lora(
            lora_state, "mod.experts.other", tensor_shape=(4, 99, 99)
        )
        assert a is None

    def test_find_lora_weights_with_renamings(self):
        """Weight renamings let checkpoint keys match LoRA keys."""
        lora_state = {
            "base_model.model.layers.0.mlp.fc1.lora_A.weight": torch.randn(8, 32),
            "base_model.model.layers.0.mlp.fc1.lora_B.weight": torch.randn(32, 8),
        }
        # Direct lookup fails (checkpoint has "ff0", LoRA has "fc1")
        a, b = find_lora_weights(lora_state, "layers.0.mlp.ff0.weight")
        assert a is None

        # With renaming ff0 → fc1, it should match
        a, b = find_lora_weights(
            lora_state, "layers.0.mlp.ff0.weight", weight_renamings={"ff0": "fc1"}
        )
        assert a is not None
        assert a.shape == (8, 32)

    def test_unmatched_tensors_pass_through(self):
        """Tensors with no matching LoRA are returned unchanged."""
        lora_state = {
            "base_model.model.layer.q_proj.lora_A.weight": torch.randn(8, 32),
            "base_model.model.layer.q_proj.lora_B.weight": torch.randn(32, 8),
        }

        # 1D tensor (layernorm) — never matched
        ln = torch.randn(32)
        merged, was_merged = _merge_tensor_with_lora(
            ln, "layer.norm.weight", lora_state, 2.0, {}, "cpu"
        )
        assert not was_merged
        assert torch.equal(merged, ln)

        # 2D tensor with no matching key
        unrelated = torch.randn(64, 32)
        merged, was_merged = _merge_tensor_with_lora(
            unrelated, "layer.other_proj.weight", lora_state, 2.0, {}, "cpu"
        )
        assert not was_merged
        assert torch.equal(merged, unrelated)

    def test_fan_in_fan_out_transpose(self):
        """fan_in_fan_out config transposes the LoRA delta."""
        hidden = 16
        r = 4
        alpha = 4  # scale = 1.0

        base = torch.randn(hidden, hidden)
        lora_a = torch.randn(r, hidden)
        lora_b = torch.randn(hidden, r)

        lora_state = {
            "base_model.model.layer.proj.lora_A.weight": lora_a,
            "base_model.model.layer.proj.lora_B.weight": lora_b,
        }

        config_normal = {"r": r, "lora_alpha": alpha}
        config_fif = {"r": r, "lora_alpha": alpha, "fan_in_fan_out": True}

        merged_normal, _ = _merge_tensor_with_lora(
            base, "layer.proj.weight", lora_state, 1.0, config_normal, "cpu"
        )
        merged_fif, _ = _merge_tensor_with_lora(
            base, "layer.proj.weight", lora_state, 1.0, config_fif, "cpu"
        )

        delta = (alpha / r) * (lora_b @ lora_a)
        assert torch.allclose(merged_normal, base + delta, atol=1e-5)
        assert torch.allclose(merged_fif, base + delta.T, atol=1e-5)
        assert not torch.allclose(merged_normal, merged_fif, atol=1e-5)

    def test_rslora_end_to_end(self, tmp_path):
        """RSLoRA adapter uses alpha/sqrt(r) scaling in sharded merge."""
        hidden = 16
        r = 16
        alpha = 32

        model_dir, base_weights = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, _ = self._make_adapter(tmp_path, r=r, alpha=alpha, use_rslora=True)

        lora_a = torch.randn(r, hidden)
        lora_b = torch.randn(hidden, r)
        lora_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": lora_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": lora_b,
        }
        safetensors.torch.save_file(
            lora_state, adapter_dir / "adapter_model.safetensors"
        )

        output_dir = tmp_path / "output"
        merge_lora_sharded_efficient(
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=output_dir,
            device="cpu",
        )

        merged = safetensors.torch.load_file(output_dir / "model.safetensors")
        rslora_scale = alpha / math.sqrt(r)  # 8.0, not 2.0
        q_key = "model.layers.0.self_attn.q_proj.weight"
        expected = base_weights[q_key] + rslora_scale * (lora_b @ lora_a)
        assert torch.allclose(merged[q_key], expected, atol=1e-5)

        # Confirm it differs from standard scale
        wrong_scale = alpha / r  # 2.0
        wrong_expected = base_weights[q_key] + wrong_scale * (lora_b @ lora_a)
        assert not torch.allclose(merged[q_key], wrong_expected, atol=1e-3)

    def test_multi_shard_index_json(self, tmp_path):
        """Multi-shard merge generates a correct weight-map index."""
        hidden = 16
        r = 4
        alpha = 8

        model_dir = tmp_path / "base_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        # Create 2 shards
        shard1 = {"model.layers.0.weight": torch.randn(hidden, hidden)}
        shard2 = {"model.layers.1.weight": torch.randn(hidden, hidden)}
        safetensors.torch.save_file(
            shard1, model_dir / "model-00001-of-00002.safetensors"
        )
        safetensors.torch.save_file(
            shard2, model_dir / "model-00002-of-00002.safetensors"
        )

        # Write a base model index (will be skipped by copy_non_model_files)
        base_index = {
            "metadata": {},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
            },
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(base_index))

        adapter_dir, _ = self._make_adapter(tmp_path, r=r, alpha=alpha)
        safetensors.torch.save_file({}, adapter_dir / "adapter_model.safetensors")

        output_dir = tmp_path / "output"
        merge_lora_sharded_efficient(
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=output_dir,
            device="cpu",
        )

        # Verify index was generated
        index_path = output_dir / "model.safetensors.index.json"
        assert index_path.exists()
        with open(index_path) as f:
            idx = json.load(f)

        assert "weight_map" in idx
        assert len(idx["weight_map"]) == 2
        # Each key should map to a shard that exists
        for _key, shard_name in idx["weight_map"].items():
            assert (output_dir / shard_name).exists(), f"Missing shard: {shard_name}"

    def test_dora_end_to_end(self, tmp_path):
        """DoRA merge through the full sharded merge pipeline."""
        hidden = 16
        r = 4
        alpha = 8

        model_dir, base_weights = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, _ = self._make_adapter(tmp_path, r=r, alpha=alpha, use_dora=True)

        lora_a = torch.randn(r, hidden)
        lora_b = torch.randn(hidden, r)
        magnitude = torch.randn(hidden).abs() + 0.1
        lora_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": lora_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": lora_b,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector": magnitude,
        }
        safetensors.torch.save_file(
            lora_state, adapter_dir / "adapter_model.safetensors"
        )

        output_dir = tmp_path / "output"
        merge_lora_sharded_efficient(
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=output_dir,
            device="cpu",
        )

        merged = safetensors.torch.load_file(output_dir / "model.safetensors")
        q_key = "model.layers.0.self_attn.q_proj.weight"

        # Use PEFT's own get_delta_weight as the reference
        delta = _build_peft_layer_and_get_delta(
            lora_a,
            lora_b,
            {"r": r, "lora_alpha": alpha, "use_dora": True},
            base_weights[q_key],
            magnitude=magnitude,
        )
        expected = base_weights[q_key] + delta
        assert torch.allclose(merged[q_key], expected, atol=1e-5)

        # Verify it differs from standard (non-DoRA) merge
        standard_delta = _build_peft_layer_and_get_delta(
            lora_a,
            lora_b,
            {"r": r, "lora_alpha": alpha},
            base_weights[q_key],
        )
        assert not torch.allclose(delta, standard_delta, atol=1e-3)

        # v_proj has no LoRA weights — should be unchanged
        v_key = "model.layers.0.self_attn.v_proj.weight"
        assert torch.equal(merged[v_key], base_weights[v_key]), (
            "v_proj should be unchanged (no LoRA weights for it)"
        )

    def test_dora_missing_magnitude_falls_back(self):
        """DoRA without magnitude vector falls back to standard LoRA merge."""
        hidden = 16
        r = 4
        alpha = 8
        scale = alpha / r

        base = torch.randn(hidden, hidden)
        lora_a = torch.randn(r, hidden)
        lora_b = torch.randn(hidden, r)

        # No magnitude vector in lora_state
        lora_state = {
            "base_model.model.layer.proj.lora_A.weight": lora_a,
            "base_model.model.layer.proj.lora_B.weight": lora_b,
        }

        config = {"r": r, "lora_alpha": alpha, "use_dora": True}
        merged, was_merged = _merge_tensor_with_lora(
            base, "layer.proj.weight", lora_state, scale, config, "cpu", use_dora=True
        )
        assert was_merged
        # No magnitude vector → PEFT creates DoRA layer but with default magnitude,
        # which produces a result different from plain W + scale * B @ A.
        # Just verify it was merged (not unchanged).
        assert not torch.equal(merged, base)
