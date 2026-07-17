import json
import math
from unittest.mock import Mock, patch

import pytest
import safetensors.torch
import torch

from axolotl.cli.merge_lora import do_merge_lora
from axolotl.cli.utils.lora_merge import (
    _build_peft_layer_and_get_delta,
    _find_param_wrapper_lora,
    _merge_tensor_with_lora,
    _resolve_lora_alpha_for_key,
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

    def test_resolve_alpha_for_key_returns_none_without_pattern(self):
        assert (
            _resolve_lora_alpha_for_key("model.layers.0.self_attn.q_proj.weight", {})
            is None
        )
        assert (
            _resolve_lora_alpha_for_key(
                "model.layers.0.self_attn.q_proj.weight",
                {"alpha_pattern": {}},
            )
            is None
        )

    def test_resolve_alpha_for_key_matches_peft_suffix_semantics(self):
        cfg = {"alpha_pattern": {"layers.0.self_attn.q_proj": 64}}
        assert (
            _resolve_lora_alpha_for_key("model.layers.0.self_attn.q_proj.weight", cfg)
            == 64
        )
        # No match falls back to None so the caller keeps the global alpha.
        assert (
            _resolve_lora_alpha_for_key("model.layers.0.self_attn.v_proj.weight", cfg)
            is None
        )

    def test_resolve_alpha_for_key_follows_weight_renamings(self):
        # Pattern keyed against the runtime name; merge sees the checkpoint name.
        cfg = {"alpha_pattern": {"model.new.layers.0.q_proj": 64}}
        renamings = {r"^model\.old\.": "model.new."}
        assert (
            _resolve_lora_alpha_for_key(
                "model.old.layers.0.q_proj.weight", cfg, renamings
            )
            == 64
        )
        # Without the renamings, the same lookup misses and falls back to global.
        assert (
            _resolve_lora_alpha_for_key("model.old.layers.0.q_proj.weight", cfg) is None
        )

    def test_pattern_no_longer_rejected_as_adalora(self, tmp_path):
        """Memory-efficient merge accepts rank_pattern/alpha_pattern (plain LoRA, not AdaLoRA)."""
        hidden = 32
        r = 8
        alpha = 16

        model_dir, _ = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, config = self._make_adapter(tmp_path, r=r, alpha=alpha)
        config["rank_pattern"] = {"layers.0.self_attn.q_proj": r}
        config["alpha_pattern"] = {"layers.0.self_attn.q_proj": alpha}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

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

        merge_lora_sharded_efficient(
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=tmp_path / "output",
            device="cpu",
        )

    def test_merge_applies_alpha_pattern_per_module(self, tmp_path):
        """End-to-end: q_proj uses alpha_pattern=64, v_proj uses global lora_alpha=16."""
        hidden = 32
        r = 8
        global_alpha = 16
        q_alpha = 64

        model_dir, base_weights = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, config = self._make_adapter(tmp_path, r=r, alpha=global_alpha)
        config["alpha_pattern"] = {"layers.0.self_attn.q_proj": q_alpha}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        q_a = torch.randn(r, hidden)
        q_b = torch.randn(hidden, r)
        v_a = torch.randn(r, hidden)
        v_b = torch.randn(hidden, r)
        lora_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
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
        v_key = "model.layers.0.self_attn.v_proj.weight"
        expected_q = base_weights[q_key] + (q_alpha / r) * (q_b @ q_a)
        expected_v = base_weights[v_key] + (global_alpha / r) * (v_b @ v_a)
        assert torch.allclose(merged[q_key], expected_q, atol=1e-5)
        assert torch.allclose(merged[v_key], expected_v, atol=1e-5)

    def test_merge_applies_rank_pattern_per_module(self, tmp_path):
        """End-to-end: q_proj uses rank_pattern=16, v_proj uses global r=8."""
        hidden = 32
        global_r = 8
        q_r = 16
        alpha = 16

        model_dir, base_weights = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, config = self._make_adapter(tmp_path, r=global_r, alpha=alpha)
        config["rank_pattern"] = {"layers.0.self_attn.q_proj": q_r}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        q_a = torch.randn(q_r, hidden)
        q_b = torch.randn(hidden, q_r)
        v_a = torch.randn(global_r, hidden)
        v_b = torch.randn(hidden, global_r)
        lora_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
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
        v_key = "model.layers.0.self_attn.v_proj.weight"
        expected_q = base_weights[q_key] + (alpha / q_r) * (q_b @ q_a)
        expected_v = base_weights[v_key] + (alpha / global_r) * (v_b @ v_a)
        assert torch.allclose(merged[q_key], expected_q, atol=1e-5)
        assert torch.allclose(merged[v_key], expected_v, atol=1e-5)

    def test_merge_applies_rank_and_alpha_pattern_combined(self, tmp_path):
        """End-to-end: q_proj overrides both rank (16) and alpha (64), v_proj uses globals."""
        hidden = 32
        global_r = 8
        global_alpha = 16
        q_r = 16
        q_alpha = 64

        model_dir, base_weights = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, config = self._make_adapter(
            tmp_path, r=global_r, alpha=global_alpha
        )
        config["rank_pattern"] = {"layers.0.self_attn.q_proj": q_r}
        config["alpha_pattern"] = {"layers.0.self_attn.q_proj": q_alpha}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        q_a = torch.randn(q_r, hidden)
        q_b = torch.randn(hidden, q_r)
        v_a = torch.randn(global_r, hidden)
        v_b = torch.randn(hidden, global_r)
        lora_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
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
        v_key = "model.layers.0.self_attn.v_proj.weight"
        expected_q = base_weights[q_key] + (q_alpha / q_r) * (q_b @ q_a)
        expected_v = base_weights[v_key] + (global_alpha / global_r) * (v_b @ v_a)
        assert torch.allclose(merged[q_key], expected_q, atol=1e-5)
        assert torch.allclose(merged[v_key], expected_v, atol=1e-5)

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

    def test_fuse_skipped_for_incomplete_expert_shard(self):
        """Expert lists split across shard boundaries must not be fused from a partial shard."""
        from transformers.core_model_loading import (
            Concatenate,
            MergeModulelist,
            WeightConverter,
        )

        from axolotl.cli.utils.lora_merge import _fuse_and_unfuse_with_merge

        hidden = 16
        intermediate = 32
        num_experts = 8

        # Layer 0: gate/up expert counts mismatch within the shard (previously
        # crashed in torch.cat). Layer 1: contiguous but partial expert list
        # (previously silently fused a subset).
        shard_tensors = {}
        for i in range(5):
            shard_tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = (
                torch.randn(intermediate, hidden)
            )
        for i in range(4):
            shard_tensors[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = (
                torch.randn(intermediate, hidden)
            )
        for i in range(4):
            shard_tensors[f"model.layers.1.mlp.experts.{i}.gate_proj.weight"] = (
                torch.randn(intermediate, hidden)
            )
            shard_tensors[f"model.layers.1.mlp.experts.{i}.up_proj.weight"] = (
                torch.randn(intermediate, hidden)
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
        ]

        config = {"r": 4, "lora_alpha": 8}
        result, merged_count, processed_keys = _fuse_and_unfuse_with_merge(
            shard_tensors,
            converters,
            {},
            2.0,
            config,
            "cpu",
            expected_num_experts=num_experts,
        )

        assert merged_count == 0
        assert not processed_keys
        # No fused keys; every per-expert tensor passes through unchanged
        assert "model.layers.0.mlp.experts.gate_up_proj" not in result
        assert "model.layers.1.mlp.experts.gate_up_proj" not in result
        assert set(result.keys()) == set(shard_tensors.keys())

        # Complete layer but still-quantized tensors (nvfp4 uint8 qdata with
        # weight_scale siblings): fusing raw qdata would orphan the scales.
        quant_tensors = {}
        for i in range(num_experts):
            for proj in ("gate_proj", "up_proj"):
                key = f"model.layers.2.mlp.experts.{i}.{proj}.weight"
                quant_tensors[key] = torch.randint(
                    0, 255, (intermediate, hidden // 2), dtype=torch.uint8
                )
                quant_tensors[key + "_scale"] = torch.randn(
                    intermediate, hidden // 16
                ).to(torch.float8_e4m3fn)
        result, merged_count, processed_keys = _fuse_and_unfuse_with_merge(
            quant_tensors,
            converters,
            {},
            2.0,
            config,
            "cpu",
            expected_num_experts=num_experts,
        )
        assert merged_count == 0
        assert not processed_keys
        assert "model.layers.2.mlp.experts.gate_up_proj" not in result
        assert set(result.keys()) == set(quant_tensors.keys())

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

    def test_dora_merge_honors_alpha_pattern(self, tmp_path):
        """DoRA + alpha_pattern: q_proj uses overridden alpha, v_proj uses global."""
        hidden = 16
        r = 4
        global_alpha = 8
        q_alpha = 32

        model_dir, base_weights = self._make_base_model(tmp_path, hidden=hidden)
        adapter_dir, config = self._make_adapter(
            tmp_path, r=r, alpha=global_alpha, use_dora=True
        )
        config["alpha_pattern"] = {"layers.0.self_attn.q_proj": q_alpha}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        q_a = torch.randn(r, hidden)
        q_b = torch.randn(hidden, r)
        q_mag = torch.randn(hidden).abs() + 0.1
        v_a = torch.randn(r, hidden)
        v_b = torch.randn(hidden, r)
        v_mag = torch.randn(hidden).abs() + 0.1
        lora_state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector": q_mag,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
            "base_model.model.model.layers.0.self_attn.v_proj.lora_magnitude_vector": v_mag,
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
        v_key = "model.layers.0.self_attn.v_proj.weight"
        expected_q_delta = _build_peft_layer_and_get_delta(
            q_a,
            q_b,
            {"r": r, "lora_alpha": global_alpha, "use_dora": True},
            base_weights[q_key],
            magnitude=q_mag,
            lora_alpha_override=q_alpha,
        )
        expected_v_delta = _build_peft_layer_and_get_delta(
            v_a,
            v_b,
            {"r": r, "lora_alpha": global_alpha, "use_dora": True},
            base_weights[v_key],
            magnitude=v_mag,
        )
        assert torch.allclose(
            merged[q_key], base_weights[q_key] + expected_q_delta, atol=1e-5
        )
        assert torch.allclose(
            merged[v_key], base_weights[v_key] + expected_v_delta, atol=1e-5
        )
        # Sanity: q_proj delta must differ from a non-overridden alpha computation.
        wrong_q_delta = _build_peft_layer_and_get_delta(
            q_a,
            q_b,
            {"r": r, "lora_alpha": global_alpha, "use_dora": True},
            base_weights[q_key],
            magnitude=q_mag,
        )
        assert not torch.allclose(expected_q_delta, wrong_q_delta, atol=1e-3)

    def test_dora_missing_magnitude_raises(self):
        """DoRA with missing magnitude vector raises an explicit error."""
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
        with pytest.raises(ValueError, match="DoRA merge requires a magnitude vector"):
            _merge_tensor_with_lora(
                base,
                "layer.proj.weight",
                lora_state,
                scale,
                config,
                "cpu",
                use_dora=True,
            )


class TestQuantizedBaseMerge:
    """Per-(dtype x module-type) merge correctness for quantized base weights.

    The efficient merge folds ``scaling*(B@A)`` into the base weight; for a quantized base it must do
    so on the DEQUANTIZED value and keep bf16 (re-rounding the sum to the quant format drops the low-
    magnitude LoRA delta). These are hermetic (synthetic tensors, CPU) regression guards.
    """

    E4M3_MAX = 448.0

    @staticmethod
    def _make_block_fp8(w: torch.Tensor, block: int):
        """bf16 weight -> (float8_e4m3fn weight, fp32 block scale_inv). Block axes = last two dims."""
        *lead, N, K = w.shape
        sr, sc = N // block, K // block
        wb = w.float().reshape(*lead, sr, block, sc, block)
        amax = (
            wb.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(1e-12)
        )  # per (sr,sc) block
        scale = amax / TestQuantizedBaseMerge.E4M3_MAX
        q = torch.clamp(
            wb / scale,
            -TestQuantizedBaseMerge.E4M3_MAX,
            TestQuantizedBaseMerge.E4M3_MAX,
        )
        q = q.reshape(*lead, N, K).to(torch.float8_e4m3fn)
        scale_inv = scale.reshape(*lead, sr, sc).to(torch.float32)
        return q, scale_inv

    @staticmethod
    def _dequant(q: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
        *lead, N, K = q.shape
        sr, sc = scale_inv.shape[-2], scale_inv.shape[-1]
        bn, bk = N // sr, K // sc
        wf = q.float().reshape(*lead, sr, bn, sc, bk)
        s = scale_inv.float().reshape(*lead, sr, 1, sc, 1)
        return (wf * s).reshape(*lead, N, K)

    @pytest.mark.parametrize("shape,block", [((16, 16), 8), ((2, 16, 16), 8)])
    def test_dequantize_block_fp8_shard(self, shape, block):
        """_dequantize_block_fp8_shard reproduces the block dequant + drops scale_inv (2D + 3D)."""
        from axolotl.cli.utils.lora_merge import _dequantize_quantized_shard

        torch.manual_seed(0)
        w = torch.randn(*shape, dtype=torch.bfloat16) * 0.2
        key = "m.experts.gate_up_proj" if len(shape) == 3 else "m.q_proj.weight"
        q, si = self._make_block_fp8(w, block)
        shard = {key: q, key + "_scale_inv": si, "m.norm.weight": torch.ones(4)}

        out, _, _, _ = _dequantize_quantized_shard(shard, "cpu")

        assert key + "_scale_inv" not in out  # scale dropped
        assert out[key].dtype == torch.bfloat16
        assert "m.norm.weight" in out  # unrelated tensor untouched
        ref = self._dequant(q, si)
        assert torch.allclose(out[key].float(), ref, atol=2e-2)

    def test_block_fp8_untouched_without_scale_inv(self):
        """A float8 weight with no *_scale_inv sibling is left as-is (can't dequant)."""
        from axolotl.cli.utils.lora_merge import _dequantize_quantized_shard

        q = torch.randn(8, 8).to(torch.float8_e4m3fn)
        out, _, _, _ = _dequantize_quantized_shard({"x.weight": q}, "cpu")
        assert out["x.weight"].dtype == torch.float8_e4m3fn

    def test_merge_block_fp8_linear_folds_into_dequantized(self):
        """End-to-end 2D: dequant shard then merge == dequant(base) + scaling*(B@A), output bf16."""
        from axolotl.cli.utils.lora_merge import (
            _dequantize_quantized_shard,
            _merge_tensor_with_lora,
        )

        torch.manual_seed(1)
        hidden, r, alpha, block = 32, 8, 16, 16
        scale = alpha / r
        w = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2
        q, si = self._make_block_fp8(w, block)
        key = "model.layers.0.self_attn.q_proj.weight"
        lora_a = torch.randn(r, hidden) * 0.1
        lora_b = torch.randn(hidden, r) * 0.1
        lora_state = {
            f"base_model.model.{key[: -len('.weight')]}.lora_A.weight": lora_a,
            f"base_model.model.{key[: -len('.weight')]}.lora_B.weight": lora_b,
        }

        deq_shard, _, _, _ = _dequantize_quantized_shard(
            {key: q, key + "_scale_inv": si}, "cpu"
        )
        merged, was_merged = _merge_tensor_with_lora(
            deq_shard[key], key, lora_state, scale, {"r": r, "lora_alpha": alpha}, "cpu"
        )

        assert was_merged
        assert merged.dtype == torch.bfloat16  # NOT re-rounded to fp8
        expected = self._dequant(q, si) + scale * (lora_b @ lora_a)
        rel = (merged.float() - expected).norm() / expected.norm()
        assert rel < 5e-3, f"block-fp8 merge rel {rel:.2e}"

    def test_raw_fp8_merge_is_wrong_regression(self):
        """Guard the fix: folding the delta into the RAW fp8 (skipping dequant) is materially wrong,
        so the dequant step must stay. Same synthetic case, merged without _dequantize_block_fp8_shard."""
        from axolotl.cli.utils.lora_merge import _merge_tensor_with_lora

        torch.manual_seed(1)
        hidden, r, alpha, block = 32, 8, 16, 16
        scale = alpha / r
        w = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2
        q, si = self._make_block_fp8(w, block)
        key = "model.layers.0.self_attn.q_proj.weight"
        lora_a, lora_b = torch.randn(r, hidden) * 0.1, torch.randn(hidden, r) * 0.1
        lora_state = {
            f"base_model.model.{key[: -len('.weight')]}.lora_A.weight": lora_a,
            f"base_model.model.{key[: -len('.weight')]}.lora_B.weight": lora_b,
        }
        # merge on the raw fp8 tensor (the old, broken behaviour)
        merged, _ = _merge_tensor_with_lora(
            q, key, lora_state, scale, {"r": r, "lora_alpha": alpha}, "cpu"
        )
        expected = self._dequant(q, si) + scale * (lora_b @ lora_a)
        rel = (merged.float() - expected).norm() / expected.norm()
        assert rel > 0.1, (
            "raw-fp8 merge unexpectedly close — the dequant guard may be untested"
        )

    def test_strip_quantization_config(self, tmp_path):
        from axolotl.cli.utils.lora_merge import _strip_quantization_config

        cfg = {
            "model_type": "mistral_large4",
            "torch_dtype": "float8_e4m3fn",
            "quantization_config": {"quant_method": "fp8"},
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        _strip_quantization_config(tmp_path)
        out = json.loads((tmp_path / "config.json").read_text())
        assert "quantization_config" not in out
        assert out["torch_dtype"] == "bfloat16"

    def test_block_fp8_merge_forward_equivalence_end_to_end(self, tmp_path):
        """FORWARD-level, full disk round-trip: build a block-fp8 base + perturbed LoRA, run the real
        merge_lora_sharded_efficient, reload, and compare OUTPUTS of unmerged (dequant-base + LoRA
        path) vs merged (plain bf16 matmul). Catches config-strip / scale-drop / dtype / re-quant-on-
        load issues the weight-level oracle can't."""
        torch.manual_seed(3)
        hidden, r, alpha, block = 32, 8, 16, 16
        scale = alpha / r

        w = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2
        q, si = self._make_block_fp8(w, block)
        deq_base = self._dequant(q, si)  # what the model actually computes with

        model_dir = tmp_path / "base"
        model_dir.mkdir()
        key = "model.layers.0.self_attn.q_proj.weight"
        safetensors.torch.save_file(
            {key: q, key + "_scale_inv": si}, model_dir / "model.safetensors"
        )
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "torch_dtype": "float8_e4m3fn",
                    "quantization_config": {"quant_method": "fp8"},
                }
            )
        )

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        lora_a = torch.randn(r, hidden) * 0.1
        lora_b = (
            torch.randn(hidden, r) * 0.1
        )  # perturbed (NOT zero-init) -> real effect
        safetensors.torch.save_file(
            {
                f"base_model.model.{key[: -len('.weight')]}.lora_A.weight": lora_a,
                f"base_model.model.{key[: -len('.weight')]}.lora_B.weight": lora_b,
            },
            adapter_dir / "adapter_model.safetensors",
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": r, "lora_alpha": alpha, "peft_type": "LORA"})
        )

        out_dir = tmp_path / "merged"
        merge_lora_sharded_efficient(
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=out_dir,
            device="cpu",
            dequant=True,  # explicit --dequant: bf16 output
        )

        # --dequant: merged config is de-quantized
        merged_cfg = json.loads((out_dir / "config.json").read_text())
        assert "quantization_config" not in merged_cfg

        merged = {}
        with safetensors.torch.safe_open(
            out_dir / "model.safetensors", framework="pt"
        ) as f:
            for k in f.keys():
                merged[k] = f.get_tensor(k)
        assert key + "_scale_inv" not in merged
        assert merged[key].dtype == torch.bfloat16

        x = torch.randn(4, hidden)
        # unmerged forward: dequant-base linear + LoRA branch (the module's actual computation)
        y_unmerged = x @ deq_base.float().T + scale * (x @ lora_a.T) @ lora_b.T
        y_merged = x @ merged[key].float().T
        rel = (y_unmerged - y_merged).norm() / y_unmerged.norm()
        assert rel < 5e-3, f"forward mismatch after block-fp8 merge: rel {rel:.2e}"

    def test_block_fp8_merge_preserves_format_default(self, tmp_path):
        """DEFAULT merge is FORMAT-PRESERVING: a block-fp8 base stays block-fp8 (fp8 weight +
        weight_scale_inv), quantization_config is kept, and the forward still matches within fp8 tol.
        Non-LoRA quantized weights pass through byte-identical."""
        torch.manual_seed(4)
        hidden, r, alpha, block = 32, 8, 16, 16
        scale = alpha / r
        w = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2
        q, si = self._make_block_fp8(w, block)
        deq_base = self._dequant(q, si)
        # a second block-fp8 weight WITHOUT LoRA -> must pass through untouched
        q2, si2 = self._make_block_fp8(
            torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2, block
        )

        model_dir = tmp_path / "base"
        model_dir.mkdir()
        key = "model.layers.0.self_attn.q_proj.weight"
        okey = "model.layers.0.self_attn.o_proj.weight"
        safetensors.torch.save_file(
            {
                key: q,
                key + "_scale_inv": si,
                okey: q2,
                okey + "_scale_inv": si2,
            },
            model_dir / "model.safetensors",
        )
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "torch_dtype": "float8_e4m3fn",
                    "quantization_config": {"quant_method": "fp8"},
                }
            )
        )
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        lora_a = torch.randn(r, hidden) * 0.1
        lora_b = torch.randn(hidden, r) * 0.1
        safetensors.torch.save_file(
            {
                f"base_model.model.{key[: -len('.weight')]}.lora_A.weight": lora_a,
                f"base_model.model.{key[: -len('.weight')]}.lora_B.weight": lora_b,
            },
            adapter_dir / "adapter_model.safetensors",
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": r, "lora_alpha": alpha, "peft_type": "LORA"})
        )

        out_dir = tmp_path / "merged"
        merge_lora_sharded_efficient(  # default: NO dequant flag -> format-preserving
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=out_dir,
            device="cpu",
        )
        merged = {}
        with safetensors.torch.safe_open(
            out_dir / "model.safetensors", framework="pt"
        ) as f:
            for k in f.keys():
                merged[k] = f.get_tensor(k)

        # format preserved: fp8 weight + fp32 scale kept, config NOT stripped
        assert merged[key].dtype == torch.float8_e4m3fn
        assert merged[key + "_scale_inv"].dtype == torch.float32
        assert "quantization_config" in json.loads(
            (out_dir / "config.json").read_text()
        )
        # no-LoRA weight passes through byte-identical
        assert torch.equal(merged[okey], q2) and torch.equal(
            merged[okey + "_scale_inv"], si2
        )

        # forward matches within fp8 tolerance (the delta survives re-quantization with fresh scales)
        def deq(qq, s):
            O, K = qq.shape
            sr, sc = s.shape
            return (
                qq.float().reshape(sr, O // sr, sc, K // sc) * s.reshape(sr, 1, sc, 1)
            ).reshape(O, K)

        x = torch.randn(4, hidden)
        y_unmerged = x @ deq_base.float().T + scale * (x @ lora_a.T) @ lora_b.T
        y_merged = x @ deq(merged[key], merged[key + "_scale_inv"]).T
        rel = (y_unmerged - y_merged).norm() / y_unmerged.norm()
        assert rel < 6e-2, f"format-preserving merge forward rel {rel:.2e}"

    @staticmethod
    def _make_mxfp8(w: torch.Tensor, block: int = 32):
        """bf16 -> (e4m3 weight, uint8 e8m0 scale) with FLOOR e8m0 (block along last dim)."""
        *lead, N, K = w.shape
        nb = K // block
        wb = w.float().reshape(*lead, N, nb, block)
        amax = wb.abs().amax(dim=-1).clamp_min(1e-12)
        exp = torch.floor(torch.log2(amax)) - 8.0
        q = torch.clamp(wb / torch.exp2(exp)[..., None], -448.0, 448.0).reshape(
            *lead, N, K
        )
        return q.to(torch.float8_e4m3fn), (exp + 127.0).clamp(0, 254).to(torch.uint8)

    def test_dequantize_mxfp8_shard_and_merge(self):
        """mxfp8 (e4m3 + e8m0/32) fused dequant + 2D merge folds into the dequantized weight."""
        from axolotl.cli.utils.lora_merge import (
            _dequantize_quantized_shard,
            _merge_tensor_with_lora,
        )

        torch.manual_seed(2)
        hidden, r, alpha = 64, 8, 16
        scale = alpha / r
        w = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2
        q, s = self._make_mxfp8(w, 32)
        key = "model.layers.0.self_attn.q_proj.weight"
        deq_shard, did, _, _ = _dequantize_quantized_shard(
            {key: q, key + "_scale": s}, "cpu"
        )
        assert did
        assert (
            key + "_scale" not in deq_shard and deq_shard[key].dtype == torch.bfloat16
        )

        # dequant reference (block-32 e8m0)
        ref_deq = (
            q.float().reshape(hidden, hidden // 32, 32)
            * torch.exp2(s.float() - 127.0)[..., None]
        ).reshape(hidden, hidden)
        assert torch.allclose(deq_shard[key].float(), ref_deq, atol=2e-2)

        lora_a, lora_b = torch.randn(r, hidden) * 0.1, torch.randn(hidden, r) * 0.1
        lora_state = {
            f"base_model.model.{key[: -len('.weight')]}.lora_A.weight": lora_a,
            f"base_model.model.{key[: -len('.weight')]}.lora_B.weight": lora_b,
        }
        merged, was = _merge_tensor_with_lora(
            deq_shard[key], key, lora_state, scale, {"r": r, "lora_alpha": alpha}, "cpu"
        )
        assert was and merged.dtype == torch.bfloat16
        expected = ref_deq + scale * (lora_b @ lora_a)
        assert (merged.float() - expected).norm() / expected.norm() < 5e-3

    def test_dequantize_nvfp4_shard(self):
        """nvfp4 (packed uint8 + e4m3 block-16 scale + per-tensor scale_2) fused dequant: components
        built by torchao ``to_nvfp4`` (the same schema the loader reads) recover the original weight
        within nvfp4 tolerance, and the scale tensors are dropped. ``_dequant_nvfp4`` reconstructs the
        NVFP4Tensor identically to nvfp4_moe_loading's loader path."""
        pytest.importorskip("torchao")
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        except Exception:  # pragma: no cover
            pytest.skip("NVFP4Tensor unavailable")
        from axolotl.cli.utils.lora_merge import _dequantize_quantized_shard

        torch.manual_seed(4)
        N, K = 32, 64
        w = torch.randn(N, K, dtype=torch.bfloat16) * 0.2
        p = (w.abs().max() / (6.0 * 448.0)).reshape(1).float().clamp(min=1e-12)
        try:
            nv = NVFP4Tensor.to_nvfp4(w, per_tensor_scale=p, is_swizzled_scales=False)
        except Exception as ex:  # pragma: no cover - torchao API / device gaps
            pytest.skip(f"to_nvfp4 unavailable on this host: {ex}")

        key = "model.layers.0.self_attn.q_proj.weight"
        shard = {
            key: nv.qdata,
            key + "_scale": nv.scale,
            key + "_scale_2": nv.per_tensor_scale.reshape(()),
        }
        out, did, _, _ = _dequantize_quantized_shard(shard, "cpu")
        assert did
        assert key + "_scale" not in out and key + "_scale_2" not in out
        assert out[key].dtype == torch.bfloat16
        # nvfp4 round-trip recovers the weight within fp4 (3-mantissa-bit) tolerance; a wrong
        # packing/scale interpretation would give garbage, not ~w.
        rel = (out[key].float() - w.float()).norm() / w.float().norm()
        assert rel < 0.2, f"nvfp4 dequant round-trip rel {rel:.2e}"

    def test_undequantized_quant_base_warns(self):
        """A LoRA folding into a still-quantized weight (an unhandled format the shard dequant left
        as uint8/fp8) must WARN loudly rather than silently corrupt — the safety net for mxfp4 /
        per-tensor-fp8 / per-expert-unfused layouts."""
        from unittest.mock import patch

        import axolotl.cli.utils.lora_merge as lm

        lm._WARNED_UNDEQUANT.clear()
        hidden, r, alpha = 16, 4, 8
        # a packed-4bit-like uint8 weight that the dequant step did NOT handle
        tensor = torch.randint(0, 255, (hidden, hidden), dtype=torch.uint8)
        key = "model.layers.0.self_attn.q_proj.weight"
        lora_state = {
            f"base_model.model.{key[: -len('.weight')]}.lora_A.weight": torch.randn(
                r, hidden
            ),
            f"base_model.model.{key[: -len('.weight')]}.lora_B.weight": torch.randn(
                hidden, r
            ),
        }
        with patch.object(lm.LOG, "warning") as mock_warn:
            # the warning fires BEFORE the fold; folding into raw uint8 then errors (as it should) —
            # tolerate that, we're asserting the guard warned.
            try:
                lm._merge_tensor_with_lora(
                    tensor,
                    key,
                    lora_state,
                    alpha / r,
                    {"r": r, "lora_alpha": alpha},
                    "cpu",
                )
            except RuntimeError:
                pass
        msgs = " ".join(str(c.args[0]) for c in mock_warn.call_args_list)
        assert "still %s" in msgs or "quantized format" in msgs, (
            "expected undequant warning"
        )

    def test_dequantize_mxfp4_shard(self):
        """mxfp4 (packed e2m1 + e8m0/32) fused dequant recovers the fp4-quantized weight. Built with
        the codebook + low/high nibble order the ScatterMoE MX forward uses, so the merged weight
        equals what the model computes."""
        from axolotl.cli.utils.lora_merge import (
            _FP4_E2M1_LUT,
            _dequantize_quantized_shard,
        )

        torch.manual_seed(5)
        N, K, block = 8, 64, 32
        lut = torch.tensor(_FP4_E2M1_LUT)
        w = torch.randn(N, K) * 0.5
        # per-32-block e8m0 scale, then quantize each element to the nearest codebook value
        nb = K // block
        amax = w.reshape(N, nb, block).abs().amax(-1).clamp_min(1e-6)
        exp = torch.floor(torch.log2(amax / 6.0))  # 6 = fp4 max
        scale = torch.exp2(exp)  # [N, nb]
        wn = (w.reshape(N, nb, block) / scale[..., None]).reshape(N, K)
        idx = (wn.unsqueeze(-1) - lut).abs().argmin(-1)  # nearest codebook index [N,K]
        qvals = lut[idx]
        packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).to(
            torch.uint8
        )  # low=even, high=odd
        ebyte = (exp + 127.0).to(torch.uint8)
        key = "model.layers.0.mlp.experts.gate_up_proj"  # 2D here for simplicity
        out, did, _, _ = _dequantize_quantized_shard(
            {key: packed, key + "_scale": ebyte}, "cpu"
        )
        assert did and out[key].dtype == torch.bfloat16 and key + "_scale" not in out
        expected = (qvals.reshape(N, nb, block) * scale[..., None]).reshape(N, K)
        assert torch.allclose(out[key].float(), expected, atol=1e-2)

    def test_dequantize_nvfp4_single_level(self):
        """Single-level nvfp4 (e4m3 block-16 scale, NO scale_2) dequants with per_tensor_scale=1."""
        pytest.importorskip("torchao")
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        except Exception:  # pragma: no cover
            pytest.skip("NVFP4Tensor unavailable")
        from axolotl.cli.utils.lora_merge import _dequantize_quantized_shard

        torch.manual_seed(6)
        w = torch.randn(16, 64, dtype=torch.bfloat16) * 0.2
        try:
            nv = NVFP4Tensor.to_nvfp4(
                w, is_swizzled_scales=False
            )  # no per_tensor_scale
        except Exception as ex:  # pragma: no cover
            pytest.skip(f"to_nvfp4 unavailable: {ex}")
        key = "model.layers.0.self_attn.q_proj.weight"
        out, did, _, _ = _dequantize_quantized_shard(
            {key: nv.qdata, key + "_scale": nv.scale}, "cpu"
        )
        assert did and out[key].dtype == torch.bfloat16
        assert (out[key].float() - w.float()).norm() / w.float().norm() < 0.2

    def test_mxfp8_ragged_blocks(self):
        """mxfp8 dequant: each e8m0 scale covers a fixed 32-wide MX block; the final ragged block
        (K=40 -> blocks of 32 + 8) is trimmed, NOT floor-spread evenly across K."""
        from axolotl.cli.utils.lora_merge import _dequant_mxfp8

        torch.manual_seed(7)
        N, K = (
            4,
            40,
        )  # nb = ceil(40/32) = 2: first 32 elems use scale[0], last 8 use scale[1]
        w = (torch.randn(N, K) * 0.1).to(torch.float8_e4m3fn)
        s = torch.randint(120, 130, (N, 2), dtype=torch.uint8)
        out = _dequant_mxfp8(w, s, "cpu")
        assert out.shape == (N, K)
        scale = torch.exp2(s.float() - 127.0)  # [N, 2]
        exp = w.float().clone()
        exp[:, :32] *= scale[:, :1]
        exp[:, 32:] *= scale[:, 1:2]
        assert torch.allclose(out.float(), exp, atol=1e-2)

    def test_dora_on_block_fp8_base(self):
        """DoRA merge on a block-fp8 base: dequant -> DoRA magnitude-normalized fold -> bf16 (the
        quant base goes through the same DoRA path, no crash, finite result)."""
        from axolotl.cli.utils.lora_merge import (
            _dequantize_quantized_shard,
            _merge_tensor_with_lora,
        )

        torch.manual_seed(8)
        hidden, r, alpha = 32, 8, 16
        w = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2
        q, si = self._make_block_fp8(w, 16)
        key = "model.layers.0.self_attn.q_proj.weight"
        deq, _, _, _ = _dequantize_quantized_shard(
            {key: q, key + "_scale_inv": si}, "cpu"
        )
        lora_state = {
            f"base_model.model.{key[: -len('.weight')]}.lora_A.weight": torch.randn(
                r, hidden
            )
            * 0.1,
            f"base_model.model.{key[: -len('.weight')]}.lora_B.weight": torch.randn(
                hidden, r
            )
            * 0.1,
            f"base_model.model.{key[: -len('.weight')]}.lora_magnitude_vector": torch.randn(
                hidden
            ).abs()
            + 0.1,
        }
        merged, was = _merge_tensor_with_lora(
            deq[key],
            key,
            lora_state,
            alpha / r,
            {"r": r, "lora_alpha": alpha, "use_dora": True},
            "cpu",
            use_dora=True,
        )
        assert (
            was
            and merged.dtype == torch.bfloat16
            and torch.isfinite(merged.float()).all()
        )

    def test_resized_embeddings_override_carried(self, tmp_path):
        """A resized embed_tokens/lm_head saved as full weights in the adapter REPLACES the base and
        bumps config.json vocab_size (otherwise the trained/enlarged vocab is silently dropped)."""
        from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient

        hidden, base_vocab, new_vocab = 8, 10, 12
        model_dir = tmp_path / "base"
        model_dir.mkdir()
        safetensors.torch.save_file(
            {
                "model.embed_tokens.weight": torch.randn(base_vocab, hidden),
                "lm_head.weight": torch.randn(base_vocab, hidden),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            },
            model_dir / "model.safetensors",
        )
        (model_dir / "config.json").write_text(json.dumps({"vocab_size": base_vocab}))

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        new_embed = torch.randn(new_vocab, hidden)
        new_head = torch.randn(new_vocab, hidden)
        safetensors.torch.save_file(
            {
                "base_model.model.model.embed_tokens.weight": new_embed,
                "base_model.model.lm_head.weight": new_head,
            },
            adapter_dir / "adapter_model.safetensors",
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": 8, "lora_alpha": 16, "peft_type": "LORA"})
        )

        out_dir = tmp_path / "merged"
        merge_lora_sharded_efficient(
            base_model_path=model_dir,
            lora_adapter_path=adapter_dir,
            output_path=out_dir,
            device="cpu",
        )
        merged = {}
        with safetensors.torch.safe_open(
            out_dir / "model.safetensors", framework="pt"
        ) as f:
            for k in f.keys():
                merged[k] = f.get_tensor(k)
        assert merged["model.embed_tokens.weight"].shape[0] == new_vocab
        assert merged["lm_head.weight"].shape[0] == new_vocab
        assert torch.allclose(merged["model.embed_tokens.weight"].float(), new_embed)
        assert torch.allclose(merged["lm_head.weight"].float(), new_head)
        assert (
            json.loads((out_dir / "config.json").read_text())["vocab_size"] == new_vocab
        )

    def test_per_expert_unfused_mismatch_warns(self, tmp_path):
        """A fused expert LoRA adapter over a PER-EXPERT-unfused base must warn (the expert LoRA would
        otherwise be silently dropped by the shard merge)."""
        from unittest.mock import patch

        import axolotl.cli.utils.lora_merge as lm

        model_dir = tmp_path / "base"
        model_dir.mkdir()
        # per-expert unfused base experts
        safetensors.torch.save_file(
            {
                "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
                "model.layers.0.mlp.experts.1.gate_proj.weight": torch.randn(8, 8),
            },
            model_dir / "model.safetensors",
        )
        (model_dir / "config.json").write_text("{}")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        # fused expert LoRA (targets experts.gate_up_proj)
        safetensors.torch.save_file(
            {
                "base_model.model.model.layers.0.mlp.experts.lora_A.weight": torch.randn(
                    16, 8
                ),
                "base_model.model.model.layers.0.mlp.experts.lora_B.weight": torch.randn(
                    16, 16
                ),
            },
            adapter_dir / "adapter_model.safetensors",
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": 8, "lora_alpha": 16, "peft_type": "LORA"})
        )

        with patch.object(lm.LOG, "warning") as mock_warn:
            lm.merge_lora_sharded_efficient(
                base_model_path=model_dir,
                lora_adapter_path=adapter_dir,
                output_path=tmp_path / "merged",
                device="cpu",
            )
        assert any("PER-EXPERT" in str(c.args[0]) for c in mock_warn.call_args_list), (
            "expected a per-expert-unfused mismatch warning"
        )

    def test_nvfp4_expert_merge_writer(self, tmp_path):
        """The expert-merge writer folds a FUSED expert LoRA into a PER-EXPERT unfused NVFP4 base:
        per expert the output must be BITWISE requant(dequant(base) + delta), including a layer split
        across the shard boundary; non-LoRA layers pass through byte-identical; --dequant emits the
        fused bf16 param instead."""
        pytest.importorskip("torchao")
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        from axolotl.cli.utils.lora_merge import (
            _build_peft_layer_and_get_delta,
            _dequant_nvfp4,
            _find_param_wrapper_lora,
            _requant_by_format,
            merge_lora_sharded_efficient,
        )

        torch.manual_seed(0)
        E, H, I, r, alpha = 4, 64, 16, 4, 8

        def quant(w2d):
            p = (w2d.abs().max() / (6.0 * 448.0)).reshape(1).clamp_min(1e-12)
            nv = NVFP4Tensor.to_nvfp4(
                w2d.float(), per_tensor_scale=p, is_swizzled_scales=False
            )
            return (
                nv.qdata,
                nv.scale.to(torch.float8_e4m3fn),
                nv.per_tensor_scale.reshape(()),
            )

        def make_layer(layer, seed):
            torch.manual_seed(seed)
            keys = {}
            for proj, (n, k) in (
                ("gate_proj", (I, H)),
                ("up_proj", (I, H)),
                ("down_proj", (H, I)),
            ):
                for e in range(E):
                    qd, sc, pts = quant(torch.randn(n, k) * k**-0.5)
                    base = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                    keys[f"{base}.weight"] = qd
                    keys[f"{base}.weight_scale"] = sc
                    keys[f"{base}.weight_scale_2"] = pts
            return keys

        def subset(keys, experts):
            return {
                k: v
                for k, v in keys.items()
                if int(k.split(".experts.")[1].split(".")[0]) in experts
            }

        # layer 0 complete in shard 0; layer 1 SPLIT across shards; layer 2 has no LoRA
        l0, l1, l2 = make_layer(0, 10), make_layer(1, 11), make_layer(2, 12)
        shard0 = {**l0, **subset(l1, {0, 1})}
        shard1 = {**subset(l1, {2, 3}), **l2}

        torch.manual_seed(99)
        adapter = {}
        for layer in (0, 1):
            p = f"base_model.model.model.layers.{layer}.mlp.experts"
            # outer wrapper = down_proj, inner .base_layer = gate_up (sonicmoe chain);
            # Qwen3 orientation [E, out, in]: lora_A [r*E, in], lora_B [out, r*E]
            adapter[f"{p}.lora_A.weight"] = (
                torch.randn(r * E, I, dtype=torch.bfloat16) * 0.2
            )
            adapter[f"{p}.lora_B.weight"] = (
                torch.randn(H, r * E, dtype=torch.bfloat16) * 0.2
            )
            adapter[f"{p}.base_layer.lora_A.weight"] = (
                torch.randn(r * E, H, dtype=torch.bfloat16) * 0.2
            )
            adapter[f"{p}.base_layer.lora_B.weight"] = (
                torch.randn(2 * I, r * E, dtype=torch.bfloat16) * 0.2
            )

        base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
        base_dir.mkdir(), adapter_dir.mkdir()
        safetensors.torch.save_file(
            shard0, base_dir / "model-00001-of-00002.safetensors"
        )
        safetensors.torch.save_file(
            shard1, base_dir / "model-00002-of-00002.safetensors"
        )
        wmap = {k: "model-00001-of-00002.safetensors" for k in shard0}
        wmap.update({k: "model-00002-of-00002.safetensors" for k in shard1})
        (base_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": wmap})
        )
        (base_dir / "config.json").write_text(
            json.dumps({"model_type": "synthetic-test", "num_experts": E})
        )
        safetensors.torch.save_file(adapter, adapter_dir / "adapter_model.safetensors")
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": r, "lora_alpha": alpha, "peft_type": "LORA"})
        )

        out = tmp_path / "merged"
        merge_lora_sharded_efficient(base_dir, adapter_dir, out, device="cpu")
        merged = {}
        for f in sorted(out.glob("*.safetensors")):
            merged.update(safetensors.torch.load_file(f))

        all_base = {**shard0, **shard1}
        assert set(merged) == set(all_base)
        assert all(
            merged[k].dtype == v.dtype and merged[k].shape == v.shape
            for k, v in all_base.items()
        )
        assert all(torch.equal(merged[k], v) for k, v in l2.items())

        def fused_dequant(keys, layer, projs):
            per = [
                torch.stack(
                    [
                        _dequant_nvfp4(
                            keys[f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"],
                            keys[
                                f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight_scale"
                            ],
                            keys[
                                f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight_scale_2"
                            ],
                            "cpu",
                        )
                        for e in range(E)
                    ]
                )
                for proj in projs
            ]
            return per[0] if len(per) == 1 else torch.cat(per, dim=1)

        cfg = {"r": r, "lora_alpha": alpha}
        for layer, src in ((0, l0), (1, l1)):
            for fused_name, projs in (
                ("gate_up_proj", ("gate_proj", "up_proj")),
                ("down_proj", ("down_proj",)),
            ):
                base_f = fused_dequant(src, layer, projs)
                lora_a, lora_b, _ = _find_param_wrapper_lora(
                    adapter,
                    f"model.layers.{layer}.mlp.experts.{fused_name}",
                    tensor_shape=tuple(base_f.shape),
                )
                assert lora_a is not None, f"orientation fix: L{layer} {fused_name}"
                delta = _build_peft_layer_and_get_delta(
                    lora_a, lora_b, cfg, base_f, is_param_wrapper=True
                )
                want = (base_f.float() + delta.float()).to(torch.bfloat16)
                col = 0
                for proj in projs:
                    n_rows = src[
                        f"model.layers.{layer}.mlp.experts.0.{proj}.weight"
                    ].shape[0]
                    for e in range(E):
                        kb = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                        ref = _requant_by_format(
                            "nvfp4",
                            want[e, col : col + n_rows, :],
                            {
                                "_scale": src[f"{kb}.weight_scale"],
                                "_scale_2": src[f"{kb}.weight_scale_2"],
                            },
                            "cpu",
                        )
                        assert torch.equal(merged[f"{kb}.weight"], ref[""])
                        assert torch.equal(
                            merged[f"{kb}.weight_scale"].view(torch.uint8),
                            ref["_scale"].view(torch.uint8),
                        )
                        assert torch.equal(
                            merged[f"{kb}.weight_scale_2"], ref["_scale_2"]
                        )
                    col += n_rows

        # --dequant: the writer emits the merged FUSED bf16 param instead
        out2 = tmp_path / "merged_dq"
        merge_lora_sharded_efficient(
            base_dir, adapter_dir, out2, device="cpu", dequant=True
        )
        merged2 = {}
        for f in sorted(out2.glob("*.safetensors")):
            merged2.update(safetensors.torch.load_file(f))
        for layer in (0, 1):
            for fused_name, projs in (
                ("gate_up_proj", ("gate_proj", "up_proj")),
                ("down_proj", ("down_proj",)),
            ):
                key = f"model.layers.{layer}.mlp.experts.{fused_name}"
                assert merged2[key].dtype == torch.bfloat16
                src = l0 if layer == 0 else l1
                base_f = fused_dequant(src, layer, projs)
                lora_a, lora_b, _ = _find_param_wrapper_lora(
                    adapter, key, tensor_shape=tuple(merged2[key].shape)
                )
                delta = _build_peft_layer_and_get_delta(
                    lora_a, lora_b, cfg, base_f, is_param_wrapper=True
                )
                want = (base_f.float() + delta.float()).to(torch.bfloat16)
                assert torch.allclose(merged2[key].float(), want.float(), atol=1e-2)

    def test_nvfp4_merge_aware_quantizer_identity(self):
        """The merge-aware invariant: one quantizer, bitwise, on both sides. Fresh mode
        must equal torchao's ``to_nvfp4`` (the ecosystem encoder), quantizing the fused
        ``[E, 2I, H]`` training view must equal quantizing per-projection row slices with
        the same pts, and ``fake_quant_nvfp4`` must be idempotent (a snapped weight
        re-quantizes to itself -> merge retention by construction)."""
        pytest.importorskip("torchao")
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
            fake_quant_nvfp4,
            quantize_nvfp4_merge,
        )

        torch.manual_seed(0)
        w = (torch.randn(48, 128) * 0.02).float()
        pts = (w.abs().max() / (6.0 * 448.0)).reshape(())
        ref = NVFP4Tensor.to_nvfp4(w, per_tensor_scale=pts)
        p, s = quantize_nvfp4_merge(w, pts, scale_mode="fresh")
        assert torch.equal(p, ref.qdata)
        assert torch.equal(
            s.view(torch.uint8), ref.scale.reshape(s.shape).view(torch.uint8)
        )

        E, N, K = 4, 64, 128
        wf = (torch.randn(E, N, K) * 0.02).to(torch.bfloat16)
        pts_e = torch.rand(E) * 1e-4 + 1e-5
        pf, sf = quantize_nvfp4_merge(wf, pts_e, scale_mode="fresh")
        for e in range(E):
            for rows in (slice(0, N // 2), slice(N // 2, N)):
                pe, se = quantize_nvfp4_merge(
                    wf[e, rows].contiguous(), pts_e[e], scale_mode="fresh"
                )
                assert torch.equal(pe, pf[e, rows])
                assert torch.equal(se.view(torch.uint8), sf[e, rows].view(torch.uint8))

        fq = fake_quant_nvfp4(wf, pts_e)
        nv = NVFP4Tensor(
            pf, sf, 16, torch.bfloat16, per_tensor_scale=pts_e.reshape(-1, 1, 1)
        )
        assert fq.dtype == torch.bfloat16
        assert torch.equal(fq, nv.dequantize(torch.bfloat16))
        assert torch.equal(fake_quant_nvfp4(fq, pts_e), fq)

    def test_nonexpert_fresh_requant_matches_training_snap(self):
        """The non-expert (2D linear, e.g. attention) writer path: fresh-mode
        ``_requant_by_format`` on the merged bf16 weight must reproduce bitwise the
        grid the merge-aware LoRA-linear forward trained against (same
        ``quantize_nvfp4_merge`` + frozen base pts)."""
        pytest.importorskip("torchao")
        import axolotl.cli.utils.lora_merge as lm
        from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
            fake_quant_nvfp4,
            quantize_nvfp4_merge,
        )

        torch.manual_seed(0)
        w0 = (torch.randn(32, 64) * 0.02).to(torch.bfloat16)
        pts = (w0.float().abs().max() / (6.0 * 448.0)).reshape(())
        base_w = fake_quant_nvfp4(w0, pts)
        _, base_scale = quantize_nvfp4_merge(base_w, pts, scale_mode="fresh")

        w_eff = base_w + (torch.randn_like(base_w) * 0.002).to(torch.bfloat16)
        train_packed, train_scale = quantize_nvfp4_merge(w_eff, pts, scale_mode="fresh")

        out = lm._requant_by_format(
            "nvfp4",
            w_eff,
            {"_scale": base_scale, "_scale_2": pts},
            "cpu",
            nvfp4_scale_mode="fresh",
        )
        assert torch.equal(out[""], train_packed)
        assert torch.equal(
            out["_scale"].view(torch.uint8), train_scale.view(torch.uint8)
        )
        assert torch.equal(out["_scale_2"], pts)

    def test_nvfp4_expert_writer_fresh_mode_matches_training_grid(self):
        """``scale_mode="fresh"``: the expert writer's bytes ARE the training grid.
        Gate/up export UNEQUAL pts on purpose, so the loader's fused-max fold is in
        play; the writer must rebuild that fused view, emit the fused max as every
        projection's ``weight_scale_2``, and dequantizing the emitted tensors must
        BITWISE equal ``fake_quant_nvfp4`` of the fused merged weight (what a
        merge-aware training forward computed on its last step)."""
        pytest.importorskip("torchao")
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        from axolotl.cli.utils.lora_merge import (
            _build_peft_layer_and_get_delta,
            _find_param_wrapper_lora,
            _Nvfp4ExpertMergeWriter,
        )
        from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
            fake_quant_nvfp4,
        )

        torch.manual_seed(3)
        E, H, I, r, alpha = 2, 64, 32, 4, 8

        def quant(w2d):
            p = (w2d.abs().max() / (6.0 * 448.0)).reshape(1).clamp_min(1e-12)
            nv = NVFP4Tensor.to_nvfp4(
                w2d.float(), per_tensor_scale=p, is_swizzled_scales=False
            )
            return (
                nv.qdata,
                nv.scale.to(torch.float8_e4m3fn),
                nv.per_tensor_scale.reshape(()),
            )

        shard = {}
        # gate scaled 3x vs up -> unequal weight_scale_2 across the fused pair
        for proj, (n, k), mult in (
            ("gate_proj", (I, H), 3.0),
            ("up_proj", (I, H), 1.0),
            ("down_proj", (H, I), 1.0),
        ):
            for e in range(E):
                qd, sc, pts = quant(torch.randn(n, k) * k**-0.5 * mult)
                base = f"model.layers.0.mlp.experts.{e}.{proj}"
                shard[f"{base}.weight"] = qd
                shard[f"{base}.weight_scale"] = sc
                shard[f"{base}.weight_scale_2"] = pts

        p = "base_model.model.model.layers.0.mlp.experts"
        adapter = {
            f"{p}.lora_A.weight": torch.randn(r * E, I, dtype=torch.bfloat16) * 0.2,
            f"{p}.lora_B.weight": torch.randn(H, r * E, dtype=torch.bfloat16) * 0.2,
            f"{p}.base_layer.lora_A.weight": torch.randn(r * E, H, dtype=torch.bfloat16)
            * 0.2,
            f"{p}.base_layer.lora_B.weight": torch.randn(
                2 * I, r * E, dtype=torch.bfloat16
            )
            * 0.2,
        }
        cfg = {"r": r, "lora_alpha": alpha}

        writer = _Nvfp4ExpertMergeWriter(adapter, cfg, E, "cpu", scale_mode="fresh")
        remaining, emitted, merged = writer.consume(shard)
        writer.assert_drained()
        assert merged == 2 and not remaining

        for fused_name, projs in (
            ("gate_up_proj", ("gate_proj", "up_proj")),
            ("down_proj", ("down_proj",)),
        ):
            # the training view: fuse exactly as fuse_nvfp4_experts does
            pts_all = [
                torch.stack(
                    [
                        shard[f"model.layers.0.mlp.experts.{e}.{proj}.weight_scale_2"]
                        for e in range(E)
                    ]
                ).view(-1, 1, 1)
                for proj in projs
            ]
            pts_fused = pts_all[0]
            for pts_i in pts_all[1:]:
                pts_fused = torch.maximum(pts_fused, pts_i)
            per = []
            for i, proj in enumerate(projs):
                qd = torch.stack(
                    [
                        shard[f"model.layers.0.mlp.experts.{e}.{proj}.weight"]
                        for e in range(E)
                    ]
                )
                sc = torch.stack(
                    [
                        shard[f"model.layers.0.mlp.experts.{e}.{proj}.weight_scale"]
                        for e in range(E)
                    ]
                )
                if not torch.allclose(pts_all[i], pts_fused):
                    sc = (sc.float() * (pts_all[i] / pts_fused)).to(torch.float8_e4m3fn)
                per.append(
                    NVFP4Tensor(
                        qd, sc, 16, torch.bfloat16, per_tensor_scale=pts_fused
                    ).dequantize(torch.bfloat16)
                )
            base_f = per[0] if len(per) == 1 else torch.cat(per, dim=1)
            lora_a, lora_b, _ = _find_param_wrapper_lora(
                adapter,
                f"model.layers.0.mlp.experts.{fused_name}",
                tensor_shape=tuple(base_f.shape),
            )
            assert lora_a is not None
            delta = _build_peft_layer_and_get_delta(
                lora_a, lora_b, cfg, base_f, is_param_wrapper=True
            )
            w_eff = (base_f.float() + delta.float()).to(torch.bfloat16)
            train_view = fake_quant_nvfp4(w_eff, pts_fused.reshape(-1))

            # every projection must carry the fused-max pts, so the next load
            # fuses exactly (no ratio fold, no warning)
            for proj in projs:
                for e in range(E):
                    kb = f"model.layers.0.mlp.experts.{e}.{proj}"
                    assert torch.equal(
                        emitted[f"{kb}.weight_scale_2"].reshape(()),
                        pts_fused[e].reshape(()),
                    )
            # reload the emitted tensors the way fuse_nvfp4_experts will:
            # stack experts, cat projections, one per-expert pts
            qd_r = torch.cat(
                [
                    torch.stack(
                        [
                            emitted[f"model.layers.0.mlp.experts.{e}.{proj}.weight"]
                            for e in range(E)
                        ]
                    )
                    for proj in projs
                ],
                dim=1,
            )
            sc_r = torch.cat(
                [
                    torch.stack(
                        [
                            emitted[
                                f"model.layers.0.mlp.experts.{e}.{proj}.weight_scale"
                            ]
                            for e in range(E)
                        ]
                    )
                    for proj in projs
                ],
                dim=1,
            )
            reloaded = NVFP4Tensor(
                qd_r, sc_r, 16, torch.bfloat16, per_tensor_scale=pts_fused
            ).dequantize(torch.bfloat16)
            assert torch.equal(reloaded, train_view)

    def test_nvfp4_merge_aware_metadata_resolution(self):
        """adapter_config.json quantizer-identity metadata drives the requant mode:
        absent -> reuse; present -> fresh; any identity mismatch hard-errors unless
        overridden (a wrong quantizer silently voids the retention guarantee)."""
        pytest.importorskip("torchao")
        import torchao

        from axolotl.cli.utils.lora_merge import _resolve_nvfp4_scale_mode

        assert _resolve_nvfp4_scale_mode({"r": 4}) == "reuse"

        good = {
            "nvfp4_merge_aware": {
                "scale_mode": "fresh",
                "pts_policy": "base_fused_max",
                "encoder": f"torchao-{torchao.__version__}",
                "start_step": 0,
            }
        }
        assert _resolve_nvfp4_scale_mode(good) == "fresh"
        # unrecorded encoder (older stamp): accepted
        assert _resolve_nvfp4_scale_mode({"nvfp4_merge_aware": {}}) == "fresh"

        with pytest.raises(RuntimeError, match="override-quantizer"):
            _resolve_nvfp4_scale_mode(
                {"nvfp4_merge_aware": {"encoder": "torchao-0.0.1"}}
            )
        assert (
            _resolve_nvfp4_scale_mode(
                {"nvfp4_merge_aware": {"encoder": "torchao-0.0.1"}},
                override_quantizer=True,
            )
            == "fresh"
        )
        with pytest.raises(ValueError, match="scale_mode"):
            _resolve_nvfp4_scale_mode({"nvfp4_merge_aware": {"scale_mode": "reuse"}})
        with pytest.raises(ValueError, match="pts_policy"):
            _resolve_nvfp4_scale_mode({"nvfp4_merge_aware": {"pts_policy": "per_proj"}})

    @staticmethod
    def _quant_expert(w2d):
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        p = (w2d.abs().max() / (6.0 * 448.0)).reshape(1).clamp_min(1e-12)
        nv = NVFP4Tensor.to_nvfp4(
            w2d.float(), per_tensor_scale=p, is_swizzled_scales=False
        )
        return (
            nv.qdata,
            nv.scale.to(torch.float8_e4m3fn),
            nv.per_tensor_scale.reshape(()),
        )

    def _make_expert_shard(self, E, H, I, gate_mult=1.0):  # noqa: E741
        shard = {}
        for proj, (n, k), mult in (
            ("gate_proj", (I, H), gate_mult),
            ("up_proj", (I, H), 1.0),
            ("down_proj", (H, I), 1.0),
        ):
            for e in range(E):
                qd, sc, pts = self._quant_expert(torch.randn(n, k) * k**-0.5 * mult)
                base = f"model.layers.0.mlp.experts.{e}.{proj}"
                shard[f"{base}.weight"] = qd
                shard[f"{base}.weight_scale"] = sc
                shard[f"{base}.weight_scale_2"] = pts
        return shard

    @staticmethod
    def _make_fused_expert_adapter(E, H, I, r, scale=0.2):  # noqa: E741
        p = "base_model.model.model.layers.0.mlp.experts"
        return {
            f"{p}.lora_A.weight": torch.randn(r * E, I, dtype=torch.bfloat16) * scale,
            f"{p}.lora_B.weight": torch.randn(H, r * E, dtype=torch.bfloat16) * scale,
            f"{p}.base_layer.lora_A.weight": torch.randn(r * E, H, dtype=torch.bfloat16)
            * scale,
            f"{p}.base_layer.lora_B.weight": torch.randn(
                2 * I, r * E, dtype=torch.bfloat16
            )
            * scale,
        }

    def test_nvfp4_merge_aware_metadata_selects_fresh_mode_e2e(self, tmp_path):
        """A stamped adapter merged through merge_lora_sharded_efficient must requant
        with fresh scales: gate/up emit the SAME (fused-max) weight_scale_2 even though
        the base exported unequal pts; --dequant is rejected (the un-snapped bf16 merge
        is not the trained function); a stale encoder refuses to merge unless
        overridden."""
        pytest.importorskip("torchao")
        import torchao

        from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient

        torch.manual_seed(11)
        E, H, I, r, alpha = 2, 64, 32, 4, 8  # noqa: E741
        shard = self._make_expert_shard(E, H, I, gate_mult=3.0)
        adapter = self._make_fused_expert_adapter(E, H, I, r)

        base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
        base_dir.mkdir(), adapter_dir.mkdir()
        safetensors.torch.save_file(shard, base_dir / "model.safetensors")
        (base_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {k: "model.safetensors" for k in shard},
                }
            )
        )
        (base_dir / "config.json").write_text(
            json.dumps({"model_type": "synthetic-test", "num_experts": E})
        )
        safetensors.torch.save_file(adapter, adapter_dir / "adapter_model.safetensors")
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": r,
                    "lora_alpha": alpha,
                    "peft_type": "LORA",
                    "nvfp4_merge_aware": {
                        "scale_mode": "fresh",
                        "pts_policy": "base_fused_max",
                        "encoder": f"torchao-{torchao.__version__}",
                        "start_step": 0,
                    },
                }
            )
        )

        out = tmp_path / "merged"
        merge_lora_sharded_efficient(base_dir, adapter_dir, out, device="cpu")
        merged = {}
        for f in sorted(out.glob("*.safetensors")):
            merged.update(safetensors.torch.load_file(f))

        for e in range(E):
            kb = f"model.layers.0.mlp.experts.{e}"
            g = merged[f"{kb}.gate_proj.weight_scale_2"].reshape(())
            u = merged[f"{kb}.up_proj.weight_scale_2"].reshape(())
            want = torch.maximum(
                shard[f"{kb}.gate_proj.weight_scale_2"],
                shard[f"{kb}.up_proj.weight_scale_2"],
            )
            assert torch.equal(g, u)
            assert torch.equal(g, want.reshape(()))

        with pytest.raises(ValueError, match="dequant"):
            merge_lora_sharded_efficient(
                base_dir, adapter_dir, tmp_path / "m_dq", device="cpu", dequant=True
            )

        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        cfg["nvfp4_merge_aware"]["encoder"] = "torchao-0.0.1"
        (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))
        with pytest.raises(RuntimeError, match="override-quantizer"):
            merge_lora_sharded_efficient(
                base_dir, adapter_dir, tmp_path / "m2", device="cpu"
            )
        merge_lora_sharded_efficient(
            base_dir,
            adapter_dir,
            tmp_path / "m3",
            device="cpu",
            override_quantizer=True,
        )

    def test_nvfp4_near_noop_merge_warns(self):
        """An unprepared adapter whose delta is far below the grid step must warn
        that the format-preserving merge is a near-no-op."""
        pytest.importorskip("torchao")
        from unittest.mock import patch

        import axolotl.cli.utils.lora_merge as lm

        torch.manual_seed(13)
        E, H, I, r, alpha = 2, 64, 32, 4, 8  # noqa: E741
        shard = self._make_expert_shard(E, H, I)
        # sub-grid-step delta: ~1e-6 relative, rounds back to the base codes
        adapter = self._make_fused_expert_adapter(E, H, I, r, scale=1e-6)

        writer = lm._Nvfp4ExpertMergeWriter(
            adapter, {"r": r, "lora_alpha": alpha}, E, "cpu"
        )
        _, emitted, merged = writer.consume(shard)
        assert merged == 2 and emitted
        with patch.object(lm.LOG, "warning") as mock_warn:
            writer.assert_drained()
        assert any("NEAR-NO-OP" in str(c.args[0]) for c in mock_warn.call_args_list)

    def test_fused_expert_base_no_false_positive(self, tmp_path):
        """A FUSED expert base (Mistral-Large-4 style) + fused adapter must NOT trigger the per-expert
        warning."""
        from axolotl.cli.utils.lora_merge import _detect_per_expert_unfused_mismatch

        model_dir = tmp_path / "base"
        model_dir.mkdir()
        safetensors.torch.save_file(
            {"model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 16, 8)},
            model_dir / "model.safetensors",
        )
        lora_state = {
            "base_model.model.model.layers.0.mlp.experts.lora_A.weight": torch.randn(
                16, 8
            )
        }
        assert not _detect_per_expert_unfused_mismatch(
            [model_dir / "model.safetensors"], lora_state
        )

    def test_expert_lora_delta_matches_scattermoe_kernel(self):
        """The merge's PEFT ParamWrapper expert delta EXACTLY equals what the ScatterMoE training
        kernel applies (expert-major B via peft_lora_B_to_scattermoe). Guards against a rank-major vs
        expert-major layout regression — the naive slice differs materially."""
        pytest.importorskip("triton")
        from axolotl.cli.utils.lora_merge import _build_peft_layer_and_get_delta
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            peft_lora_B_to_scattermoe,
        )

        torch.manual_seed(0)
        E, r, IN, OUT, alpha = 4, 8, 16, 24, 16
        scaling = alpha / r
        A = torch.randn(r * E, IN) * 0.1
        B = torch.randn(OUT, r * E) * 0.1
        base = torch.randn(E, OUT, IN)

        mine = _build_peft_layer_and_get_delta(
            A, B, {"r": r, "lora_alpha": alpha}, base, is_param_wrapper=True
        )
        smB = peft_lora_B_to_scattermoe(B, E, r)
        kern = torch.stack(
            [
                scaling * (smB[:, e * r : (e + 1) * r] @ A[e * r : (e + 1) * r, :])
                for e in range(E)
            ]
        )
        naive = torch.stack(
            [
                scaling * (B[:, e * r : (e + 1) * r] @ A[e * r : (e + 1) * r, :])
                for e in range(E)
            ]
        )
        assert torch.allclose(mine.float(), kern.float(), atol=1e-5)
        # sanity: the layouts genuinely differ, so the test isn't vacuous
        assert (mine.float() - naive.float()).norm() / naive.float().norm() > 0.5

    def test_expert_3d_block_fp8_merge_folds(self):
        """Full 3D fused-expert merge on a block-fp8 base: dequant -> PEFT expert delta -> matches the
        kernel-layout reconstruction on the dequantized base."""
        pytest.importorskip("triton")
        from axolotl.cli.utils.lora_merge import (
            _build_peft_layer_and_get_delta,
            _dequantize_quantized_shard,
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            peft_lora_B_to_scattermoe,
        )

        torch.manual_seed(1)
        E, r, IN, OUT, alpha = 4, 8, 128, 256, 16
        scaling = alpha / r
        w = torch.randn(E, OUT, IN, dtype=torch.bfloat16) * 0.1
        q, si = self._make_block_fp8(w, 64)
        key = "model.layers.0.mlp.experts.gate_up_proj"
        deq, _, _, _ = _dequantize_quantized_shard(
            {key: q, key + "_scale_inv": si}, "cpu"
        )
        base_deq = self._dequant(q, si)

        A = torch.randn(r * E, IN) * 0.05
        B = torch.randn(OUT, r * E) * 0.05
        delta = _build_peft_layer_and_get_delta(
            A, B, {"r": r, "lora_alpha": alpha}, deq[key], is_param_wrapper=True
        )
        merged = deq[key].float() + delta.float()
        smB = peft_lora_B_to_scattermoe(B, E, r)
        kern_delta = torch.stack(
            [
                scaling * (smB[:, e * r : (e + 1) * r] @ A[e * r : (e + 1) * r, :])
                for e in range(E)
            ]
        )
        expected = base_deq.float() + kern_delta.float()
        assert (merged - expected).norm() / expected.norm() < 5e-3

    def test_swizzled_nvfp4_detected(self):
        """Swizzled-scale nvfp4 with a PADDED (unaligned) shape is auto-detected via the scale-numel
        mismatch and dequants correctly. (On-disk nvfp4 checkpoints ship NON-swizzled scales — the
        heuristic can only catch the padded case, not a same-numel reorder at block-aligned shapes;
        the default non-swizzled path is correct for real checkpoints.)"""
        pytest.importorskip("torchao")
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
        except Exception:  # pragma: no cover
            pytest.skip("NVFP4Tensor unavailable")
        from axolotl.cli.utils.lora_merge import _dequantize_quantized_shard

        torch.manual_seed(9)
        N, K = (
            32,
            64,
        )  # N<128 -> swizzle pads the scale grid, so numel differs from N*K/16
        w = torch.randn(N, K, dtype=torch.bfloat16) * 0.2
        p = (w.abs().max() / (6.0 * 448.0)).reshape(1).float().clamp(min=1e-12)
        try:
            nv = NVFP4Tensor.to_nvfp4(w, per_tensor_scale=p, is_swizzled_scales=True)
        except Exception as ex:  # pragma: no cover
            pytest.skip(f"swizzled to_nvfp4 unavailable: {ex}")
        if nv.scale.numel() == N * (K // 16):
            pytest.skip(
                "this torchao build did not pad the swizzled scale; detection N/A"
            )
        key = "model.layers.0.self_attn.q_proj.weight"
        out, did, _, _ = _dequantize_quantized_shard(
            {key: nv.qdata, key + "_scale": nv.scale, key + "_scale_2": p.reshape(())},
            "cpu",
        )
        assert did and out[key].dtype == torch.bfloat16
        assert (out[key].float() - w.float()).norm() / w.float().norm() < 0.2

    def test_modules_to_save_override_path(self):
        """_find_full_override resolves the PEFT modules_to_save layout
        (...<module>.modules_to_save.default.weight)."""
        from axolotl.cli.utils.lora_merge import _find_full_override

        w = torch.randn(6, 8)
        lora_state = {
            "base_model.model.score.modules_to_save.default.weight": w,
        }
        got = _find_full_override(lora_state, "score.weight")
        assert got is not None and torch.equal(got, w)

    def test_dequantize_mxfp4_ragged(self):
        """mxfp4 dequant: fixed 32-wide MX blocks with a trimmed ragged tail (K=40 -> 32 + 8), not a
        floor-spread."""
        from axolotl.cli.utils.lora_merge import _dequant_mxfp4, _unpack_fp4

        torch.manual_seed(10)
        N, Khalf = 4, 20  # K = 40 nibbles, nb = ceil(40/32) = 2
        packed = torch.randint(0, 255, (N, Khalf), dtype=torch.uint8)
        s = torch.randint(120, 130, (N, 2), dtype=torch.uint8)
        out = _dequant_mxfp4(packed, s, "cpu")
        assert out.shape == (N, 40)
        vals = _unpack_fp4(packed, "cpu")  # [N, 40]
        scale = torch.exp2(s.float() - 127.0)
        exp = vals.clone()
        exp[:, :32] *= scale[:, :1]
        exp[:, 32:] *= scale[:, 1:2]
        assert torch.allclose(out.float(), exp, atol=1e-3)

    def test_partial_dequant_reports_left_quantized(self):
        """A shard with a dequantizable tensor AND an unsupported quantized tensor (per-tensor fp8,
        scalar scale) must report left_quantized=True so the caller keeps quantization_config."""
        from axolotl.cli.utils.lora_merge import _dequantize_quantized_shard

        blk = torch.randn(16, 16).to(torch.float8_e4m3fn)  # block-fp8 (handled)
        pt = torch.randn(8, 8).to(
            torch.float8_e4m3fn
        )  # per-tensor fp8 (unsupported here)
        shard = {
            "a.weight": blk,
            "a.weight_scale_inv": torch.ones(1, 1),
            "b.weight": pt,
            "b.weight_scale": torch.tensor(2.0),  # scalar fp32 -> not e8m0, not block
        }
        out, did, left, _ = _dequantize_quantized_shard(shard, "cpu")
        assert did and left
        assert out["a.weight"].dtype == torch.bfloat16  # dequantized
        assert out["b.weight"].dtype == torch.float8_e4m3fn  # left as-is
        assert "b.weight_scale" in out  # its scale not dropped
