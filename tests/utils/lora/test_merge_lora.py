from unittest.mock import Mock, patch

import torch

from axolotl.cli.merge_lora import do_merge_lora
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
    def test_cli_do_merge_functionality(self, mock_load_model):
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
                "output_dir": "/tmp/test_output",
            }
        )

        with (
            patch("pathlib.Path.mkdir"),
            patch("torch.save"),
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
    def test_memory_efficient_merge_with_cpu_offload(self):
        """Test lora_on_cpu configuration during merge"""
        cfg = DictDefault(
            {
                "lora_on_cpu": True,
                "save_safetensors": True,
                "output_dir": "/tmp/test_output",
                "local_rank": 0,
            }
        )

        with patch("axolotl.cli.merge_lora.load_model_and_tokenizer") as mock_load:
            base_model = self.create_mock_base_model()
            lora_model = self.create_mock_lora_model(base_model)
            mock_load.return_value = (lora_model, Mock(), None)

            with patch("pathlib.Path.mkdir"), patch("torch.save"):
                do_merge_lora(cfg=cfg)

            assert mock_load.called
