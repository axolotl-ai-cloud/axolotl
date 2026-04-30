"""
Tests for axolotl.utils.quantization
"""

import pytest
import torch
from torch import nn
from torchao.quantization import IntxUnpackedToInt8Tensor
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
)
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor
from transformers import AutoModelForCausalLM
from transformers.trainer_callback import TrainerState

from axolotl.utils.callbacks.qat import QATCallback
from axolotl.utils.quantization import (
    convert_qat_model,
    get_quantization_config,
    prepare_model_for_qat,
    quantize_model,
    save_quantized_model,
)
from axolotl.utils.schemas.enums import TorchAOQuantDType
from axolotl.utils.schemas.quantization import QATConfig

from tests.e2e.utils import (
    require_torch_2_8_0,
    requires_cuda_ge_8_9,
    requires_sm_ge_100,
)


def _get_fake_quant_config_dtype(config):
    """Get the weight dtype from a fake quantize config, handling different config types."""
    if hasattr(config, "dtype"):
        return config.dtype
    # Int4WeightFakeQuantizeConfig doesn't have .dtype — weight is always int4
    return torch.int4


@pytest.fixture()
def model():
    dummy_model = AutoModelForCausalLM.from_pretrained(
        "axolotl-ai-co/tiny-qwen2-129m",
        device_map="auto",
        dtype=torch.bfloat16,
    )
    with torch.device(dummy_model.device):
        dummy_model.model.embed_tokens = torch.nn.Embedding(
            dummy_model.model.embed_tokens.weight.shape[0],
            dummy_model.model.embed_tokens.weight.shape[1],
            dtype=dummy_model.model.embed_tokens.weight.dtype,
        )
    yield dummy_model
    del dummy_model


ptq_config_test_cases = [
    # weight_dtype, activation_dtype, group_size, expected_type
    (
        TorchAOQuantDType.int4,
        TorchAOQuantDType.int8,
        None,
        Int8DynamicActivationIntxWeightConfig,
    ),
    (
        TorchAOQuantDType.float8_e4m3fn,
        TorchAOQuantDType.float8_e4m3fn,
        None,
        Float8DynamicActivationFloat8WeightConfig,
    ),
    (
        TorchAOQuantDType.int4,
        TorchAOQuantDType.float8_e4m3fn,
        None,
        Float8DynamicActivationInt4WeightConfig,
    ),
]

ptq_test_cases = [
    # weight_dtype, activation_dtype, group_size, quantize_embedding, expected_exception, expected_tensor_class
    (TorchAOQuantDType.int4, None, 4, True, None, Int4Tensor),
    (
        TorchAOQuantDType.int4,
        TorchAOQuantDType.int8,
        8,
        False,
        None,
        IntxUnpackedToInt8Tensor,
    ),
    # (
    #     TorchAOQuantDType.int4,
    #     TorchAOQuantDType.float8_e4m3fn,
    #     None,
    #     False,
    #     None,
    #     Int4Tensor,
    # ),
    (TorchAOQuantDType.int4, None, None, False, None, Int4Tensor),
    # Deprecated configs
    (TorchAOQuantDType.int8, None, 8, False, ValueError, None),
    (TorchAOQuantDType.int4, TorchAOQuantDType.int4, 8, False, ValueError, None),
    (TorchAOQuantDType.int8, TorchAOQuantDType.int8, 8, True, ValueError, None),
]


class TestQuantization:
    """
    Test quantization utilities
    """

    @pytest.mark.parametrize(
        "weight_dtype,activation_dtype,group_size,expected_type",
        ptq_config_test_cases,
    )
    @requires_cuda_ge_8_9
    @require_torch_2_8_0
    def test_get_ptq_config(
        self, weight_dtype, activation_dtype, group_size, expected_type
    ):
        config = get_quantization_config(weight_dtype, activation_dtype, group_size)
        assert isinstance(config, expected_type)

    @require_torch_2_8_0
    @requires_sm_ge_100
    def test_get_ptq_config_mxfp4(self):
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig

        config = get_quantization_config(TorchAOQuantDType.mxfp4, None, 32)
        assert isinstance(config, MXDynamicActivationMXWeightConfig)
        assert config.weight_dtype == torch.float4_e2m1fn_x2
        assert config.block_size == 32

    @require_torch_2_8_0
    @requires_sm_ge_100
    def test_get_ptq_config_mxfp4_invalid_group_size(self):
        with pytest.raises(
            ValueError, match="MXFP4 quantization must use a block_size"
        ):
            get_quantization_config(TorchAOQuantDType.mxfp4, None, 16)

    @requires_cuda_ge_8_9
    @require_torch_2_8_0
    def test_get_ptq_config_int4_weight_only(self):
        from torchao.quantization.quant_api import Int4WeightOnlyConfig

        config = get_quantization_config(TorchAOQuantDType.int4, None, 4)
        assert isinstance(config, Int4WeightOnlyConfig)

    @pytest.mark.parametrize(
        "weight_dtype,activation_dtype,group_size,quantize_embedding,expected_exception,expected_tensor_class",
        ptq_test_cases,
    )
    @requires_cuda_ge_8_9
    @require_torch_2_8_0
    def test_quantize_model_for_ptq(
        self,
        model,
        weight_dtype,
        activation_dtype,
        group_size,
        quantize_embedding,
        expected_exception,
        expected_tensor_class,
    ):
        # TODO: add mslk-cuda as a CI dependency once pytorch 2.10.x is available
        # (see https://pypi.org/project/mslk-cuda/)
        if expected_tensor_class is Int4Tensor and activation_dtype is None:
            try:
                from torchao.quantization.quantize_.workflows.int4.int4_tensor import (
                    int4_row_quantize_zp,
                )

                if int4_row_quantize_zp is None:
                    pytest.skip("Int4Tensor requires mslk >= 1.0.0")
            except ImportError:
                pytest.skip("Int4Tensor requires mslk >= 1.0.0")
        if expected_exception:
            with pytest.raises(expected_exception):
                quantize_model(
                    model,
                    weight_dtype,
                    group_size,
                    activation_dtype,
                    quantize_embedding,
                )
        else:
            quantize_model(
                model, weight_dtype, group_size, activation_dtype, quantize_embedding
            )
            if quantize_embedding:
                assert isinstance(
                    model.model.embed_tokens.weight, expected_tensor_class
                ), "Embedding weight should be quantized"
            for child in list(model.children()):
                if isinstance(child, torch.nn.Linear):
                    assert isinstance(child.weight, expected_tensor_class)

    @require_torch_2_8_0
    @requires_sm_ge_100
    def test_quantize_model_for_ptq_fp8(
        self,
        model,
    ):
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
            QuantizeTensorToFloat8Kwargs,
        )

        quantize_model(
            model,
            TorchAOQuantDType.float8_e4m3fn,
            None,
            TorchAOQuantDType.float8_e4m3fn,
        )
        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                assert isinstance(child.weight, Float8Tensor)
                assert child.weight.act_quant_kwargs is not None and isinstance(
                    child.weight.act_quant_kwargs, QuantizeTensorToFloat8Kwargs
                )

    @require_torch_2_8_0
    @requires_sm_ge_100
    def test_quantize_model_for_ptq_nvfp4(
        self,
        model,
    ):
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            QuantizeTensorToNVFP4Kwargs,
        )

        quantize_model(model, TorchAOQuantDType.nvfp4, 16, TorchAOQuantDType.nvfp4)
        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                assert isinstance(child.weight, NVFP4Tensor)
                assert child.weight.act_quant_kwargs is not None and isinstance(
                    child.weight.act_quant_kwargs, QuantizeTensorToNVFP4Kwargs
                )

    @pytest.mark.parametrize(
        "weight_dtype,activation_dtype,group_size,quantize_embedding",
        [
            (TorchAOQuantDType.int4, None, 8, False),
            (TorchAOQuantDType.int4, None, 16, True),
            (TorchAOQuantDType.int4, TorchAOQuantDType.int8, 8, False),
            (TorchAOQuantDType.int4, TorchAOQuantDType.int8, 16, True),
            (
                TorchAOQuantDType.float8_e4m3fn,
                TorchAOQuantDType.float8_e4m3fn,
                None,
                False,
            ),
            (TorchAOQuantDType.int4, TorchAOQuantDType.float8_e4m3fn, None, True),
        ],
    )
    @require_torch_2_8_0
    @requires_cuda_ge_8_9
    def test_prepare_model_for_qat(
        self, model, weight_dtype, activation_dtype, group_size, quantize_embedding
    ):
        prepare_model_for_qat(
            model,
            weight_dtype,
            group_size,
            activation_dtype,
            quantize_embedding,
        )
        if quantize_embedding:
            assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
            assert hasattr(model.model.embed_tokens, "weight_fake_quantizer")
            embed_config = model.model.embed_tokens.weight_fake_quantizer.config
            assert _get_fake_quant_config_dtype(embed_config) == weight_dtype.value
            if group_size:
                assert embed_config.group_size == group_size

        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                assert isinstance(child, FakeQuantizedLinear)
                assert hasattr(child, "weight_fake_quantizer")
                w_config = child.weight_fake_quantizer.config
                assert _get_fake_quant_config_dtype(w_config) == weight_dtype.value
                if group_size:
                    assert w_config.group_size == group_size
                if activation_dtype:
                    assert hasattr(child, "activation_fake_quantizer")
                    a_config = child.activation_fake_quantizer.config
                    assert (
                        _get_fake_quant_config_dtype(a_config) == activation_dtype.value
                    )
                else:
                    assert child.activation_fake_quantizer is None

    @pytest.mark.parametrize(
        "weight_dtype,activation_dtype,group_size,quantize_embedding",
        [
            (TorchAOQuantDType.mxfp4, None, 32, False),
        ],
    )
    @require_torch_2_8_0
    @requires_sm_ge_100
    def test_prepare_model_for_qat_mxfp4(
        self, model, weight_dtype, activation_dtype, group_size, quantize_embedding
    ):
        prepare_model_for_qat(
            model,
            weight_dtype,
            group_size,
            activation_dtype,
            quantize_embedding,
        )

        from torchao.prototype.qat import MXFakeQuantizedLinear

        if quantize_embedding:
            assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
            assert hasattr(model.model.embed_tokens, "weight_fake_quantizer")

        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                assert isinstance(child, MXFakeQuantizedLinear)
                assert hasattr(child, "weight_config")

    @require_torch_2_8_0
    @requires_cuda_ge_8_9
    def test_convert_qat_model(self, model):
        config = QATConfig(
            weight_dtype="int4",
            activation_dtype="int8",
            group_size=8,
            quantize_embedding=True,
        )

        # quantize model for qat
        prepare_model_for_qat(
            model,
            config.weight_dtype,
            config.group_size,
            config.activation_dtype,
            config.quantize_embedding,
        )

        assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
        assert isinstance(model.lm_head, FakeQuantizedLinear)

        # apply conversion
        convert_qat_model(
            model,
            config.quantize_embedding,
        )
        # ensure modules have been swapped out
        assert not isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
        assert not isinstance(model.lm_head, FakeQuantizedLinear)

        # ensure weights have been quantized
        assert isinstance(model.model.embed_tokens.weight, nn.Parameter)
        assert isinstance(model.lm_head.weight, nn.Parameter)


class TestMXQuantizeSaveLoad:
    """Tests for MX format (mxfp4) quantize-save-load round-trip via save_pretrained.

    Uses a tiny HF model built from config (no download) so tests exercise the
    real save_pretrained / from_pretrained code path — the same one the CLI uses.
    MX format models are saved with safe_serialization=False (torch.save) because
    MXTensor does not yet support safetensors serialization.
    """

    @staticmethod
    def _make_tiny_model():
        """Build a minimal HF causal-LM that can be quantized on CPU."""
        from transformers import Qwen2Config

        config = Qwen2Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            max_position_embeddings=64,
            torch_dtype="bfloat16",
        )
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
        return model

    @require_torch_2_8_0
    def test_mxfp4_quantize_save_pretrained(self, tmp_path):
        """quantize_model(mxfp4) -> save_pretrained -> from_pretrained round-trip."""
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        model = self._make_tiny_model()
        original_keys = set(model.state_dict().keys())

        quantize_model(model, TorchAOQuantDType.mxfp4, 32)

        # Weights should be MXTensor after quantization
        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert isinstance(module.weight, MXTensor)

        # Model should be flagged for MX-style save
        assert getattr(model, "_is_mx_quantized", False)

        # save_pretrained with safe_serialization=False (torch.save path)
        save_dir = str(tmp_path / "mxfp4_model")
        save_quantized_model(model, save_dir)

        # Verify checkpoint files were written
        import glob

        assert glob.glob(f"{save_dir}/*.bin") or glob.glob(f"{save_dir}/**/*.bin")

        # from_pretrained should load without error
        loaded = AutoModelForCausalLM.from_pretrained(
            save_dir, torch_dtype=torch.bfloat16
        )
        loaded_keys = set(loaded.state_dict().keys())
        assert original_keys == loaded_keys, (
            f"Key mismatch: missing={original_keys - loaded_keys}, "
            f"extra={loaded_keys - original_keys}"
        )

    @require_torch_2_8_0
    def test_mxfp4_is_mx_flag_set(self):
        """quantize_model sets _is_mx_quantized for MX configs."""
        model = self._make_tiny_model()
        quantize_model(model, TorchAOQuantDType.mxfp4, 32)
        assert getattr(model, "_is_mx_quantized", False)

    @require_torch_2_8_0
    @requires_cuda_ge_8_9
    def test_non_mx_uses_torchao_quantizer(self):
        """Non-MX quantization attaches TorchAoHfQuantizer, not _is_mx_quantized."""
        model = self._make_tiny_model()
        try:
            quantize_model(model, TorchAOQuantDType.int4, group_size=32)
        except ImportError:
            pytest.skip("int4 quantization requires mslk >= 1.0.0")
        assert not getattr(model, "_is_mx_quantized", False)
        assert hasattr(model, "hf_quantizer")

    @require_torch_2_8_0
    def test_mxfp4_qat_then_ptq_save_pretrained(self, tmp_path):
        """Full QAT -> convert -> PTQ -> save_pretrained -> from_pretrained."""
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.qat import MXFakeQuantizedLinear

        model = self._make_tiny_model()
        original_keys = set(model.state_dict().keys())

        # QAT preparation
        prepare_model_for_qat(model, TorchAOQuantDType.mxfp4, 32)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert isinstance(module, MXFakeQuantizedLinear)

        # Convert QAT back to normal linear
        convert_qat_model(model)

        # PTQ quantize
        quantize_model(model, TorchAOQuantDType.mxfp4, 32)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert isinstance(module.weight, MXTensor)

        # save_pretrained round-trip
        save_dir = str(tmp_path / "mxfp4_qat_model")
        save_quantized_model(model, save_dir)

        loaded = AutoModelForCausalLM.from_pretrained(
            save_dir, torch_dtype=torch.bfloat16
        )
        loaded_keys = set(loaded.state_dict().keys())
        assert original_keys == loaded_keys


class TestQuantizationCallback:
    """
    Test QATCallback
    """

    @pytest.fixture()
    def trainer_state(self):
        return TrainerState(
            global_step=0,
        )

    @require_torch_2_8_0
    def test_qat_callback_fake_quant_after_n_steps(self, model, trainer_state):
        cfg = QATConfig(
            weight_dtype="int4",
            activation_dtype="int8",
            group_size=8,
            quantize_embedding=True,
            fake_quant_after_n_steps=100,
        )

        prepare_model_for_qat(
            model,
            cfg.weight_dtype,
            cfg.group_size,
            cfg.activation_dtype,
            cfg.quantize_embedding,
        )

        # ensure model has been quantized
        assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
        assert isinstance(model.lm_head, FakeQuantizedLinear)

        # Only test enable/disable toggling if the fake quantizer supports it
        # (Int4WeightFakeQuantizer does not have an 'enabled' attribute)
        supports_toggle = hasattr(
            model.model.embed_tokens.weight_fake_quantizer, "enabled"
        )
        if supports_toggle:
            assert model.model.embed_tokens.weight_fake_quantizer.enabled
            assert model.lm_head.weight_fake_quantizer.enabled

        qat_callback = QATCallback(cfg)

        # simulate first training step
        qat_callback.on_step_begin(
            args=None,
            state=trainer_state,
            control=None,
            model=model,
        )

        if supports_toggle:
            # quantization should have been disabled
            assert not model.model.embed_tokens.weight_fake_quantizer.enabled
            assert not model.lm_head.weight_fake_quantizer.enabled

        trainer_state.global_step = 100
        qat_callback.on_step_begin(
            args=None,
            state=trainer_state,
            control=None,
            model=model,
        )

        if supports_toggle:
            # quantization should have been enabled
            assert model.model.embed_tokens.weight_fake_quantizer.enabled
            assert model.lm_head.weight_fake_quantizer.enabled

    @require_torch_2_8_0
    def test_qat_callback_fake_quant_after_n_steps_is_none(self, model, trainer_state):
        cfg = QATConfig(
            weight_dtype="int4",
            activation_dtype="int8",
            group_size=8,
            quantize_embedding=True,
            fake_quant_after_n_steps=None,
        )

        prepare_model_for_qat(
            model,
            cfg.weight_dtype,
            cfg.group_size,
            cfg.activation_dtype,
            cfg.quantize_embedding,
        )

        # ensure model has been quantized
        assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
        assert isinstance(model.lm_head, FakeQuantizedLinear)
        if hasattr(model.model.embed_tokens.weight_fake_quantizer, "enabled"):
            assert model.model.embed_tokens.weight_fake_quantizer.enabled
            assert model.lm_head.weight_fake_quantizer.enabled

        qat_callback = QATCallback(cfg)
        # simulate first training step
        qat_callback.on_step_begin(
            args=None,
            state=trainer_state,
            control=None,
            model=model,
        )

        # quantization should be enabled from the get-go
        if hasattr(model.model.embed_tokens.weight_fake_quantizer, "enabled"):
            assert model.model.embed_tokens.weight_fake_quantizer.enabled
            assert model.lm_head.weight_fake_quantizer.enabled
