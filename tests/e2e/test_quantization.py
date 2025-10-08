"""
Tests for axolotl.utils.quantization
"""

import pytest
import torch
from torch import nn
from torchao.quantization import LinearActivationQuantizedTensor
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt4WeightConfig,
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
)
from axolotl.utils.schemas.enums import TorchAOQuantDType
from axolotl.utils.schemas.quantization import QATConfig

from tests.e2e.utils import (
    require_torch_2_8_0,
    requires_cuda_ge_8_9,
    requires_sm_ge_100,
)


@pytest.fixture()
def model():
    dummy_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
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
        Int8DynamicActivationInt4WeightConfig,
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
        LinearActivationQuantizedTensor,
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
            assert (
                model.model.embed_tokens.weight_fake_quantizer.config.dtype
                == weight_dtype.value
            )
            if group_size:
                assert (
                    model.model.embed_tokens.weight_fake_quantizer.config.group_size
                    == group_size
                )

        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                assert isinstance(child, FakeQuantizedLinear)
                assert hasattr(child, "weight_fake_quantizer")
                assert child.weight_fake_quantizer.config.dtype == weight_dtype.value
                if group_size:
                    assert child.weight_fake_quantizer.config.group_size == group_size
                if activation_dtype:
                    assert hasattr(child, "activation_fake_quantizer")
                    assert (
                        child.activation_fake_quantizer.config.dtype
                        == activation_dtype.value
                    )
                else:
                    assert child.activation_fake_quantizer is None

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
        assert model.model.embed_tokens.weight_fake_quantizer.enabled
        assert isinstance(model.lm_head, FakeQuantizedLinear)
        assert model.lm_head.weight_fake_quantizer.enabled

        qat_callback = QATCallback(cfg)

        # simulate first training step
        qat_callback.on_step_begin(
            args=None,
            state=trainer_state,
            control=None,
            model=model,
        )

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
        assert model.model.embed_tokens.weight_fake_quantizer.enabled
        assert isinstance(model.lm_head, FakeQuantizedLinear)
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
        assert model.model.embed_tokens.weight_fake_quantizer.enabled
        assert model.lm_head.weight_fake_quantizer.enabled
