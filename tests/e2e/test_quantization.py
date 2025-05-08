import pytest
import torch
from torchao.experimental.quant_api import Int8DynamicActivationIntxWeightConfig
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.quant_api import (
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
)
from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
from transformers import AutoModelForCausalLM

from axolotl.utils.quantization import (
    IntxWeightOnlyConfig,
    get_ptq_config,
    quantize_model_for_qat,
    quantize_model_for_ptq,
)
from axolotl.utils.schemas.qat import TorchIntDType

ptq_config_test_cases = [
    # weight_dtype, activation_dtype, group_size, expected_type, expected_params
    (
        TorchIntDType.uint4,
        None,
        None,
        UIntXWeightOnlyConfig,
        {"dtype": torch.uint4, "group_size": None},
    ),
    (TorchIntDType.int8, None, None, Int8WeightOnlyConfig, {"group_size": None}),
    (TorchIntDType.int4, None, 4, Int4WeightOnlyConfig, {"group_size": 4}),
    (
        TorchIntDType.int4,
        TorchIntDType.int4,
        None,
        Int4DynamicActivationInt4WeightConfig,
        {},
    ),
    (
        TorchIntDType.int8,
        TorchIntDType.int8,
        None,
        Int8DynamicActivationInt8WeightConfig,
        {},
    ),
    (
        TorchIntDType.int4,
        TorchIntDType.int8,
        None,
        Int8DynamicActivationIntxWeightConfig,
        {
            "weight_dtype": torch.int4,
            "granularity": PerAxis(0),
            "has_weight_zeros": False,
        },
    ),
]


class TestQuantization:

    @pytest.mark.parametrize(
        "weight_dtype,activation_dtype,group_size,expected_type,expected_params",
        ptq_config_test_cases,
    )
    def test_get_ptq_config(
        self, weight_dtype, activation_dtype, group_size, expected_type, expected_params
    ):
        config = get_ptq_config(weight_dtype, activation_dtype, group_size)

        assert isinstance(config, expected_type)

        for param_name, param_value in expected_params.items():
            if isinstance(param_value, (PerAxis, PerGroup)):
                if isinstance(param_value, PerAxis):
                    assert isinstance(getattr(config, param_name), PerAxis)
                    assert getattr(config, param_name).axis == param_value.axis
                else:
                    assert isinstance(getattr(config, param_name), PerGroup)
                    assert (
                        getattr(config, param_name).group_size == param_value.group_size
                    )
            else:
                assert getattr(config, param_name) == param_value

    @pytest.mark.parametrize(
        "weight_dtype", [TorchIntDType.int8, TorchIntDType.int4]
    )
    @pytest.mark.parametrize(
        "activation_dtype", [None, TorchIntDType.int4, TorchIntDType.int8]
    )
    @pytest.mark.parametrize("group_size", [4, 8])
    @pytest.mark.parametrize("quantize_embedding", [False, True])
    def test_quantize_model_for_qat(
        self, weight_dtype, activation_dtype, group_size, quantize_embedding
    ):
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        quantize_model_for_qat(
            model, weight_dtype, group_size, activation_dtype, quantize_embedding
        )
        if quantize_embedding:
            assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
            assert hasattr(model.model.embed_tokens, "weight_fake_quantizer")
            assert model.model.embed_tokens.weight_fake_quantizer.config.dtype == weight_dtype.value
            assert model.model.embed_tokens.weight_fake_quantizer.config.group_size == group_size

        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                assert isinstance(child, FakeQuantizedLinear)
                assert hasattr(child, "weight_fake_quantizer")
                assert child.weight_fake_quantizer.config.dtype == weight_dtype.value
                assert child.weight_fake_quantizer.config.group_size == group_size
                if activation_dtype:
                    assert hasattr(child, "activation_fake_quantizer")
                    assert (
                        child.activation_fake_quantizer.config.dtype ==
                        activation_dtype.value
                    )
                else:
                    assert child.activation_fake_quantizer is None

    @pytest.mark.parametrize(
        "weight_dtype", [TorchIntDType.int8, TorchIntDType.int4,
                         TorchIntDType.uint4, TorchIntDType.uint7]
    )
    @pytest.mark.parametrize(
        "activation_dtype", [None, None, None, None, None]
    )
    @pytest.mark.parametrize("group_size", [4, 8])
    @pytest.mark.parametrize("quantize_embedding", [False, True])
    def test_quantize_model_for_ptq(self, weight_dtype, activation_dtype, group_size, quantize_embedding):
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        quantize_model_for_ptq(model, weight_dtype, group_size,
                               activation_dtype, quantize_embedding)
        if quantize_embedding:
            assert isinstance(model.model.embed_tokens.weight, AffineQuantizedTensor)
        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                if activation_dtype:
                    assert isinstance(child.weight, LinearActivationQuantizedTensor)
                else:
                    assert isinstance(child.weight, AffineQuantizedTensor)
