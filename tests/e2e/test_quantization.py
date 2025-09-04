"""
Tests for axolotl.utils.quantization
"""

import pytest
import torch
from torch import nn
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
)
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.quant_api import (
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
)
from transformers import AutoModelForCausalLM
from transformers.trainer_callback import TrainerState

from axolotl.utils.callbacks.qat import QATCallback
from axolotl.utils.quantization import (
    convert_qat_model_for_ptq,
    get_ptq_config,
    prepare_model_for_qat,
    quantize_model_for_ptq,
)
from axolotl.utils.schemas.enums import TorchIntDType
from axolotl.utils.schemas.quantization import QATConfig

from tests.e2e.utils import require_torch_2_6_0


@pytest.fixture()
def model():
    dummy_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    with torch.device(dummy_model.device):
        dummy_model.model.embed_tokens = torch.nn.Embedding(
            dummy_model.model.embed_tokens.weight.shape[0],
            dummy_model.model.embed_tokens.weight.shape[1],
            dtype=dummy_model.model.embed_tokens.weight.dtype,
        )
    return dummy_model


ptq_config_test_cases = [
    # weight_dtype, activation_dtype, group_size, expected_type, expected_params
    (
        TorchIntDType.uint4,
        None,
        None,
        UIntXWeightOnlyConfig,
        {"dtype": torch.uint4, "group_size": None},
    ),
    (TorchIntDType.int8, None, 32, Int8WeightOnlyConfig, {"group_size": 32}),
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
]

ptq_test_cases = [
    # weight_dtype, activation_dtype, group_size, quantize_embedding, expected_exception
    (TorchIntDType.int8, None, 8, False, None),
    (TorchIntDType.int4, None, 4, True, None),
    (TorchIntDType.uint4, None, 8, False, None),
    (TorchIntDType.int4, TorchIntDType.int4, 8, False, None),
    (TorchIntDType.int8, TorchIntDType.int8, 8, True, None),
    (TorchIntDType.int8, None, None, False, ValueError),
    (TorchIntDType.int4, None, None, False, ValueError),
]


class TestQuantization:
    """
    Test quantization utilities
    """

    @pytest.mark.parametrize(
        "weight_dtype,activation_dtype,group_size,expected_type,expected_params",
        ptq_config_test_cases,
    )
    @require_torch_2_6_0
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
        "weight_dtype", [TorchIntDType.int8, TorchIntDType.int4, TorchIntDType.uint4]
    )
    @pytest.mark.parametrize(
        "activation_dtype", [None, TorchIntDType.int4, TorchIntDType.int8]
    )
    @pytest.mark.parametrize("group_size", [4, 8])
    @pytest.mark.parametrize("quantize_embedding", [False, True])
    @require_torch_2_6_0
    def test_prepare_model_for_qat(
        self, model, weight_dtype, activation_dtype, group_size, quantize_embedding
    ):  # pylint: disable=redefined-outer-name
        prepare_model_for_qat(
            model, weight_dtype, group_size, activation_dtype, quantize_embedding
        )
        if quantize_embedding:
            assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
            assert hasattr(model.model.embed_tokens, "weight_fake_quantizer")
            assert (
                model.model.embed_tokens.weight_fake_quantizer.config.dtype
                == weight_dtype.value
            )
            assert (
                model.model.embed_tokens.weight_fake_quantizer.config.group_size
                == group_size
            )

        for child in list(model.children()):
            if isinstance(child, torch.nn.Linear):
                assert isinstance(child, FakeQuantizedLinear)
                assert hasattr(child, "weight_fake_quantizer")
                assert child.weight_fake_quantizer.config.dtype == weight_dtype.value
                assert child.weight_fake_quantizer.config.group_size == group_size
                if activation_dtype:
                    assert hasattr(child, "activation_fake_quantizer")
                    assert (
                        child.activation_fake_quantizer.config.dtype
                        == activation_dtype.value
                    )
                else:
                    assert child.activation_fake_quantizer is None

    @pytest.mark.parametrize(
        "weight_dtype,activation_dtype,group_size,quantize_embedding,expected_exception",
        ptq_test_cases,
    )
    @require_torch_2_6_0
    def test_quantize_model_for_ptq(
        self,
        model,
        weight_dtype,
        activation_dtype,
        group_size,
        quantize_embedding,
        expected_exception,
    ):  # pylint: disable=redefined-outer-name
        if expected_exception:
            with pytest.raises(expected_exception):
                quantize_model_for_ptq(
                    model,
                    weight_dtype,
                    group_size,
                    activation_dtype,
                    quantize_embedding,
                )
        else:
            quantize_model_for_ptq(
                model, weight_dtype, group_size, activation_dtype, quantize_embedding
            )
            if quantize_embedding:
                assert isinstance(
                    model.model.embed_tokens.weight, AffineQuantizedTensor
                ), "Embedding weight should be quantized"
            for child in list(model.children()):
                if isinstance(child, torch.nn.Linear):
                    if activation_dtype:
                        assert isinstance(
                            child.weight, LinearActivationQuantizedTensor
                        ), "Linear weight should be quantized with activation quantization"
                    else:
                        assert isinstance(
                            child.weight, AffineQuantizedTensor
                        ), "Linear weight should be quantized without activation quantization"


class TestQuantizationCallback:
    """
    Test QATCallback
    """

    @pytest.fixture()
    def trainer_state(self):
        return TrainerState(
            global_step=0,
        )

    @require_torch_2_6_0
    def test_qat_callback_fake_quant_after_n_steps(
        self, model, trainer_state
    ):  # pylint: disable=redefined-outer-name
        cfg = QATConfig(
            weight_dtype="int8",
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

    @require_torch_2_6_0
    def test_qat_callback_fake_quant_after_n_steps_is_none(
        self, model, trainer_state
    ):  # pylint: disable=redefined-outer-name
        cfg = QATConfig(
            weight_dtype="int8",
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


class TestConvertQATModelForPTQ:
    """
    Test convert_qat_model_for_ptq
    """

    @require_torch_2_6_0
    def test_convert_qat_model_for_ptq(
        self, model
    ):  # pylint: disable=redefined-outer-name
        config = QATConfig(
            weight_dtype="int8",
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
        convert_qat_model_for_ptq(
            model,
            quantize_embedding=config.quantize_embedding,
        )
        # ensure modules have been swapped out
        assert not isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
        assert not isinstance(model.lm_head, FakeQuantizedLinear)

        # ensure weights have been quantized
        assert isinstance(model.model.embed_tokens.weight, nn.Parameter)
        assert isinstance(model.lm_head.weight, nn.Parameter)
