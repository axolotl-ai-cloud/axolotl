"""
Tests for axolotl.utils.quantization
"""

import pytest
import torch

from axolotl.utils.callbacks.qat import QATCallback
from axolotl.utils.quantization import (
    get_quantization_config,
    qat_prepare_or_convert_model,
    quantize_model_for_ptq,
)
from axolotl.utils.schemas.enums import TorchAOQuantDType
from axolotl.utils.schemas.quantization import QATConfig
from torch import nn
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
)
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
)
from transformers import AutoModelForCausalLM
from transformers.trainer_callback import TrainerState

from tests.e2e.utils import require_torch_2_6_0


@pytest.fixture()
def model():
    dummy_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        device_map="auto",
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
    (TorchAOQuantDType.int4, None, 4, Int4WeightOnlyConfig, {"group_size": 4, "version": 2}),
    (
        TorchAOQuantDType.int4,
        TorchAOQuantDType.int8,
        None,
        Int8DynamicActivationInt4WeightConfig,
        {},
    ),
]

ptq_test_cases = [
    # weight_dtype, activation_dtype, group_size, quantize_embedding, expected_exception
    # Supported cases
    (TorchAOQuantDType.int4, None, 4, True, None),
    (TorchAOQuantDType.int4, TorchAOQuantDType.int8, 8, False, None),
    (TorchAOQuantDType.int4, None, None, False, ValueError),  # group_size required for int4 weight only
    # Deprecated configs that should raise ValueError
    (TorchAOQuantDType.int8, None, 8, False, ValueError),  # Int8WeightOnlyConfig deprecated
    (TorchAOQuantDType.int4, TorchAOQuantDType.int4, 8, False, ValueError),  # Int4DynamicActivationInt4WeightConfig deprecated
    (TorchAOQuantDType.int8, TorchAOQuantDType.int8, 8, True, ValueError),  # Int8DynamicActivationInt8WeightConfig deprecated
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
        config = get_quantization_config(weight_dtype, activation_dtype, group_size)

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
    ):
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

    @pytest.mark.parametrize(
        "weight_dtype", [TorchAOQuantDType.int4]
    )
    @pytest.mark.parametrize(
        "activation_dtype", [None, TorchAOQuantDType.int8]
    )
    @pytest.mark.parametrize("group_size", [8, 16])
    @pytest.mark.parametrize("quantize_embedding", [False, True])
    @require_torch_2_6_0
    def test_prepare_model_for_qat(
        self, model, weight_dtype, activation_dtype, group_size, quantize_embedding
    ):
        qat_prepare_or_convert_model(
            model, "prepare", weight_dtype, group_size, activation_dtype, quantize_embedding
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

    @require_torch_2_6_0
    def test_convert_qat_model_for_ptq(self, model):
        config = QATConfig(
            weight_dtype="int4",
            activation_dtype="int8",
            group_size=8,
            quantize_embedding=True,
        )

        # quantize model for qat
        qat_prepare_or_convert_model(
            model,
            "prepare",
            config.weight_dtype,
            config.group_size,
            config.activation_dtype,
            config.quantize_embedding,
        )

        assert isinstance(model.model.embed_tokens, FakeQuantizedEmbedding)
        assert isinstance(model.lm_head, FakeQuantizedLinear)

        # apply conversion
        qat_prepare_or_convert_model(
            model,
            "convert",
            config.weight_dtype,
            config.group_size,
            config.activation_dtype,
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

    @require_torch_2_6_0
    def test_qat_callback_fake_quant_after_n_steps(self, model, trainer_state):
        cfg = QATConfig(
            weight_dtype="int4",
            activation_dtype="int8",
            group_size=8,
            quantize_embedding=True,
            fake_quant_after_n_steps=100,
        )

        qat_prepare_or_convert_model(
            model,
            "prepare",
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
    def test_qat_callback_fake_quant_after_n_steps_is_none(self, model, trainer_state):
        cfg = QATConfig(
            weight_dtype="int4",
            activation_dtype="int8",
            group_size=8,
            quantize_embedding=True,
            fake_quant_after_n_steps=None,
        )

        qat_prepare_or_convert_model(
            model,
            "prepare",
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
