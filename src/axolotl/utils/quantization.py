"""
Utilities for quantization including QAT and PTQ using torchao.
"""

import torch
from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    FromIntXQuantizationAwareTrainingConfig,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization.quant_api import (
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
    _is_linear,
)

from axolotl.utils.schemas.enums import TorchIntDType


def get_ptq_config(
    weight_dtype: TorchIntDType,
    activation_dtype: TorchIntDType | None = None,
    group_size: int | None = None,
) -> AOBaseConfig:
    """
    This function is used to build a post-training quantization config.

    Args:
        weight_dtype: The dtype to use for weight quantization.
        activation_dtype: The dtype to use for activation quantization.
        group_size: The group size to use for weight quantization.

    Returns:
        The post-training quantization config.

    Raises:
        ValueError: If the activation dtype is not specified and the weight dtype is not int8 or int4,
            or if the group size is not specified for int8 or int4 weight only quantization.
    """
    if activation_dtype is None:
        if not weight_dtype.value.is_signed:  # type: ignore[attr-defined,union-attr]
            return UIntXWeightOnlyConfig(
                dtype=weight_dtype.value,
                group_size=group_size,
                set_inductor_config=False,
            )
        if weight_dtype == TorchIntDType.int8:
            if group_size is None:
                raise ValueError(
                    "group_size must be specified for int8 weight only quantization"
                )
            return Int8WeightOnlyConfig(
                group_size=group_size,
            )
        if weight_dtype == TorchIntDType.int4:
            if group_size is None:
                raise ValueError(
                    "group_size must be specified for int4 weight only quantization"
                )
            return Int4WeightOnlyConfig(
                group_size=group_size,
            )
    if activation_dtype == TorchIntDType.int4 and weight_dtype == TorchIntDType.int4:
        return Int4DynamicActivationInt4WeightConfig()
    if activation_dtype == TorchIntDType.int8 and weight_dtype == TorchIntDType.int8:
        return Int8DynamicActivationInt8WeightConfig()
    if activation_dtype == TorchIntDType.int8 and weight_dtype == TorchIntDType.int4:
        return Int8DynamicActivationInt4WeightConfig()
    raise ValueError(
        f"Invalid activation/weight dtype combination: {activation_dtype}/{weight_dtype}"
    )


def prepare_model_for_qat(
    model,
    weight_dtype: TorchIntDType,
    group_size: int,
    activation_dtype: TorchIntDType | None = None,
    quantize_embedding: bool = False,
):
    """
    This function is used to prepare a model for QAT by swapping the model's linear
    layers with fake quantized linear layers, and optionally the embedding weights with
    fake quantized embedding weights.

    Args:
        model: The model to quantize.
        weight_dtype: The dtype to use for weight quantization.
        group_size: The group size to use for weight quantization.
        activation_dtype: The dtype to use for activation quantization.
        quantize_embedding: Whether to quantize the model's embedding weights.

    Raises:
        ValueError: If the activation/weight dtype combination is invalid.
    """
    if activation_dtype:
        activation_config = FakeQuantizeConfig(
            dtype=activation_dtype.value, granularity="per_token", is_symmetric=False
        )
    weight_config = FakeQuantizeConfig(dtype=weight_dtype.value, group_size=group_size)
    linear_quantize_config = IntXQuantizationAwareTrainingConfig(
        activation_config=None if activation_dtype is None else activation_config,
        weight_config=weight_config,
    )
    quantize_(model, linear_quantize_config)
    if quantize_embedding:
        # activation fake quantization is not supported for embedding layers
        embedding_quantize_config = IntXQuantizationAwareTrainingConfig(
            activation_config=None,
            weight_config=weight_config,
        )
        quantize_(
            model,
            embedding_quantize_config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )


def quantize_model_for_ptq(
    model,
    weight_dtype: TorchIntDType,
    group_size: int | None = None,
    activation_dtype: TorchIntDType | None = None,
    quantize_embedding: bool | None = None,
):
    """
    This function is used to quantize a model for post-training quantization.
    It swaps the model's linear layers with fake quantized linear layers.
    If `quantize_embedding` is True, it will also swap the model's embedding weights with fake quantized embedding weights.

    Args:
        model: The model to quantize.
        weight_dtype: The dtype to use for weight quantization.
        group_size: The group size to use for weight quantization.
        activation_dtype: The dtype to use for activation quantization.
        quantize_embedding: Whether to quantize the model's embedding weights.

    """
    linear_ptq_config = get_ptq_config(
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        group_size=group_size,
    )
    quantize_(model, linear_ptq_config)
    if quantize_embedding:
        embedding_quantize_config = get_ptq_config(
            weight_dtype=weight_dtype,
            activation_dtype=None,
            group_size=group_size,
        )
        quantize_(
            model,
            embedding_quantize_config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )


def convert_qat_model_for_ptq(
    model,
    *,
    quantize_embedding: bool | None = None,
):
    """
    This function is used to convert a swap fake-quantized modules in a model
    which has been trained with QAT back to the original modules, ready for PTQ.

    Args:
        model: The model to convert.
        quantize_embedding: Whether to quantize the model's embedding weights.
    """
    if quantize_embedding:

        def filter_fn(m, _):
            return isinstance(m, nn.Embedding) or _is_linear(m)

    else:
        filter_fn = _is_linear
    quantize_(model, FromIntXQuantizationAwareTrainingConfig(), filter_fn=filter_fn)
