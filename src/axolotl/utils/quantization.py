"""
Utilities for quantization including QAT and PTQ using torchao.
"""

import torch
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.qat import (
    QATConfig,
)
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
)

from axolotl.utils.schemas.enums import TorchAOQuantDType


def get_quantization_config(
    weight_dtype: TorchAOQuantDType,
    activation_dtype: TorchAOQuantDType | None = None,
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
        if weight_dtype == TorchAOQuantDType.int8:
            raise ValueError(
                "Int8WeightOnlyConfig is not supported by torchao QAT."
            )
        if weight_dtype == TorchAOQuantDType.int4:
            return Int4WeightOnlyConfig(group_size=group_size or -1, version=2)
    if (
        activation_dtype == TorchAOQuantDType.int4
        and weight_dtype == TorchAOQuantDType.int4
    ):
        raise ValueError(
            "Int4DynamicActivationInt4WeightConfig is not supported by torchao QAT."
        )
    if (
        activation_dtype == TorchAOQuantDType.int8
        and weight_dtype == TorchAOQuantDType.int8
    ):
        raise ValueError(
            "Int8DynamicActivationInt8WeightConfig is not supported by torchao QAT."
        )
    if (
        activation_dtype == TorchAOQuantDType.int8
        and weight_dtype == TorchAOQuantDType.int4
    ):
        return Int8DynamicActivationInt4WeightConfig(group_size=group_size or -1)
    if (
        activation_dtype == TorchAOQuantDType.float8_e4m3fn
        and weight_dtype == TorchAOQuantDType.float8_e4m3fn
    ):
        return Float8DynamicActivationFloat8WeightConfig(version=2)
    if (
        activation_dtype == TorchAOQuantDType.float8_e4m3fn
        and weight_dtype == TorchAOQuantDType.int4
    ):
        return Float8DynamicActivationInt4WeightConfig()
    raise ValueError(
        f"Invalid activation/weight dtype combination: {activation_dtype}/{weight_dtype}"
    )


def quantize_model(
    model,
    weight_dtype: TorchAOQuantDType,
    group_size: int | None = None,
    activation_dtype: TorchAOQuantDType | None = None,
    quantize_embedding: bool | None = None,
):
    """
    This function is used to quantize a model.

    Args:
        model: The model to quantize.
        weight_dtype: The dtype to use for weight quantization.
        group_size: The group size to use for weight quantization.
        activation_dtype: The dtype to use for activation quantization.
        quantize_embedding: Whether to quantize the model's embedding weights.

    """
    linear_ptq_config = get_quantization_config(
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        group_size=group_size,
    )
    quantize_(model, linear_ptq_config)
    if quantize_embedding:
        # activation fake quantization is not supported for embedding layers
        embedding_quantize_config = get_quantization_config(
            weight_dtype=weight_dtype,
            activation_dtype=None,
            group_size=group_size,
        )
        quantize_(
            model,
            embedding_quantize_config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )


def prepare_model_for_qat(
    model,
    weight_dtype: TorchAOQuantDType,
    group_size: int,
    activation_dtype: TorchAOQuantDType | None = None,
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
    base_config = get_quantization_config(
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        group_size=group_size,
    )
    qat_config = QATConfig(base_config)
    quantize_(model, qat_config)
    if quantize_embedding:
        # activation fake quantization is not supported for embedding layers
        embedding_base_config = get_quantization_config(
            weight_dtype=weight_dtype,
            activation_dtype=None,
            group_size=group_size,
        )
        embedding_qat_config = QATConfig(embedding_base_config)
        quantize_(
            model,
            embedding_qat_config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )


def convert_qat_model(
    model,
    quantize_embedding: bool = False,
):
    """
    This function converts a QAT model which has fake quantized layers back to the original model.
    """
    config = QATConfig(step="convert")
    quantize_(model, config)
    if quantize_embedding:
        quantize_(
            model,
            config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )
