from dataclasses import dataclass
from typing import Optional

import torch
from torchao.core.config import AOBaseConfig
from torchao.dtypes import QDQLayout
from torchao.dtypes.utils import Layout
from torchao.experimental.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
)
from torchao.quantization.quant_api import _is_linear

from torchao.quantization.qat import FromIntXQuantizationAwareTrainingConfig
from torchao.quantization import quantize_
from torchao.quantization.granularity import Granularity, PerAxis, PerGroup
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization.quant_api import (
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
)
from torchao.quantization.quant_primitives import MappingType
import torch.nn as nn
import logging
from axolotl.utils.schemas.enums import TorchIntDType
from axolotl.utils.schemas.quantization import QATConfig

LOG = logging.getLogger(__name__)

def get_ptq_config(
    weight_dtype: TorchIntDType,
    activation_dtype: TorchIntDType | None = None,
    group_size: int | None = None,
):
    if activation_dtype is None:
        if "u" in weight_dtype.name:
            return UIntXWeightOnlyConfig(
                dtype=weight_dtype.value,
                group_size=group_size,
                set_inductor_config=False
            )
        elif weight_dtype == TorchIntDType.int8:
            if group_size is None:
                raise ValueError(
                    "group_size must be specified for int8 weight only quantization")
            return Int8WeightOnlyConfig(
                group_size=group_size,
            )
        elif weight_dtype == TorchIntDType.int4:
            if group_size is None:
                raise ValueError(
                    "group_size must be specified for int4 weight only quantization")
            return Int4WeightOnlyConfig(
                group_size=group_size,
            )
    elif activation_dtype == TorchIntDType.int4 and weight_dtype == TorchIntDType.int4:
        return Int4DynamicActivationInt4WeightConfig()
    elif activation_dtype == TorchIntDType.int8 and weight_dtype == TorchIntDType.int8:
        return Int8DynamicActivationInt8WeightConfig()
    raise ValueError(
        f"Invalid activation/weight dtype combination: {activation_dtype}/{weight_dtype}"
    )


def quantize_model_for_qat(
    model,
    weight_dtype: TorchIntDType,
    group_size: int,
    activation_dtype: TorchIntDType | None = None,
    quantize_embedding: bool = False,
):
    """
    This function is used to quantize a model for QAT.
    It swaps the model's linear layers with fake quantized linear layers.
    If `quantize_embedding` is True, it will also swap the model's embedding weights with fake quantized embedding weights. 
    Args:
        model: The model to quantize.
        weight_dtype: The dtype to use for weight quantization.
        group_size: The group size to use for weight quantization.
        activation_dtype: The dtype to use for activation quantization.
        quantize_embedding: Whether to quantize the model's embedding weights.  

    Returns:
        The quantized model.
    Raises:
        ValueError: If the activation/weight dtype combination is invalid.
    """
    if activation_dtype:
        activation_config = FakeQuantizeConfig(
            dtype=activation_dtype.value, granularity="per_token", is_symmetric=False
        )
    weight_config = FakeQuantizeConfig(dtype=weight_dtype.value, group_size=group_size)
    quantize_config = IntXQuantizationAwareTrainingConfig(
        activation_config=None if activation_dtype is None else activation_config,
        weight_config=weight_config,
    )
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
    quantize_(model, quantize_config)


def quantize_model_for_ptq(
    model,
    weight_dtype: TorchIntDType,
    group_size: int | None = None,
    activation_dtype: TorchIntDType | None = None,
    quantize_embedding: bool | None = None,
):
    ptq_config = get_ptq_config(
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        group_size=group_size,
    )
    quantize_(model, ptq_config)
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


def convert_qat_model_for_ptq(model, 
                              weight_dtype: TorchIntDType,
                              group_size: int,
                              activation_dtype: TorchIntDType | None = None,
                              quantize_embedding: bool | None = None,
                              ):
    if quantize_embedding:
        def filter_fn(m, _): return isinstance(m, nn.Embedding) or _is_linear(m)
    else:
        filter_fn = _is_linear
    quantize_(model, FromIntXQuantizationAwareTrainingConfig(), filter_fn=filter_fn)
