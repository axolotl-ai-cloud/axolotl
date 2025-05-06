from dataclasses import dataclass
from typing import Optional

import torch
from torchao.core.config import AOBaseConfig
from torchao.dtypes import QDQLayout
from torchao.dtypes.utils import Layout
from torchao.experimental.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
)
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
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6

from axolotl.utils.schemas.qat import TorchIntDType

# adapted from https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py#L1825
# TODO @SalmanMohammadi to be imported under torchao.quantization.quant_api.IntxWeightOnlyConfig w/torchao 0.11.0


@dataclass
class IntxWeightOnlyConfig(AOBaseConfig):
    """
    Configuration for quantizing weights to torch.intx, with 1 <= x <= 8.
    Weights are quantized with scales/zeros in a groupwise or channelwise
    manner using the number of bits specified by weight_dtype.
    args:
        weight_dtype: The dtype to use for weight quantization.  Must be torch.intx, where 1 <= x <= 8.
            torch.intx with x < 8 requires TORCH_VERSION_AT_LEAST_2_6
        granularity: The granularity to use for weight quantization.  Must be PerGroup or PerAxis(0).
        mapping_type: The type of mapping to use for the weight quantization.
            Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.
        scale_dtype: The dtype to use for the weight scale.
        layout: The layout to use for the packed weight tensor:
            - QDQLayout: this layout is designed for export to ExecuTorch.this layout represents the quantization with Q/DQ quant primitives,
                and is intended for export applications like ExecuTorch.
    """

    weight_dtype: torch.dtype = torch.int8
    granularity: Granularity = PerAxis(0)
    mapping_type: MappingType = MappingType.SYMMETRIC
    scale_dtype: Optional[torch.dtype] = None
    layout: Layout = QDQLayout()

    def __post_init__(self):
        assert TORCH_VERSION_AT_LEAST_2_6, "IntxWeightOnlyConfig requires torch 2.6+"
        assert self.weight_dtype in [
            getattr(torch, f"int{b}") for b in range(1, 9)
        ], f"weight_dtype must be torch.intx, where 1 <= x <= 8, but got {self.weight_dtype}"
        assert isinstance(
            self.granularity, (PerAxis, PerGroup)
        ), f"granularity must be PerAxis or PerGroup, but got {self.granularity}"
        if isinstance(self.granularity, PerAxis):
            assert (
                self.granularity.axis == 0
            ), f"axis must be 0 with PerAxis, but got {self.granularity.axis}"
        assert self.mapping_type in [
            MappingType.ASYMMETRIC,
            MappingType.SYMMETRIC,
        ], f"mapping_type must be MappingType.ASYMMETRIC or MappingType.SYMMETRIC, but got {self.mapping_type}"


def get_ptq_config(
    weight_dtype: TorchIntDType,
    activation_dtype: TorchIntDType | None = None,
    group_size: int | None = None,
):
    if group_size is not None:
        granularity = PerGroup(group_size)
    else:
        granularity = PerAxis(0)

    if activation_dtype is None:
        if "u" in weight_dtype.name:
            return UIntXWeightOnlyConfig(
                dtype=weight_dtype.value,
                group_size=group_size,
            )
        elif weight_dtype == TorchIntDType.int8:
            return Int8WeightOnlyConfig(
                group_size=group_size,
            )
        elif weight_dtype == TorchIntDType.int4:
            return Int4WeightOnlyConfig(
                group_size=group_size,
            )
        else:
            return IntxWeightOnlyConfig(
                weight_dtype=weight_dtype.value,
                granularity=granularity,
            )
    elif activation_dtype == TorchIntDType.int4 and weight_dtype == TorchIntDType.int4:
        return Int4DynamicActivationInt4WeightConfig()
    elif activation_dtype == TorchIntDType.int8 and weight_dtype == TorchIntDType.int8:
        return Int8DynamicActivationInt8WeightConfig()
    elif activation_dtype == TorchIntDType.int8:
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype.value,
            granularity=granularity,
            has_weight_zeros=False,
        )
    else:
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
    quantize_(model, ptq_config)
