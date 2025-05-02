from torchao.quantization.quant_api import _is_linear
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization import quantize_
from dataclasses import dataclass
import torch
from torchao.quantization.quant_api import (
    UIntXWeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    Int4WeightOnlyConfig,
)
from torchao.experimental.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
)
from typing import Optional
from torchao.core.config import AOBaseConfig
from torchao.quantization.granularity import Granularity, PerAxis, PerGroup
from torchao.quantization.quant_primitives import MappingType
from torchao.dtypes.utils import Layout
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6
from axolotl.utils.schemas.qat import TorchIntDType
from torchao.dtypes import QDQLayout
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
        assert self.weight_dtype in [getattr(torch, f"int{b}") for b in range(1, 9)], (
            f"weight_dtype must be torch.intx, where 1 <= x <= 8, but got {self.weight_dtype}"
        )
        assert isinstance(self.granularity, (PerAxis, PerGroup)), (
            f"granularity must be PerAxis or PerGroup, but got {self.granularity}"
        )
        if isinstance(self.granularity, PerAxis):
            assert self.granularity.axis == 0, (
                f"axis must be 0 with PerAxis, but got {self.granularity.axis}"
            )
        assert self.mapping_type in [MappingType.ASYMMETRIC, MappingType.SYMMETRIC], (
            f"mapping_type must be MappingType.ASYMMETRIC or MappingType.SYMMETRIC, but got {self.mapping_type}"
        )


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
        return Int4DynamicActivationInt4WeightConfig(
        )
    elif activation_dtype == TorchIntDType.int8 and weight_dtype == TorchIntDType.int8:
        return Int8DynamicActivationInt8WeightConfig(
        )
    elif activation_dtype == TorchIntDType.int8:
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype.value,
            granularity=granularity,
            has_weight_zeros=False,
        )
    else:
        raise ValueError(
            f"Invalid activation/weight dtype combination: {activation_dtype}/{weight_dtype}")


def quantize_model_for_qat(model, qat_cfg):
    quantize_embedding = qat_cfg.quantize_embedding

    if qat_cfg.activation_dtype:
        activation_config = FakeQuantizeConfig(
            dtype=qat_cfg.activation_dtype, granularity="per_token", is_symmetric=False
        )
    else:
        activation_config = None

    if qat_cfg.weight_dtype:
        weight_config = FakeQuantizeConfig(
            dtype=qat_cfg.weight_dtype, group_size=qat_cfg.group_size
        )
    else:
        weight_config = None
    quantize_config = IntXQuantizationAwareTrainingConfig(
        activation_config,
        weight_config,
    )

    if quantize_embedding:
        def filter_fn(m, _): return (isinstance(m, torch.nn.Embedding) or _is_linear(m))
    else:
        filter_fn = None
    quantize_(model,
              quantize_config,
              filter_fn=filter_fn
              )

def quantize_model_for_ptq(model, qat_cfg):
    ptq_config = get_ptq_config(
        weight_dtype=qat_cfg.weight_dtype,
        activation_dtype=qat_cfg.activation_dtype,
        group_size=qat_cfg.group_size,
    )
    quantize_(model, ptq_config)
