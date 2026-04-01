"""
Utilities for quantization including QAT and PTQ using torchao.
"""

import torch
from packaging import version
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import (
    QATConfig,
)
from torchao.quantization.qat.fake_quantize_config import Int4WeightFakeQuantizeConfig
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
)

from axolotl.utils.schemas.enums import TorchAOQuantDType

quantization_config_to_str = {
    Int8DynamicActivationIntxWeightConfig: "int8int4",
    Float8DynamicActivationFloat8WeightConfig: "fp8fp8",
    Float8DynamicActivationInt4WeightConfig: "fp8int4",
}

if version.parse(torch.__version__) >= version.parse("2.8.0"):
    try:
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig

        quantization_config_to_str[NVFP4WeightOnlyConfig] = "nvfp4"
    except (ImportError, RuntimeError):
        pass

    # int4 weight config imports will fail on machines with fbgemm-gpu installed
    # without a CUDA runtime available so we do this safely
    try:
        from torchao.quantization.quant_api import Int4WeightOnlyConfig

        quantization_config_to_str[Int4WeightOnlyConfig] = "int4"
    except (ImportError, RuntimeError):
        pass

    try:
        from torchao.prototype.mx_formats import (
            MXDynamicActivationMXWeightConfig as MXLinearConfig,
        )

        quantization_config_to_str[MXLinearConfig] = "mxfp4"
    except (ImportError, RuntimeError):
        pass


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
            raise ValueError("Int8WeightOnlyConfig is not supported by torchao QAT.")
        if weight_dtype == TorchAOQuantDType.int4:
            from torchao.quantization.quant_api import Int4WeightOnlyConfig

            if group_size is not None:
                return Int4WeightOnlyConfig(group_size=group_size, version=2)
            else:
                return Int4WeightOnlyConfig(version=2)
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
        kwargs = {"weight_dtype": torch.int4}
        if group_size is not None:
            kwargs["weight_granularity"] = PerGroup(group_size=group_size)
        return Int8DynamicActivationIntxWeightConfig(**kwargs)
    if (
        activation_dtype == TorchAOQuantDType.float8_e4m3fn
        and weight_dtype == TorchAOQuantDType.float8_e4m3fn
    ):
        return Float8DynamicActivationFloat8WeightConfig()
    if (
        activation_dtype == TorchAOQuantDType.float8_e4m3fn
        and weight_dtype == TorchAOQuantDType.int4
    ):
        return Float8DynamicActivationInt4WeightConfig()
    if weight_dtype == TorchAOQuantDType.nvfp4:
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig

        if group_size is not None and group_size != 16:
            raise ValueError("NVFP4 quantization must use a group_size of 16")
        return NVFP4WeightOnlyConfig()

    if weight_dtype == TorchAOQuantDType.mxfp4:
        # MXFP4 uses block_size=32 by default (vs NVFP4's 16)
        block_size = group_size if group_size is not None else 32
        if block_size != 32:
            raise ValueError(
                "MXFP4 quantization must use a block_size (group_size) of 32"
            )

        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig

        return MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float4_e2m1fn_x2,
            weight_dtype=torch.float4_e2m1fn_x2,
            block_size=block_size,
        )

    raise ValueError(
        f"Invalid activation/weight dtype combination: {activation_dtype}/{weight_dtype}"
    )


def _attach_torchao_quantizer(
    model, quantization_config, include_input_output_embeddings=False
):
    """Attach a TorchAoHfQuantizer to the model so save_pretrained uses
    torchao's flatten_tensor_state_dict path, preserving quantized weights
    (e.g. MXTensor qdata+scale) in the safetensors file.

    Without this, save_pretrained falls through to the default path which
    calls safetensors storage_ptr() on tensor subclasses and crashes.
    """
    from transformers import TorchAoConfig
    from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer

    ao_config = TorchAoConfig(
        quant_type=quantization_config,
        include_input_output_embeddings=include_input_output_embeddings,
    )
    model.config.quantization_config = ao_config
    quantizer = TorchAoHfQuantizer(ao_config)
    model.hf_quantizer = quantizer


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

    _attach_torchao_quantizer(
        model,
        linear_ptq_config,
        include_input_output_embeddings=bool(quantize_embedding),
    )


def _make_qat_config(
    base_config: AOBaseConfig,
    weight_dtype: TorchAOQuantDType,
    activation_dtype: TorchAOQuantDType | None,
    group_size: int | None,
) -> QATConfig:
    """Build a QATConfig, explicitly constructing fake quantize configs to ensure
    group_size and other params are properly propagated (torchao's QATConfig(base_config)
    does not always map these correctly)."""
    from torchao.quantization.qat.fake_quantize_config import (
        Float8FakeQuantizeConfig,
        IntxFakeQuantizeConfig,
    )

    if weight_dtype == TorchAOQuantDType.mxfp4:
        from torchao.prototype.qat import MXFakeQuantizeConfig

        block_size = getattr(base_config, "block_size", 32)
        mx_fq = MXFakeQuantizeConfig(
            dtype=torch.float4_e2m1fn_x2, block_size=block_size
        )
        return QATConfig(activation_config=mx_fq, weight_config=mx_fq)

    # Build explicit weight config
    weight_fq_config: (
        Int4WeightFakeQuantizeConfig
        | IntxFakeQuantizeConfig
        | Float8FakeQuantizeConfig
        | None
    ) = None
    if weight_dtype == TorchAOQuantDType.int4:
        gs = (
            group_size
            if group_size is not None
            else getattr(base_config, "group_size", 128)
        )
        activation_dt = None
        if activation_dtype == TorchAOQuantDType.int8:
            activation_dt = torch.bfloat16
        elif activation_dtype == TorchAOQuantDType.float8_e4m3fn:
            activation_dt = torch.float8_e4m3fn
        kwargs = {"group_size": gs}
        if activation_dt is not None:
            kwargs["activation_dtype"] = activation_dt
        weight_fq_config = Int4WeightFakeQuantizeConfig(**kwargs)
    elif weight_dtype == TorchAOQuantDType.float8_e4m3fn:
        weight_fq_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn)

    # Build explicit activation config
    activation_fq_config = None
    if activation_dtype == TorchAOQuantDType.int8:
        activation_fq_config = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_token", is_symmetric=False
        )
    elif activation_dtype == TorchAOQuantDType.float8_e4m3fn:
        activation_fq_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn)

    if weight_fq_config is not None:
        return QATConfig(
            weight_config=weight_fq_config,
            activation_config=activation_fq_config,
        )

    # Fallback to base_config for unhandled combos
    return QATConfig(base_config)


def prepare_model_for_qat(
    model,
    weight_dtype: TorchAOQuantDType,
    group_size: int | None = None,
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
    qat_config = _make_qat_config(
        base_config, weight_dtype, activation_dtype, group_size
    )
    quantize_(model, qat_config)
    if quantize_embedding:
        # activation fake quantization is not supported for embedding layers
        embedding_base_config = get_quantization_config(
            weight_dtype=weight_dtype,
            activation_dtype=None,
            group_size=group_size,
        )
        embedding_qat_config = _make_qat_config(
            embedding_base_config, weight_dtype, None, group_size
        )
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
