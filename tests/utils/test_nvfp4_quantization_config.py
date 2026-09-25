"""NVFP4 activation requests must not silently select weight-only quantization."""

import pytest

from axolotl.utils.quantization import (
    get_quantization_config,
    quantization_config_to_str,
)
from axolotl.utils.schemas.enums import TorchAOQuantDType as DType


@pytest.mark.parametrize("group_size", [None, 16])
@pytest.mark.parametrize("dynamic", [False, True])
def test_nvfp4_factory_preserves_activation_request(group_size, dynamic):
    from torchao.prototype.mx_formats import (
        NVFP4DynamicActivationNVFP4WeightConfig,
        NVFP4WeightOnlyConfig,
    )

    config = get_quantization_config(
        DType.nvfp4, DType.nvfp4 if dynamic else None, group_size
    )
    if dynamic:
        assert config.use_triton_kernel is False
    assert quantization_config_to_str[type(config)] == (
        "nvfp4-dynamic" if dynamic else "nvfp4"
    )
    assert isinstance(
        config,
        NVFP4DynamicActivationNVFP4WeightConfig if dynamic else NVFP4WeightOnlyConfig,
    )


def test_nvfp4_factory_rejects_unsupported_activation_or_block():
    with pytest.raises(ValueError, match="activations"):
        get_quantization_config(DType.nvfp4, DType.float8_e4m3fn)
    with pytest.raises(ValueError, match="group_size"):
        get_quantization_config(DType.nvfp4, DType.nvfp4, 32)
