"""JSON schema and runtime dtype compatibility for quantization configs."""

import warnings

import pytest
from pydantic.json_schema import PydanticJsonSchemaWarning

from axolotl.utils.schemas.config import AxolotlInputConfig
from axolotl.utils.schemas.enums import TorchAOQuantDType
from axolotl.utils.schemas.quantization import PTQConfig, QATConfig


def test_full_config_json_schema():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PydanticJsonSchemaWarning)
        schema = AxolotlInputConfig.model_json_schema(by_alias=False)
    assert set(schema["properties"]) == set(AxolotlInputConfig.model_fields)
    for name in ("QATConfig", "PTQConfig"):
        assert schema["$defs"][name]["properties"]["weight_dtype"]["default"] == "int8"
    assert schema["$defs"]["TorchAOQuantDType"]["type"] == "string"


@pytest.mark.parametrize("model", [QATConfig, PTQConfig])
@pytest.mark.parametrize(
    "name", [dtype.name for dtype in TorchAOQuantDType] + ["fp8", "float8"]
)
def test_dtype_json_roundtrip_preserves_runtime_values(model, name):
    config = model(weight_dtype=name, activation_dtype=name)
    expected = TorchAOQuantDType.from_string(name)
    assert config.weight_dtype is expected
    assert config.model_dump()["weight_dtype"] is expected
    assert config.weight_dtype.value is expected.value
    assert config.model_dump(mode="json")["weight_dtype"] == expected.name
    assert model.model_validate_json(config.model_dump_json()) == config
    assert name in model.model_json_schema()["$defs"]["TorchAOQuantDType"]["enum"]


@pytest.mark.parametrize("model", [QATConfig, PTQConfig])
def test_default_dtype_json_roundtrip(model):
    config = model()
    assert config.weight_dtype is TorchAOQuantDType.int8
    assert config.activation_dtype is None
    assert model.model_validate_json(config.model_dump_json()) == config
