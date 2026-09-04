"""pytest tests for the `export` config schema."""

import pytest
from pydantic import ValidationError

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.export import ExportConfig


class TestExportConfig:
    """Tests for ExportConfig validation and normalization."""

    def test_defaults(self):
        config = ExportConfig()
        assert (config.format, config.outtype, config.quantize) == ("gguf", "f16", [])

    @pytest.mark.parametrize(
        "quantize, expected",
        [
            (None, []),
            ("", []),
            ("q4_k_m", ["Q4_K_M"]),
            ("Q4_K_M, Q8_0", ["Q4_K_M", "Q8_0"]),
            (["q8_0", "BF16"], ["Q8_0", "BF16"]),
        ],
    )
    def test_quantize_normalization(self, quantize, expected):
        assert ExportConfig(quantize=quantize).quantize == expected

    def test_unknown_quant_type(self):
        with pytest.raises(
            ValidationError, match=r"Unknown GGUF quant type\(s\): \['Q9_K'\]"
        ):
            ExportConfig(quantize="Q4_K_M,Q9_K")

    @pytest.mark.parametrize("field, value", [("format", "onnx"), ("outtype", "int4")])
    def test_unsupported_values(self, field, value):
        with pytest.raises(ValidationError):
            ExportConfig(**{field: value})

    def test_q8_0_outtype_rejects_quantize(self):
        with pytest.raises(ValidationError, match="cannot requantize from q8_0"):
            ExportConfig(outtype="q8_0", quantize=["Q4_K_M"])

    @pytest.mark.parametrize("outtype", ["f16", "bf16", "f32", "auto"])
    def test_dequantizable_outtypes_allow_quantize(self, outtype):
        assert ExportConfig(outtype=outtype, quantize=["Q4_K_M"]).quantize == ["Q4_K_M"]

    def test_q8_0_outtype_alone_is_allowed(self):
        assert ExportConfig(outtype="q8_0").outtype == "q8_0"


class TestExportConfigInAxolotlConfig:
    """The `export` block round-trips through full config validation."""

    def test_absent_by_default(self, min_base_cfg):
        assert validate_config(min_base_cfg).export is None

    def test_normalized_in_place(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(export={"quantize": ["q4_k_m"]})
        assert validate_config(cfg).export["quantize"] == ["Q4_K_M"]
