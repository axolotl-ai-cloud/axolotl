"""
Export Config Schema
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Weight types accepted by llama.cpp's `llama-quantize`. Validated up front so a typo
# fails immediately rather than after the (slow) f16 conversion has already run.
GGUF_QUANT_TYPES = frozenset(
    {
        "F32", "F16", "BF16", "Q8_0",
        "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
        "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K", "Q2_K_S",
        "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
        "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
        "TQ1_0", "TQ2_0", "MXFP4_MOE",
    }
)  # fmt: skip


class ExportConfig(BaseModel):
    """Config for exporting a trained model to a deployment format."""

    format: Literal["gguf"] = Field(
        default="gguf", description="Deployment format to export to."
    )
    outtype: Literal["f32", "f16", "bf16", "q8_0", "auto"] = Field(
        default="f16", description="Weight type of the unquantized GGUF conversion."
    )
    quantize: list[str] = Field(
        default_factory=list,
        description="llama.cpp quant types to additionally emit, e.g. ['Q4_K_M', 'Q8_0'].",
    )
    output_dir: str | None = Field(
        default=None,
        description="Where to write exported files. Default: {output_dir}/gguf.",
    )
    llama_cpp_dir: str | None = Field(
        default=None,
        description="Path to a built llama.cpp checkout. Falls back to $LLAMA_CPP_DIR.",
    )

    @field_validator("quantize", mode="before")
    @classmethod
    def validate_quant_types(cls, quantize: Any) -> list[str]:
        if not quantize:
            return []
        if isinstance(quantize, str):
            quantize = quantize.split(",")
        quant_types = [str(quant).strip().upper() for quant in quantize]
        if unknown := sorted(set(quant_types) - GGUF_QUANT_TYPES):
            raise ValueError(
                f"Unknown GGUF quant type(s): {unknown}. "
                f"Must be one of: {sorted(GGUF_QUANT_TYPES)}"
            )
        return quant_types
