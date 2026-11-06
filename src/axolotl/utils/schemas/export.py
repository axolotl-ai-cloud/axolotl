"""
Export Config Schema
"""

from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

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

# `convert_lora_to_gguf.py` takes fewer weight types than `convert_hf_to_gguf.py`.
GGUF_LORA_OUTTYPES = frozenset({"f32", "f16", "bf16", "q8_0", "auto"})
GGUF_LORA_DEFAULT_OUTTYPE = "f32"
GGUF_DEFAULT_OUTTYPE = "f16"


def validate_lora_export(outtype: str, quantize: Sequence[str]) -> None:
    """Reject settings the LoRA converter cannot honour."""
    if quantize:
        raise ValueError(
            "`llama-quantize` only takes full models, so `export.quantize` cannot be "
            "combined with a LoRA export. Run `axolotl merge-lora` and export the "
            "merged model instead."
        )
    if outtype not in GGUF_LORA_OUTTYPES:
        raise ValueError(
            f"`convert_lora_to_gguf.py` cannot write {outtype}. `export.outtype` must "
            f"be one of: {sorted(GGUF_LORA_OUTTYPES)}."
        )


class ExportConfig(BaseModel):
    """Config for exporting a trained model to a deployment format."""

    format: Literal["gguf"] = Field(
        default="gguf", description="Deployment format to export to."
    )
    outtype: Literal["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"] | None = (
        Field(
            default=None,
            description="Weight type of the GGUF conversion. Default: f16 for a full "
            "model, f32 for a LoRA adapter.",
        )
    )
    quantize: list[str] = Field(
        default_factory=list,
        description="llama.cpp quant types to additionally emit, e.g. ['Q4_K_M', 'Q8_0'].",
    )
    outfile: str | None = Field(
        default=None,
        description="Output path; `{ftype}` is replaced by each weight type. Default: {output_dir}/gguf/{run}-{ftype}.gguf",
    )
    llama_cpp_dir: str | None = Field(
        default=None,
        description="Path to a built llama.cpp checkout. Falls back to $LLAMA_CPP_DIR.",
    )
    lora: bool = Field(
        default=False,
        description="Export the adapter as a standalone GGUF LoRA instead of a full model.",
    )

    @field_validator("quantize", mode="before")
    @classmethod
    def validate_quant_types(cls, quantize: Any) -> list[str]:
        if not quantize:
            return []
        if isinstance(quantize, str):
            quantize = quantize.split(",")
        quant_types = list(dict.fromkeys(str(q).strip().upper() for q in quantize))
        if unknown := sorted(set(quant_types) - GGUF_QUANT_TYPES):
            raise ValueError(
                f"Unknown GGUF quant type(s): {unknown}. "
                f"Must be one of: {sorted(GGUF_QUANT_TYPES)}"
            )
        return quant_types

    @model_validator(mode="after")
    def validate_quantize(self):
        if not self.quantize:
            return self
        # llama.cpp refuses to dequantize an already-quantized source.
        if self.resolved_outtype(False) in ("q8_0", "tq1_0", "tq2_0"):
            raise ValueError(
                f"llama.cpp cannot requantize from {self.outtype}. Use an f16/bf16/f32 "
                "`export.outtype`, or drop `export.quantize`."
            )
        if self.outfile and "{ftype}" not in self.outfile:
            raise ValueError(
                "`export.outfile` needs a `{ftype}` placeholder when `export.quantize` "
                "is set, e.g. `model-{ftype}.gguf`."
            )
        return self

    @model_validator(mode="after")
    def validate_lora(self):
        if self.lora:
            validate_lora_export(self.resolved_outtype(True), self.quantize)
        return self

    def resolved_outtype(self, lora: bool) -> str:
        """llama.cpp's two converters ship different defaults, f16 and f32."""
        if self.outtype is not None:
            return self.outtype
        return GGUF_LORA_DEFAULT_OUTTYPE if lora else GGUF_DEFAULT_OUTTYPE
