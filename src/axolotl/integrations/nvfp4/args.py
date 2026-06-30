"""NVFP4-GEMM training config schema (nested) — real FP4 forward/backward compute, distinct from the fake-quant ``quantization:`` block. Flat ``@property`` shims expose the old attribute names the relocated swap/kernel code still reads."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Reserved `quantize` keywords (bespoke swaps); other entries are body-linear name fragments.
QUANT_KEYWORDS = ("lm_head", "embeddings", "vision_tower")
QUANT_ALL = "all"


class NVFP4RecipeArgs(BaseModel):
    """Numerics of the FP4 convergence recipe (safe defaults; rarely touched)."""

    model_config = ConfigDict(extra="forbid")

    stochastic_rounding: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Stochastic rounding on gradient operands (NVFP4Recipe)."
        },
    )
    hadamard: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Random Hadamard transform on wgrad inputs (NVFP4Recipe)."
        },
    )


class NVFP4CrossEntropyArgs(BaseModel):
    """Head-aware fused lm_head + cross-entropy (vocab-tiled, logits never materialized; frozen head only)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "auto", "fp4", "bf16"] = Field(
        default="off",
        json_schema_extra={
            "description": "Fused lm_head CE kernel: 'auto' picks fp4 (FP4 head), "
            "bf16 (plain frozen head), else materialized; 'fp4'/'bf16' force a kernel."
        },
    )
    vocab_block: int = Field(
        default=4096,
        gt=0,
        json_schema_extra={
            "description": "Vocab-tile width (all CE kernels); loss-invariant speed/VRAM "
            "dial (4096 balanced, 8192 max throughput). Env override: "
            "AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK."
        },
    )


class NVFP4KeepHPBlocks(BaseModel):
    """Transformer blocks kept in high precision (the NVFP4 paper keeps ~15% in bf16, weighted to the tail)."""

    model_config = ConfigDict(extra="forbid")

    first: int = Field(default=0, ge=0)
    last: int = Field(default=0, ge=0)


class NVFP4Args(BaseModel):
    """``nvfp4_training:`` config block."""

    # Reject unknown keys so a stale/mistyped field fails loudly instead of no-opping.
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable NVFP4-GEMM training (real FP4 forward/backward "
            "GEMMs); Blackwell-only and only fast under torch.compile."
        },
    )

    quantize: list[str] = Field(
        default_factory=list,
        json_schema_extra={
            "description": "Module-selection list — quantizes exactly what you name. "
            "Entries are body-linear name fragments (e.g. q_proj, mlp.gate_proj) or "
            "keywords: 'all' (every eligible body linear), 'lm_head' (frozen-tied "
            "shared once, trainable-tied raises), 'embeddings' (frozen W4A16, hidden "
            "%16), 'vision_tower' (frozen vision-encoder linears). Empty quantizes "
            "nothing, so `enabled: true` requires a non-empty list ([all] for the "
            "full-throughput default)."
        },
    )
    keep_hp_blocks: NVFP4KeepHPBlocks | Literal["paper"] = Field(
        default_factory=NVFP4KeepHPBlocks,
        json_schema_extra={
            "description": "Transformer blocks kept high-precision (subtracted from "
            "selected body linears): {first: N, last: M}, or 'paper' for the recipe's "
            "~15% policy (first=1, last=round(0.13*L))."
        },
    )

    base: Literal["compute", "storage", "hp"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Adapter base handling (adapter modes only): 'compute' "
            "(recommended; pre-quantized FP4 fprop+dgrad, ~1.75x weight memory), "
            "'storage' (==NVFP4-QLoRA; FP4-packed ~3.5x, backward dequants to bf16), "
            "'hp' (HP base re-quantized per step, no memory win). null = storage for "
            "qlora else compute."
        },
    )

    cross_entropy: NVFP4CrossEntropyArgs = Field(default_factory=NVFP4CrossEntropyArgs)
    recipe: NVFP4RecipeArgs = Field(default_factory=NVFP4RecipeArgs)

    save_packed: bool = Field(
        default=False,
        json_schema_extra={
            "description": "On each save, write eligible weights NVFP4-packed to a "
            "sidecar (nvfp4_packed.pt, ~3.5x smaller). Frozen weights round-trip "
            "bit-exact; for FFT it is LOSSY for resume — storage/inference export only."
        },
    )

    @model_validator(mode="after")
    def _validate(self):
        if self.enabled and not self.quantize:
            raise ValueError(
                "nvfp4_training.enabled is true but `quantize` is empty — nothing "
                "would be quantized. List body name fragments and/or keywords "
                f"({', '.join(QUANT_KEYWORDS)}), or use quantize: [all] for every "
                "eligible body linear."
            )
        return self

    # Flat read-only views over the nested fields.
    @property
    def stochastic_rounding(self) -> bool:
        return self.recipe.stochastic_rounding

    @property
    def hadamard(self) -> bool:
        return self.recipe.hadamard

    @property
    def quantize_lm_head(self) -> bool:
        return "lm_head" in self.quantize

    @property
    def quantize_embeddings(self) -> bool:
        return "embeddings" in self.quantize

    @property
    def quantize_vision_tower(self) -> bool:
        return "vision_tower" in self.quantize

    @property
    def quantize_all_body(self) -> bool:
        """`all` keyword present — swap every eligible body linear."""
        return QUANT_ALL in self.quantize

    @property
    def quantize_body_fragments(self) -> tuple[str, ...]:
        """Body-linear name fragments in `quantize` (keywords excluded); restricts the body swap when `quantize_all_body` is False."""
        skip = set(QUANT_KEYWORDS) | {QUANT_ALL}
        return tuple(q for q in self.quantize if q not in skip)

    @property
    def base_mode(self) -> str | None:
        return self.base

    @property
    def quantize_base(self) -> bool:
        return self.base in ("storage", "compute")

    @property
    def lm_head_cross_entropy(self) -> str:
        return self.cross_entropy.mode

    @property
    def fused_ce_vocab_block(self) -> int:
        return self.cross_entropy.vocab_block

    @property
    def save_nvfp4(self) -> bool:
        return self.save_packed

    @property
    def skip_first_n_blocks(self) -> int:
        if self.keep_hp_blocks == "paper":
            return 1
        return self.keep_hp_blocks.first

    @property
    def skip_last_n_blocks(self) -> int:
        # The "paper" preset's tail (round(0.13*L)) needs the block count.
        if self.keep_hp_blocks == "paper":
            return 0
        return self.keep_hp_blocks.last

    @property
    def keep_hp_paper_preset(self) -> bool:
        return self.keep_hp_blocks == "paper"

    @property
    def fp4_cross_entropy_active(self) -> bool:
        """True when the resolved CE kernel reads the NVFP4-packed lm_head."""
        if self.cross_entropy.mode == "fp4":
            return True
        if self.cross_entropy.mode == "auto":
            return self.quantize_lm_head
        return False


class NVFP4PluginArgs(BaseModel):
    """Top-level input args contributed by the plugin (the ``nvfp4_training:`` key)."""

    nvfp4_training: NVFP4Args | None = None
