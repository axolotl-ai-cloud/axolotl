"""NVFP4-GEMM training config schema.

Real low-precision COMPUTE (FP4 forward/backward GEMMs on Blackwell), distinct
from the fake-quant QAT/PTQ `quantization:` block.
"""

import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NVFP4TrainingConfig(BaseModel):
    """NVFP4-GEMM training settings (module-swap on eligible nn.Linear)."""

    # Reject unknown keys so a stale `attention:`/`linear_attn:`/`mlp:` block (the
    # removed native FP4 attention path) fails loudly instead of silently no-opping.
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable NVFP4-GEMM training (real FP4 forward/backward GEMMs). "
            "Blackwell-only. The speedup only materializes under torch.compile."
        },
    )
    backend: Literal["native", "te"] = Field(
        default="native",
        json_schema_extra={
            "description": "FP4 compute backend. 'native' (default): our "
            "torch._scaled_mm module-swap; runs the full SR+RHT convergence recipe "
            "on any FP4-capable GPU (sm_100/sm_120), no extra build. 'te': NVIDIA "
            "Transformer Engine NVFP4BlockScaling; faster hand-tuned kernels, but "
            "needs `pip install axolotl[transformer-engine]` (source build) and on "
            "consumer Blackwell (sm_120) its RHT/SR kernels do not run — TE there is "
            "recipe-less (convergence unproven). Supports FFT and LoRA/QLoRA, but on "
            "adapter paths it keeps a high-precision base (ignores "
            "base_mode/quantize_base, so no FP4-storage saving) and is incompatible "
            "with the fused LoRA kernels."
        },
    )
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
    quantize_base: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Adapter modes only. When false (LoRA + FP4 compute): the "
            "frozen base weight stays high-precision and only the GEMM operands are "
            "FP4 — a throughput win, no memory saving. When true (NVFP4-QLoRA): the "
            "frozen base weight is stored packed in FP4 (~3.5x weight memory saving), "
            "replacing bnb NF4 storage. This is the FP4 equivalent of QLoRA, so it "
            "conflicts with load_in_4bit/load_in_8bit (drop those). `adapter: qlora` "
            "implies this (qlora == quantized base intent)."
        },
    )
    base_mode: Literal["compute", "storage", "hp"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Adapter modes only; overrides quantize_base when set. "
            "'compute' (recommended): the frozen base is pre-quantized once into two "
            "NVFP4 layouts so fprop+dgrad run as pure FP4 GEMMs with no per-step base "
            "quant prologue — fastest base compute, ~1.75x weight memory. 'storage' "
            "(== NVFP4-QLoRA): base packed in FP4, ~3.5x memory, backward dequants to "
            "bf16 (max memory, modest speed). 'hp': base kept high-precision, "
            "re-quantized each step (FP4 GEMM, no memory win)."
        },
    )
    exclude_modules: list[str] = Field(
        default_factory=lambda: ["lm_head", "embed_tokens"],
        json_schema_extra={
            "description": "Module name fragments kept in high precision (not swapped)."
        },
    )
    quantize_lm_head: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Remove lm_head from the high-precision exclusion and "
            "quantize the output projection to NVFP4 (memory + GEMM win on a large "
            "[vocab, hidden] tensor), at a small convergence cost (the NVFP4 paper "
            "keeps lm_head bf16). OFF by default. With UNtied embeddings only the "
            "lm_head is quantized (embed_tokens stays excluded unless "
            "quantize_embeddings is also set). With TIED embeddings (the frozen "
            "shared weight) the shared weight is quantized ONCE and routed to both "
            "the embedding lookup and the lm_head GEMM. A TRAINABLE tied weight "
            "still RAISES (FP4-storing it would corrupt training)."
        },
    )
    lm_head_cross_entropy: Literal["off", "auto", "fp4", "bf16", "fp8"] = Field(
        default="off",
        json_schema_extra={
            "description": "Head-aware fused lm_head + cross-entropy: tile the loss "
            "over the vocab so the [batch*seq, vocab] logit tensor is never "
            "materialized (memory win, frozen head only). 'auto' picks fp4 for an "
            "FP4 head (quantize_lm_head), bf16 for a plain frozen nn.Linear head, "
            "else the materialized path; 'fp4'/'bf16'/'fp8' force a kernel (fp8 is "
            "never auto-selected). Supersedes the deprecated "
            "fused_fp4_cross_entropy / bf16_lm_head_cross_entropy / "
            "fp8_lm_head_cross_entropy booleans."
        },
    )
    fused_fp4_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "DEPRECATED: use `lm_head_cross_entropy: fp4` (or 'auto')."
        },
    )
    fused_ce_vocab_block: int = Field(
        default=4096,
        gt=0,
        json_schema_extra={
            "description": "Vocab-tile width for lm_head_cross_entropy (all kernels). "
            "A pure speed<->VRAM dial, loss-invariant. 4096 (default) is balanced; "
            "8192 is max throughput. Overridable via the "
            "AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK env var (env wins)."
        },
    )
    bf16_lm_head_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "DEPRECATED: use `lm_head_cross_entropy: bf16` (or 'auto' "
            "with a bf16 head)."
        },
    )
    fp8_lm_head: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen lm_head to use torch FP8 scaled "
            "matmul in eval/no-grad forward only. Training forwards still use the "
            "original high-precision Linear. This is for eval/scoring/logprob "
            "throughput; greedy generation can diverge after the first changed "
            "argmax token. OFF by default."
        },
    )
    fp8_lm_head_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "DEPRECATED: use `lm_head_cross_entropy: fp8`."
        },
    )
    fp8_lm_head_granularity: Literal["tensorwise", "rowwise"] = Field(
        default="rowwise",
        json_schema_extra={
            "description": "Scaling granularity for fp8_lm_head. Rowwise keeps one "
            "scale per vocab row and had the best Qwen3.5 real-hidden-state argmax "
            "parity in the validation sweep; it is also the validated FP8 CE "
            "training default."
        },
    )
    fused_ce_fp4_matmul: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Experimental, fp4 kernel only. Run the per-tile logit "
            "matmul through FP4 torch._scaled_mm (Blackwell FP4 tensor cores). "
            "Faster than the memory-only fp4 CE path but slower than materialized "
            "bf16/Liger CE; keep off."
        },
    )
    fused_fp4_cross_entropy_fp4_matmul: bool = Field(
        default=False,
        json_schema_extra={"description": "DEPRECATED: use `fused_ce_fp4_matmul`."},
    )

    @model_validator(mode="after")
    def _resolve_fused_cross_entropy(self):
        """Map the deprecated per-kernel CE booleans onto ``lm_head_cross_entropy``;
        raise if an old flag conflicts with an explicit new value or another old flag.
        """
        legacy_map = {
            "fused_fp4_cross_entropy": "fp4",
            "bf16_lm_head_cross_entropy": "bf16",
            "fp8_lm_head_cross_entropy": "fp8",
        }
        set_legacy = [name for name in legacy_map if getattr(self, name)]
        if len(set_legacy) > 1:
            raise ValueError(
                "nvfp4_training: only one fused cross-entropy kernel may be enabled; "
                f"got deprecated {sorted(set_legacy)}. Set a single "
                "`lm_head_cross_entropy: auto|fp4|bf16|fp8`."
            )
        if set_legacy:
            mapped = legacy_map[set_legacy[0]]
            explicit = "lm_head_cross_entropy" in self.model_fields_set
            if explicit and self.lm_head_cross_entropy != mapped:
                raise ValueError(
                    f"nvfp4_training: `lm_head_cross_entropy: "
                    f"{self.lm_head_cross_entropy}` conflicts with deprecated "
                    f"`{set_legacy[0]}: true` (maps to '{mapped}'). Set only "
                    "`lm_head_cross_entropy`."
                )
            warnings.warn(
                f"nvfp4_training.{set_legacy[0]} is deprecated; use "
                f"`lm_head_cross_entropy: {mapped}`.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.lm_head_cross_entropy = mapped

        if self.fused_fp4_cross_entropy_fp4_matmul:
            if self.fused_ce_fp4_matmul is None:
                warnings.warn(
                    "nvfp4_training.fused_fp4_cross_entropy_fp4_matmul is deprecated; "
                    "use `fused_ce_fp4_matmul`.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.fused_ce_fp4_matmul = True
            elif not self.fused_ce_fp4_matmul:
                raise ValueError(
                    "nvfp4_training: `fused_ce_fp4_matmul: false` conflicts with "
                    "deprecated `fused_fp4_cross_entropy_fp4_matmul: true`. Set only "
                    "`fused_ce_fp4_matmul`."
                )
        return self

    @property
    def fp4_cross_entropy_active(self) -> bool:
        """True when the resolved CE kernel reads the NVFP4-packed lm_head."""
        if self.lm_head_cross_entropy == "fp4":
            return True
        if self.lm_head_cross_entropy == "auto":
            return bool(self.quantize_lm_head)
        return False

    quantize_embeddings: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store the input token embedding (embed_tokens) packed "
            "in FP4 and dequantize on lookup (W4A16; the lookup gathers rows, no "
            "activation quant). FROZEN only — a trainable embedding is skipped with "
            "a warning (no FP4 master for gradients). Hidden dim must be %16. OFF by "
            "default."
        },
    )
    quantize_vision_tower: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Multimodal only. Swap the FROZEN nn.Linear layers under "
            "the vision encoder (attn qkv/proj, mlp fc1/fc2) to the NVFP4 frozen "
            "base. The vision merger and patch-embed projection are left untouched, "
            "as are %32-ineligible dims. Warns if no vision tower is found (text "
            "model). OFF by default."
        },
    )
    fuse_rmsnorm: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Default off (throughput benefit is within noise; under "
            "torch.compile the reuse cache is bypassed). Fuse decoder RMSNorm with "
            "the NVFP4 activation quant in one Triton kernel so the qkv/gate-up base "
            "linears reuse the norm's pre-quantized activation. Auto-detects the "
            "gamma convention (plain `weight` vs zero-centered `1 + weight`) per norm "
            "and verifies each swap against the original before committing (reverts "
            "on mismatch)."
        },
    )
    shared_lora_base_fprop: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Experimental, LoRA/QLoRA native backend only. For fused "
            "LoRA QKV/QK projections whose frozen base layers are pre-quantized NVFP4 "
            "compute/storage bases, quantize and pack the shared activation once and "
            "reuse it across the base fprops. Omitted/null uses the "
            "AXOLOTL_NVFP4_SHARED_BASE_FPROP environment fallback (default off); "
            "set true or false in YAML to make the choice explicit. On Qwen3.5-9B "
            "with only 8 full-attention layers this was below whole-step noise, but "
            "models with more full-attention LoRA layers may benefit."
        },
    )
    fla_causal_conv_compile_boundary: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5/MoE sample-packing only. Run FLA varlen "
            "causal_conv1d behind a torch.compile boundary so packed cu_seqlens "
            "length changes do not trigger repeated Dynamo recompiles. Trades graph "
            "coverage for steadier steps."
        },
    )
    skip_first_n_blocks: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Keep the first N transformer blocks in high precision. "
            "The NVFP4 paper keeps ~15% of linear layers in bf16 (embeddings/lm_head "
            "plus the first ~2 and last ~8 blocks of a 12B model, weighted to the "
            "tail) for convergence; for a model with L blocks a reasonable default is "
            "skip_first_n_blocks=1 and skip_last_n_blocks~=round(0.13*L)."
        },
    )
    skip_last_n_blocks: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Keep the last N transformer blocks in high precision. "
            "See skip_first_n_blocks for the ~15% high-precision policy from the "
            "NVFP4 training paper (the tail blocks matter most)."
        },
    )
    save_nvfp4: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store eligible weights NVFP4-packed (qdata + scales) in "
            "a torch.save sidecar (nvfp4_packed.pt) on every checkpoint/final save, "
            "for ~3.5x smaller weight files. safetensors cannot serialize the FP4 "
            "tensor subclass, so the packed weights go to a sidecar and the bf16 "
            "weights are dropped from the safetensors shard. For FROZEN weights "
            "(LoRA/QLoRA base, lm_head, embeddings) the FP4 IS the faithful stored "
            "form — load-back is bit-exact. For FFT (NVFP4Linear keeps a bf16 master) "
            "this is LOSSY for resume: only the FP4 packing is kept, no bf16 master, "
            "so a save_nvfp4 FFT checkpoint is for storage/inference export, not exact "
            "resume. OFF by default (bf16 save, unchanged)."
        },
    )
