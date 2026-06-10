"""NVFP4-GEMM training config schema.

Real low-precision COMPUTE (FP4 forward/backward GEMMs on Blackwell), distinct
from the fake-quant QAT/PTQ `quantization:` block.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    fused_fp4_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Requires quantize_lm_head. Fuse the FP4 lm_head "
            "projection with the cross-entropy loss so the full [batch*seq, vocab] "
            "logit tensor is never materialized (~1 GiB at vocab 152k / seq 8k): "
            "the loss is computed by tiling over the vocab, dequantizing each FP4 "
            "weight tile on read, and accumulating logsumexp in fp32 (like "
            "cut_cross_entropy, but reading the NVFP4-packed weight directly). "
            "This is a MEMORY win (no logit materialization), not an FP4-GEMM "
            "throughput win — the per-tile matmul runs in bf16/fp32, so the lm_head "
            "does not hit FP4 tensor cores. Frozen lm_head only (returns dL/dhidden, "
            "no weight grad). Falls back to the materialized path if the lm_head "
            "store isn't row-sliceable (MSLK-swizzled scales) or carries a bias. "
            "OFF by default."
        },
    )
    fused_ce_vocab_block: int = Field(
        default=4096,
        gt=0,
        json_schema_extra={
            "description": "Vocab-tile width for fused_fp4_cross_entropy. The fused "
            "CE streams the vocabulary in [tokens, fused_ce_vocab_block] tiles; this "
            "is a pure speed<->VRAM dial and is loss-invariant (the tile loop is a "
            "reduction split, so loss/grad are bit-stable across block widths). "
            "4096 (default) is balanced — lighter than a bf16 head's transient logit "
            "tile while still faster. 8192 is max throughput (~+0.4% speed for ~+2 "
            "GiB peak at long seq); wider buys nothing. Overridable at runtime via "
            "the AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK env var (env wins over this "
            "config). Only affects fused_fp4_cross_entropy."
        },
    )
    bf16_lm_head_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen bias-free nn.Linear lm_head "
            "training forward to compute cross-entropy by tiling over the vocab in "
            "bf16, avoiding full [batch*seq, vocab] logits materialization (~1 GiB "
            "at vocab 152k / seq 8k) and the matching logits-gradient GEMM. The "
            "per-tile lm_head matmul runs in plain bf16 (bit-for-bit the same "
            "arithmetic as the materialized hidden @ W.t()); logsumexp/softmax and "
            "the dL/dhidden accumulation are kept in fp32. This is the exact tiled "
            "CE gradient (no low-probability vocab filtering), so it is "
            "convergence-safe under NVFP4 stochastic-rounding grads where the fused "
            "cut_cross_entropy / Liger paths collapsed. Returns dL/dhidden only (no "
            "lm_head weight grad). Incompatible with quantize_lm_head, "
            "fused_fp4_cross_entropy, and the FP8 cross-entropy patch. This is a "
            "MEMORY/backward-traffic win, not an FP4 tensor-core throughput win. "
            "OFF by default."
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
            "description": "Patch a plain frozen bias-free nn.Linear lm_head "
            "training forward to compute cross-entropy with FP8 scaled-matmul vocab "
            "tiles, avoiding full [batch*seq, vocab] logits materialization. "
            "Returns dL/dhidden only (no lm_head weight grad); backward uses FP8 "
            "scaled matmul against a prepacked dgrad weight layout. Incompatible "
            "with quantize_lm_head, "
            "fused_fp4_cross_entropy, and other cross-entropy optimization patches. "
            "OFF by default."
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
    fused_fp4_cross_entropy_fp4_matmul: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Experimental. When fused_fp4_cross_entropy is enabled, "
            "use per-tile FP4 torch._scaled_mm for lm_head logits before the tiled "
            "online logsumexp. This hits Blackwell FP4 tensor cores for the matmul, "
            "but still materializes each [tokens, vocab_block] tile and still uses "
            "a dequantized weight tile for dhidden in backward. Early microbenchmarks "
            "show it is faster than the memory-only FP4 CE path but slower than "
            "materialized bf16/Liger CE; keep this off unless benchmarking the "
            "next native CE-epilogue design."
        },
    )
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
