# DiffusionGemma Block-Diffusion Training Plugin

Fine-tune Google's **DiffusionGemma** — a 26B-A4B Mixture-of-Experts *block-diffusion*
model — in Axolotl, with LoRA and fused ScatterMoE expert kernels.

Requires `transformers >= 5.11.0` (the release that adds `DiffusionGemmaForBlockDiffusion`).

## How DiffusionGemma trains

DiffusionGemma is an **encoder-decoder** model, unlike a standard causal LM:

- The **encoder** autoregressively consumes the prompt prefix into a KV cache.
- The **decoder** denoises a fixed-length `canvas` (a block of `canvas_length` tokens,
  256 for the released checkpoint) with *bidirectional* attention, cross-attending to
  the encoder KV cache. It also takes optional **self-conditioning** logits from a
  previous denoising step.

The model exposes no `labels`/loss — the training objective lives in this plugin.

### The objective

The forward (noising) process mirrors DiffusionGemma's inference sampler
(`EntropyBoundSampler`): at diffusion time `t ∈ (0, 1]` each canvas position is
independently corrupted with probability `t`. Corruption **uniformly resamples** a
token from the vocabulary (not an absorbing `[MASK]` — an absorbing variant is
available via `corruption: mask`). The decoder is trained to recover the clean token
at the corrupted positions with a reweighted cross-entropy:

```text
loss_i = w(t_i) · (1 / N_i) · Σ_{j corrupted} CE(logits_ij, x0_ij)
```

where `w(t) = 1/t` (`loss_weighting: elbo`) or `1` (`uniform`).

With probability `self_conditioning_prob`, an extra no-grad forward pass produces the
self-conditioning logits fed to the real (grad) pass, matching inference.

## Quickstart

```bash
axolotl train examples/diffusion-gemma/26b-a4b-lora.yaml
```

```yaml
plugins:
  - axolotl.integrations.diffusion_gemma.DiffusionGemmaPlugin
  - axolotl.integrations.kernels.KernelsPlugin   # optional, for fast MoE kernels

block_diffusion:
  canvas_length: 256        # defaults to the model config's canvas_length
  corruption: uniform       # "uniform" (matches inference) | "mask"
  mask_token_id: null       # required when corruption: mask
  loss_weighting: elbo      # "elbo" (1/t) | "uniform"
  self_conditioning_prob: 0.5
```

The plugin forces `type_of_model: DiffusionGemmaForBlockDiffusion` automatically
(it is not an `AutoModelForCausalLM` head) and installs the canvas data collator.

## Data

Provide single-response `chat_template` data (prompt → completion). The collator
splits each example into the encoder prefix (the prompt, `labels == -100`) and the
decoder canvas (the assistant answer). Long answers are trained block-by-block: a
random `canvas_length` block is selected per step and the preceding answer tokens
join the prefix.

## Fast kernels (ScatterMoE)

DiffusionGemma's experts use the transformers `ExpertsInterface`, so the fused
ScatterMoE LoRA kernels apply with **no model-specific code**:

```yaml
use_kernels: true
use_scattermoe: true
lora_target_parameters:     # LoRA on the routed experts (3D nn.Parameter tensors)
  - experts.gate_up_proj
  - experts.down_proj
```

The forward matches the reference grouped-GEMM path to within fp32 kernel noise.

## Multimodal (image → text)

DiffusionGemma's encoder is multimodal (Gemma 4 vision block). Because images live in
the prompt, their tokens are encoded into the prefix while only the text answer is
denoised in the canvas. The collator carries `pixel_values` / `image_position_ids`
through and slices `mm_token_type_ids` to the prefix; the trainer forwards them to the
encoder. See `examples/diffusion-gemma/26b-a4b-sudoku-multimodal.yaml` (a picture of a
sudoku puzzle → the solved grid as text).

## Quantized (NVFP4) checkpoints

The published `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` checkpoint is
`compressed-tensors` `nvfp4-pack-quantized`, but in the **pre-fusion** layout
(transformers 5.8.0.dev0): experts are stored as *per-expert* Linears
(`experts.0.gate_proj.weight_packed`) while the released 5.11 model uses *fused 3D*
experts. So it does not load directly. Convert it once:

```bash
python scripts/convert_diffusiongemma_nvfp4_to_fused.py \
    --src RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
    --dst ./outputs/diffusiongemma-26B-A4B-it-fused-bf16
```

This decompresses each NVFP4 module to bf16 and fuses the per-expert weights into
`experts.gate_up_proj` / `experts.down_proj`, then trains LoRA over the frozen base
(`examples/diffusion-gemma/26b-a4b-nvfp4-qlora.yaml`). The NVIDIA ModelOpt NVFP4
checkpoint (`nvidia/diffusiongemma-26B-A4B-it-NVFP4`) has the same per-expert layout
and converts the same way.

### Frozen 4-bit experts (`block_diffusion.frozen_fp4_experts`)

Set `frozen_fp4_experts: nvfp4` (or `mxfp4`) to quantize the fused experts to a frozen
**torchao** FP4 tensor on load, which ScatterMoE consumes directly via its selective
dequant (no bitsandbytes round-trip — this is the path from PR #3663). The plugin
re-ties the encoder experts to the quantized decoder params.

Set `lora_target_parameters: [experts.gate_up_proj, experts.down_proj]` to train LoRA on
the frozen FP4 experts. ScatterMoE fuses the LoRA in-kernel (selective dequant of only the
active experts — no full-weight merge): the kernels integration patches
`peft...ParamWrapper.forward` so the experts-interface dispatch hands the LoRA A/B to the
fused kernel instead of materializing `base + delta`
(`scattermoe_lora/experts_lora_fastpath.py`; a dequantize-add in `torchao_fp4_add.py`
remains as a safety net for `merge_and_unload`).

Validated end-to-end on the **full 26B on a single GPU**: frozen NVFP4 experts + attention
& expert LoRA train at ~38 GiB peak (the non-fused merge path OOM'd 96 GiB). This is the
memory-efficient QLoRA-over-frozen-NVFP4 path.

## Training requirements & limitations

Validated end-to-end (data → collator → loss → backward → optimizer step) on the real
26B model on a single 96GB GPU. DiffusionGemma needs a few non-default settings, which
the example configs set:

- **`gradient_checkpointing: false`** — required. The encoder builds a KV cache the
  decoder reads; gradient checkpointing on the encoder discards that cache mid-forward,
  so the decoder sees an empty cache.
- **`flash_attention: false`** (use sdpa) — the global-attention layers use
  `head_dim=512`, above flash-attn's 256 limit.
- **`lora_qkv_kernel` / `lora_o_kernel` / `lora_mlp_kernel: false`** — the fused LoRA
  attention/MLP kernels look for a `{Model}Attention` class; DiffusionGemma names its
  attention `DiffusionGemma{Encoder,Decoder}TextAttention`. The ScatterMoE *expert*
  kernel is independent and stays on.
- The plugin auto-patches three DiffusionGemma-specific load issues: tied-expert
  `tie_weights` under `quantize_moe_experts`, the missing
  `prepare_inputs_for_generation` (PEFT), and the missing `generation_config.bos_token_id`
  (HF Trainer); and it disables the bf16 `caching_allocator_warmup` pre-allocation.
- **4-bit-frozen base (true QLoRA):** loading + bnb-4bit expert quantization works, but
  ScatterMoE's forward does not yet route bitsandbytes-4bit experts through its
  selective-dequant path (only torchao MXFP4/NVFP4). Until wired, use the bf16 base.
- Single-response data is assumed; multi-turn answers are treated as one answer span.
- `sample_packing` is not used (the collator builds prefix/canvas batches).
