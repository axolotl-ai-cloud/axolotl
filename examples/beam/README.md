# Beam — LoRA & QLoRA

`BeamForCausalLM` / `ReflectionForCausalLM` (`model_type: beam` upstream, `reflection` in Reflection's fork). ~501B total / ~23B active.
Requires Reflection's transformers fork, which ships the architecture in-tree —
there is no remote code and `trust_remote_code` is not needed.

> **Naming.** Upstream transformers adds this architecture as `beam`
> (`BeamForCausalLM`, `BeamDecoderLayer`); Reflection's fork calls it `reflection`.
> The descriptor registers both, so either checkpoint resolves. Configs here set
> `transformer_layer_cls_to_wrap: ReflectionDecoderLayer` to match the current
> checkpoint — switch it to `BeamDecoderLayer` on an upstream one.

**Status: QLoRA trains, on both paths.** Verified on 8x H200 at full scale:
FSDP2 with attention + expert adapters (3.6B trainable, see
[Verified runs](#verified-runs)), and `device_map` with attention-only adapters.

| | |
|---|---|
| Layers | 52 — 0–2 dense, 49 MoE (`decoder_sparse_step: 1`) |
| Experts | 256/layer, 8/token, grouped top-k (`n_group: 8`, `topk_group: 4`) |
| Routing | `noaux_tc` + sigmoid + `routed_scaling_factor: 2.5` (DeepSeek-V3 lineage) |
| Attention | hybrid sliding (window 512) + `full_attention` every 6th layer; GQA 48/8 |
| Distinctive | attention output gate, `ssmax`, sandwich norm, depth-scaled residuals, `final_logit_softcapping: 30`, fp32 lm_head and router |
| Checkpoint | ~1 TB bf16, multi-shard safetensors |

## Configs

| File | Path | Status |
|---|---|---|
| `beam-qlora-fsdp-staged.yaml` | FSDP2 + CPU-staged NF4 loader, attention + expert LoRA | **works**, preferred |
| `beam-qlora-mp.yaml` | device_map model parallel, attention only | **works**; no data parallelism |
| `beam-qlora-experts-mp.yaml` | device_map model parallel, + expert LoRA | untested at scale |
| `beam-tiny-smoke.yaml` | tiny random-weight model | works; ~1 min, for fast iteration |

## Dependency

Runs on axolotl `main`, with no patches to axolotl internals. Everything this
needed has landed upstream:

- #4003 — CPU-staged NF4 loader, and the chunked bitsandbytes quant/dequant a
  6.7e9-element fused expert tensor requires
- #4018 — skip re-initializing parametrized quantized weights
- #4091 — `no_split_module_classes` from `_no_split_modules`, without which a
  decoder layer could be split across GPUs and its residual add raised

This branch is therefore only the model descriptor, the CCE patch and the configs.

## Why the LoRA targets look unusual

At runtime the experts are **not** linear modules. `ReflectionExperts` stores them
as fused 3D `nn.Parameter` tensors:

```
experts.gate_up_proj   (256, 2*2560, 5120)
experts.down_proj      (256, 5120, 2560)
```

The unfused per-expert weights in the checkpoint are fused at load time by the
`"reflection": "qwen2_moe"` conversion mapping. Consequences:

1. **Experts need `lora_target_parameters`** (`mlp.experts.gate_up_proj`,
   `mlp.experts.down_proj`) — `lora_target_modules` cannot reach them.
2. **`lora_dropout` must be 0** — PEFT wraps 3D params in `ParamWrapper`, which
   rejects nonzero dropout. Axolotl validates this.
3. **The router is safe by construction.** `ReflectionTopKRouter.weight` is also a
   bare `nn.Parameter`, so linear discovery never sees it. Do not add it to
   `target_parameters`: adapting a `noaux_tc` router destabilizes routing.

**Never use `lora_target_linear: true`.** It is rejected alongside
`quantize_moe_experts`, and `gate_proj` is ambiguous here — it names both the
attention output gate (`self_attn.gate_proj`) and the dense MLP (`mlp.gate_proj`),
which PEFT matches by suffix.

## Memory

Experts are 98.4% of the weights, so quantizing them is what makes this fit.

| | Size |
|---|---|
| bf16 weights | ~1004 GB |
| nf4 weights | ~254 GB (~32 GB/GPU across 8) |
| attention + dense MLP only | ~6.4 GB |

Note the reported parameter count is misleading: axolotl prints ~255B "all params"
because 4-bit packing stores two values per byte. Real trainable fraction for
attention-only is ~0.006%, not the 1.4% implied by that denominator.

## Settings that are not optional

- **`optimizer: adamw_torch`** — `adamw_torch_4bit` (torchao) indexes optimizer
  state without regard to device and raises once `device_map` spreads adapters
  across GPUs. Adapters are small, so 4-bit optimizer state buys nothing here.
  For the expert config (~3.6B trainable, ~29 GB of AdamW state) try
  `adamw_bnb_8bit` if memory is tight.
- **`gpu_memory_limit: 140GiB`** — `infer_auto_device_map` estimates sizes on a
  **bf16** canvas (~1004 GB), not the ~254 GB nf4 reality. The limit must be large
  enough for the *estimate* to fit across the GPUs or accelerate spills layers to
  CPU and bitsandbytes refuses to load.
- **`attn_implementation: sdpa`** — no flash-attn build exists for
  torch 2.13.0+cu130 in the current image and the validator rejects
  `flash_attention_2` up front.
- **`sample_packing: false`** — `ssmax` scales attention off `position_ids`, which
  packing rewrites per document. Unverified interaction; test before enabling.

## Verified runs

### FSDP2 CPU-staged loader (preferred)

`beam-qlora-fsdp-staged.yaml`, 8x H200, 8 ranks, `sequence_len: 2048`,
`micro_batch_size: 1`, attention + expert adapters:

```
NF4 checkpoint loading and quantization   45 min   (rank 0, cold page cache)
NF4 adapter initialization                42 s
NF4 parameter shard distribution          240 s
trainable params: 3,625,582,592
loss 2.147 -> 2.038 -> 1.559 -> 1.115 -> 0.994   (train_loss 1.571)
5/5 [02:36, 27.5s/it]   max_active 39 GiB, reserved 121 GiB per GPU
adapter_model.safetensors  14 GB   (612 tensors: 416 attention, 196 expert, fp32)
```

Each step processes one sample per rank, so per-sample throughput is ~2x the
`device_map` path even before packing or a larger micro batch.

The shard distribution used to hang at this model size (one broadcast per owner,
N-1 of them discarded, serial slicing on rank 0). Upstream #4003 as merged
replaced it with a padded `dist.scatter`, and the loader now opens its own
staging control group with the `ddp_timeout` value, so the 1800 s NCCL bootstrap
barrier that used to expire during rank 0's staging no longer applies. Keep
`ddp_timeout: 21600` anyway; staging is 45 min cold.

`state_dict_type` must stay `SHARDED_STATE_DICT`: `FULL_STATE_DICT` with
`quantize_moe_experts` crashes on the first checkpoint save
(axolotl-ai-cloud/axolotl#4080, reproduced). The final adapter is gathered and
saved as a normal PEFT adapter regardless.

### device_map model parallel

`beam-qlora-mp.yaml`, 8x H200, `sequence_len: 2048`, `micro_batch_size: 1`:

```
Loading weights: 832/832 [04:15]      (warm page cache; ~45-80 min cold)
trainable params: 28,966,912
loss 1.75 -> 2.201 -> 2.632 -> 1.61 -> 1.528   (train_loss 1.944)
5/5 [00:36, 7.39s/it]   peak 52 GiB reserved/GPU
adapter_model.safetensors  116 MB
```

Loss in the 1–2.6 band is the correctness signal: a pretrained model doing SFT
should start at 0.5–2.0, while `log(vocab_size)` ~= 11.9 would mean random weights.
The tiny random-weight smoke configs sit at ~11.9 as expected.

`device_map` gives no data parallelism and must be launched with
`--num-processes 1`. Checkpoint saving with quantized experts works on this path
(verified on the tiny model: attention + expert adapter tensors in both the
step checkpoint and the final save).

## Declared capabilities

`src/axolotl/model_support/reflection/` marks these broken rather than letting
their generic fallbacks run silently:

- **`cut_cross_entropy`** (experimental, warns) — the descriptor's
  `BEFORE_MODEL_BUILD` hook registers `cut_cross_entropy.py` into CCE's
  `PATCH_FNS`, so no change to ml-cross-entropy is needed. `apply_lce` applies the
  logit softcap. Measured 87% lower peak memory at seq 2048 on the tiny model, loss
  matching to 1.5e-05. The `lm_head_dtype: fp32` path is unverified.
- **`lora_kernels`** — the fused QKV/O kernels match a llama-like projection set
  by name and would bypass the attention output gate.
- **`liger`** — its RMSNorm/SwiGLU replacements carry neither the sandwich norms
  nor the depth-scaled residuals, and expect per-expert `nn.Linear`.
- **`sdpa_varlen`** (experimental, warns) — sliding window needs a torch build with
  `varlen_attn(window_size=...)`, and `ssmax` reads `position_ids`.
