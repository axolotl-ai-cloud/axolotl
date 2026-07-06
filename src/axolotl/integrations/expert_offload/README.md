# Expert Offload Integration

Stream a MoE model's **frozen 4-bit experts** from pinned CPU RAM to the GPU one block at a time,
so a model whose experts exceed VRAM can QLoRA-train on a single small GPU.

Only the frozen experts move. Attention, the router/gate, norms, and the trainable LoRA adapters
stay GPU-resident, so per-step PCIe traffic is limited to the experts — which are the bulk of a
MoE's parameters and the reason it doesn't fit.

## How it differs from `layer_offloading` / `activation_offloading`

| Feature | Offloads | Granularity |
|---|---|---|
| `activation_offloading` | activations | per checkpoint boundary |
| `layer_offloading` | frozen params of **whole decoder layers** | per layer (attention + experts + norms) |
| **`expert_offload`** | frozen **4-bit experts only** | per MoE block |

For a MoE, the experts dominate memory, so offloading *only* them recovers nearly all the VRAM of
whole-layer offload while streaming far fewer bytes per step (attention/router/norms never leave the
GPU). It is tuned for the bitsandbytes `Linear4bit` + gradient-checkpoint-recompute path. It
composes with `activation_offloading`; it is orthogonal to `expert_parallel` (which shards experts
*across* GPUs — the opposite regime).

## Requirements

- One GPU per replica: single-GPU, or plain DDP for multi-GPU data parallel (no FSDP /
  DeepSpeed / expert-parallel — those shard or move the same weights this plugin manages).
- `load_in_4bit: true` with `adapter: qlora` — it offloads 4-bit `Linear4bit` experts.
- `gradient_checkpointing: true` with `gradient_checkpointing_kwargs.use_reentrant: false`. This is
  **required**, not just recommended — see "How it works". Set it explicitly: axolotl defaults
  `use_reentrant` to `true` for a QLoRA run when the kwargs are omitted, and the plugin errors on a
  reentrant config. The `gradient_checkpointing: offload` / `offload_disk` variants are untested
  with this integration.
- A per-expert MoE layout (`experts` is a `ModuleList`): Mixtral, Qwen2/3-MoE, OLMoE, DeepSeek-MoE,
  Jamba, etc. Fused-expert layouts (GPT-OSS, DBRX) are not 4-bit-quantized as `Linear4bit` and are
  left untouched.

## Usage

```yaml
plugins:
  - axolotl.integrations.expert_offload.ExpertOffloadPlugin

expert_offload: true
# expert_offload_pin_memory: true   # default; set false only if pinned RAM is scarce

load_in_4bit: true
adapter: qlora
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
```

See `examples/` for a full config.

### Multi-GPU (plain DDP)

The same config runs data-parallel unmodified — launch across N GPUs and each rank keeps its own
full replica, homes its own pinned copy of the experts, and streams to its own device:

```bash
axolotl train examples/olmoe-1b-7b-qlora-expert-offload.yaml   # auto-detects visible GPUs
```

Notes:

- **CPU RAM scales with world size** (one pinned home set per rank): budget
  `world_size x total 4-bit expert bytes` of pinned RAM.
- The offloaded expert weights are registered on DDP's parameter ignore list at install, so
  DDP's initial module-state sync never touches the evicted 0-element placeholders; they are
  frozen (`requires_grad=False`) and never enter gradient buckets.
- The per-step NCCL all-reduce covers only the trainable LoRA parameters (a few tens of MB), so
  contention with the expert H2D copies over PCIe is small; measure on your topology if the
  interconnect is shared.
- FSDP / DeepSpeed / expert-parallel remain unsupported and are refused at config validation.

## How it works

Each expert is a `Linear4bit` whose big tensor is the packed `weight.data`; its `quant_state`
scales are ~1/32 the size and stay resident. Every offloaded block's `weight.data` is homed in
pinned CPU RAM. A forward **pre-hook** on the MoE block copies its experts to the GPU just before
the block runs. Eviction is driven by a **single-resident-slot** policy — staging a block first
evicts the previously-staged one — and there is deliberately **no evict post-hook**: a block is
never dropped until the *next* block stages.

**Why gradient checkpointing is required.** `bnb.matmul_4bit`'s backward re-reads the packed weight
(to re-dequantize for the input gradient) via `save_for_backward`, and eviction repoints
`weight.data` at a 0-element placeholder. Gradient checkpointing (`use_reentrant=False`) discards
the initial-forward saved tensors and **recomputes** each layer in backward, re-running the block
pre-hook to re-stage its experts and rebuild the saved tensors from the staged weights. Because
nothing evicts a block until the next block stages — which, in backward (processed
last-layer-first), only happens after the current block's recomputed backward has finished — the
staged weights are always present when read, without depending on exactly when PyTorch stops a
recompute. Without checkpointing, backward would run against the initial graph whose saved weight is
now a placeholder, and every staged weight would stay pinned as a saved tensor to the full footprint
offload exists to avoid.

## Measured effect

OLMoE-1B-7B QLoRA on a single 12 GB RTX A2000 (r=8, seq 256, 60 steps, same seed and data order for
both arms — so the loss curves are directly comparable):

| config | loaded GPU | peak GPU | held-out eval (before → after) |
|---|--:|--:|--:|
| experts resident | 4.70 GB | 6.00 GB | 1.6448 → 1.2213 |
| experts offloaded | **1.08 GB** | **2.60 GB** | 1.6448 → 1.2270 |

Peak VRAM −57%, load-time footprint −77%; convergence preserved (identical `before`; final-loss gap
within fp noise of the dequant/reload path). Throughput cost is the per-block host→device copy — a
memory-for-compute trade — measured at roughly +11% s/step uncontended on this card. OLMoE fits a
12 GB card either way, so it is the measurable proxy; the same mechanism is what lets the larger
30B-A3B / 26B-A4B configs fit at all.
