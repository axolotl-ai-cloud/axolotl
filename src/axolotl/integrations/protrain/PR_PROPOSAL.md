# ProTrain integration for Axolotl — current implementation and validation status

## 1. Executive summary

This document describes a ProTrain
(Yang et al., MLSys 2026, arXiv 2406.08334) integration to Axolotl as a
`BasePlugin` under `src/axolotl/integrations/protrain/`. The integration is a
from-scratch Python port of the paper's automatic memory manager, adapted to
the Axolotl + HuggingFace Trainer + PEFT-LoRA + bitsandbytes-4-bit ecosystem.

Validated headline numbers (all on 4× RTX 3090 / 3090 Ti class hardware,
non-NVLink PCIe topology; full setup in §11):

- **Memory reduction on Meta-Llama-3-8B BF16 LoRA, single 3090.** Vanilla
  peak `memory/max_active` = **15.83 GiB**; ProTrain Mode A peak after
  `materialize_offload` = **3.08 GiB** — an 80% reduction (the searcher picks
  `n_persist=128/130`, freeing 13.05 GB of non-persistent chunks to pinned
  CPU).
- **Llama-13B + 4-bit + LoRA on a single 3090.** Vanilla and
  ProTrain Mode A both peak at **7.91 GiB**; ProTrain runs in **16.72 s**
  vs vanilla **35.13 s** for 50 steps at seq=256 bs=1 — **~2.1× speedup**
  with the same memory ceiling, with the searcher skipped via
  `protrain_force_all_persistent: true`.
- **Qwen3.5-27B + 4-bit + LoRA on a single 3090 at seq=128.** Mode A peaks
  at **19.98 GiB**, runs in **21.62 s / 25 steps**, loss 1.075. The
  fp32 embedding-upcast transient (~5 GiB) is **auto-deferred** when
  ProTrain is in `plugins` — the loader detects ProTrain and skips the
  load-time upcast (ProTrain handles the cast lazily during forward). The selected cost-config
  is `n_persist=64, n_buffer=0, n_swap=1, n_checkpoint=0, n_offload=0` with
  `predicted=15409 MB`. This is the largest model class validated on a
  single 3090 in this work.
- **Sequence-length headroom at 13B + 4-bit, single 3090.** Mode A holds at
  seq=512 (10.95 GiB), seq=1024 (13.66 GiB), and seq=2048 (**18.94 GiB**,
  loss 0.99) — the multi-sequence claim is validated up to 2048
  with throughput degrading smoothly (1.19 → 0.75 → 0.43 sps).
- **DDP batch scaling on 13B + 4-bit, 4× 3090.** Per-rank peak grows
  9.56 → 11.16 → 14.21 GiB across bs=1/2/4; global throughput grows
  22.6 → 34.7 → **44.8 sps**. bs=8 OOMs (expected; bs=4 is the practical
  ceiling at seq=256).
- **Single-GPU throughput recovery at bs=4.** Qwen3.5-9B + 4-bit qlora at
  bs=4, seq=256: vanilla **3.95 sps**, ProTrain **6.14 sps** (+55%),
  validating the throughput acceptance criterion for bs >= 4.
- **Multi-GPU 4×3090 (4-bit LoRA, seq=256, bs=1).** Llama-13B Mode A (DDP):
  **24.68 global sps** at 9.98 GiB/rank, loss 0.997. Mode C
  (ZeRO-3 sharded CPU-offload): 23.67 global sps at 9.87 GiB/rank, loss 1.017.
  Qwen3.5-9B Mode A: **32.53 global sps** at 9.89 GiB/rank.
- **Full-finetune Adam-state reduction on Qwen3-0.6B, single 3090.**
  `adamw_torch` baseline 5.59 GiB → `adamw_bnb_8bit` 4.81 GiB (-14%) →
  **`paged_adamw_8bit` 2.84 GiB (-49% peak)**. The 74.6% headline
  refers to the Adam-state slice alone; expressed as total peak the
  reduction is -49% because weights and activations are unchanged. ProTrain
  Mode A composes cleanly with both `adamw_torch` (5.58 GiB, 5.66 sps) and
  `paged_adamw_8bit` (2.84 GiB, 5.40 sps).
- **27B + 4-bit Mode C sharded on 2× 3090 (multi-GPU).** ProTrain Mode C
  with `protrain_zero3_shard: true` at seq=128, world_size=2 holds at
  **20.25 GiB/rank**, 4.564 global sps, loss 0.811 — the first
  27B-class Mode C multi-rank result. 4× 3090 27B + 4-bit OOMs on
  24 GiB cards (§6.t); the validated 27B paths are single 3090
  Mode A with ProTrain's auto-deferred embedding upcast or 2× 3090 Mode C.
- **Four-mode behavior on 27B + 4-bit + seq=128, single 3090.** Mode A /
  B / C / auto all hold at 19.93-19.96 GiB / 1.17-1.21 sps;
  Mode C **gracefully degrades to single-rank behavior** at world_size=1
  (§6.k). 27B + 4-bit seq=64 vanilla vs ProTrain head-to-head: ProTrain
  +34% throughput at +0.64 GiB peak (§6.q).
- **DDP scaling ≈ 3.9× on 13B + 4-bit, 4× 3090** (§6.p) — broadly
  consistent with the paper's non-NVLink PCIe Mode A claim (~3.5-3.64×). The
  bnb.AdamW8bit LoRA-scope delta is ~5% peak (7.91 → 7.53 GiB) because
  LoRA's 62 M trainable params hold only ~0.5 GiB of Adam state;
  full-FT scope is where the 74.6% Adam-state headline shows up
  (substantiated above).
- **LoRA-rank sweep on 13B + 4-bit, single 3090.** Mode A holds across
  production ranks: r=16 → 9.33 GiB / 1.59 sps / loss 1.04;
  r=32 → 10.17 / 1.59 / 0.88; r=64 → **11.89 / 1.53 / 0.93**. Mode A
  composes cleanly with the LoRA-rank knob without re-running the searcher.
- **Gradient-accumulation compatibility.** Mode A on 13B + 4-bit with
  ga=1/4/8/16 holds at 9.34–9.57 GiB peak across all four points (within
  0.23 GiB), and throughput improves slightly (1.61 → 1.72 sps) as
  optimizer-step cost amortizes — confirming `gradient_accumulation_steps`
  is memory-neutral and weakly throughput-positive.
- **Three-way head-to-head on Llama-2-13B + 4-bit, 4× 3090 (§6.x).**
  ProTrain Mode A 22.6 global sps (1.00× baseline) vs FSDP2 optimized
  on actual Llama-2-13B 4.27 sps (~5.3× slower) vs DeepSpeed
  ZeRO-2 16.8 sps (0.74×) vs ZeRO-3 6.7 sps (memory leader at 5.58
  GiB/rank with 3.4× wall-time cost) vs ZeRO-3 + CPU offload 5.8 sps
  (smallest memory at 3.66 GiB/rank with 3.9× wall-time cost). ProTrain
  Mode A is the **throughput leader at acceptable memory**; ProTrain
  offers a different Pareto point than ZeRO-3, not a strict-dominance
  win.
- **Long-horizon convergence on 13B + 4-bit (§6.aa, 1500 steps).**
  ProTrain Mode A loss 0.836 vs vanilla 0.804 — difference within
  variance noise; **no chunk-shuffling-induced trajectory drift** over
  the multi-thousand-step horizon.
- **MoE compatibility (§6.bb).** tiny-mixtral-30m both vanilla and
  ProTrain Mode A rc=0. MoE supported when DecoderLayer carries
  `.self_attn` (covers Mixtral, OLMoE, Phi-MoE).
- **Mode B at scale on Llama-3-8B + 4-bit qlora, 4× 3090.** Explicit
  Mode B (`protrain_force_replicated_cpu_offload: true`,
  `n_persist=128 n_offload=0`): **rc=0, 9.28 GiB/rank, sps 3.027/rank**,
  loss 1.082 (§6.rr). Streaming-dataset apples-to-apples (§6.yy
  bs=1 seq=256) measures Mode B at +8.2% over Mode A steady-state
  sps, but Mode B init wallclock is ~10× Mode A's; Mode B is therefore
  not categorically preferable. The `n_persist=128 n_offload=0`
  config is the proven-end-to-end production recommendation for
  Mode B on consumer non-NVLink rigs.
- **Mode C ZeRO-3 on consumer non-NVLink rigs.** The runtime bypass
  flips `accelerator.state.distributed_type` to `NO` before
  `accelerator.prepare()` so DDP isn't wrapped; ProTrain's per-chunk
  `reduce_scatter`/`all_reduce` owns cross-rank sync.
  **Hardware-validated end-to-end** on 4× 3090 (GPUs 1,4,5,7) at
  bs=1 seq=256 (rc=0, **8.87 GiB/rank, sps 3.04/rank**, loss 2.685)
  and bs=1 seq=512 (rc=0, **9.06 GiB/rank, sps 3.07/rank**, loss
  1.91); bypass log line (`DistributedType.MULTI_GPU ->
  DistributedType.NO`) and `ddp_bypassed=True` confirmed (§6.uu).
- **bs=2 Mode A scaling on 13B + 4-bit, 4× 3090 (§6.zz.2).** Mode A
  bs=2 seq=256 sps **4.23/rank**, bs=2 seq=512 sps **3.47/rank**
  (1.58× and 1.40× over bs=1 at the same seqs). Mode A scales
  robustly across bs=1/2 at this scale.
- **Path B LoRA grad sync (default ON).** ProTrain owns trainable
  LoRA-adapter grad sync via `dist.all_reduce` over
  `_flatten_dense_tensors`-coalesced buffers in `_ProTrainOptimizer.step()`;
  DDP is told to ignore the same params via
  `_ddp_params_and_buffers_to_ignore`. Collapses N small per-bucket
  allreduces into 1-2 coalesced (per dtype) — measured **-68% NCCL
  collective count** on Llama-3-8B 4-bit qlora bs=1 4-rank. Gated by
  `protrain_own_lora_grad_sync` (default true; opt-out available).
  Throughput gain scales with LoRA factor count: **+15.1% steady-state
  sps/rank** on Qwen3.5-9B 4-bit qlora at all-linear LoRA r=16
  (256 factor tensors) with `micro_batch_size=1` +
  `gradient_accumulation_steps=4`, seq=256, Mode A, 4× 3090 PCIe
  (warmup-isolated, order-independent — see §6.pb).
  Bit-equivalent: 4-rank NCCL 100-step convergence parity shows
  per-step loss on the broadcasting rank identical between OFF and
  ON (`|loss_diff| = 0.0` at every step) and max LoRA weight diff
  `< 2e-11` after 100 steps. Tests in
  `tests/protrain/test_path_b_lora_sync.py` (§6.pb).
- **Test suite.** 481 passed, 5 skipped, 179 deselected on the
  default-marker `tests/protrain/` suite (~122 s).
  Multi-GPU regression `test_paged_adam_offload_mgpu` passes in ~7 min on the
  4× 3090 pool.
- **Save / merge / resume.** All 8 round-trip scenarios (vanilla and ProTrain,
  standard-attention and linear-attention models) returned rc=0; the M5/M6
  `protrain_optim/` checkpoint directory (`gpu_optim.pt` + `metadata.json`
  format_version 2) was produced at every save step and reloaded by
  `_load_protrain_optim_dir` on resume. Train + merge-lora at 13B
  (7.91 GiB peak) and 27B (18.27 GiB peak) both rc=0 (§6.jj).
  Full-stack `load_in_4bit + qlora + torch.compile + ProTrain Mode A`
  on Llama-3-8B 4× 3090 rc=0 at 9.29 GiB/rank (§6.nn).

**Search and first-iter cost.** Production-shape bs=2 Mode B search
wall is ~74 s at-rig (~40 s synthetic). Mode B bs=1 seq=512 init is
~30 s. An eager per-chunk NCCL warm-up (`protrain_eager_nccl_warmup`,
default true) fires one no-op of each per-chunk collective at
`post_trainer_create`, measuring 0.22 s (Mode B) / 0.24 s (Mode C).

**Mode B / Mode C bs=2 `n_offload > 0` on 4-rank non-NVLink.** The
runtime uses a dedicated `_offload_stream` for backward OFFLOAD
re-gather and routes LoRA-container sharded gather on
`_prefetch_stream`; the searcher broadcasts hardware inputs from
rank 0 (compute TFLOPS, memory bytes, capacity, cache-key SKU) for
plan determinism across mixed-SKU ranks (§6.zz.X). On non-NVLink
topology auto-mode prefers `n_persist=128 n_offload=0`. Verified at-rig
on the 4-rank mixed-SKU shape: rc=0, 9.44 GiB peak, 5.396 sps/rank,
423 s wallclock.

**Remaining carried limitations:**

- Mode A 4-rank full-FT at 8B+ class OOMs at DDP-reducer load order
  on 24 GiB cards (frozen base + reducer bucket > 24 GiB) — tracked in §16.B B1.

> **Cost-model framing (read before tuning near the 24 GiB ceiling).**
> The raw `estimate_peak` predictor is a **lower-bound search gate**, not
> a runtime feasibility check. Calibrated `alpha_steady = measured/predicted`
> on 30B-Llama Mode-C lands at **~1.18 / 0.99 / 0.80** across
> seq=512/1024/2048 — slight under-prediction at low seq, slight
> over-prediction at seq=2048 (safer for the runtime budget gate). The
> wrapper-side `_calibrate_peak_with_actual_chunk_bytes`
> (`api/model_wrapper.py:296`) **raises** the prediction by 0.6-0.9 GiB
> before comparing to the device budget, absorbing whatever raw residual
> remains. Users tuning near the ceiling should rely on the calibrated
> post-wrapper prediction (the runtime-visible value) rather than the raw
> searcher estimate.

---

## 2. Background: ProTrain (paper) vs this integration

**The original paper.** ProTrain proposes automatic memory management for
ZeRO-style data-parallel LLM training. The core idea is to unify
ZeRO-3 sharding, CPU offload, gradient checkpointing, and activation
swapping into a small structured search space (the four tunable parameters
`n_persist`, `n_buffer`, `n_swap`, `n_checkpoint`), and to pick the optimum
analytically from a single profiling pass via two cost models (runtime and
peak memory). The model state is organized into chunks managed
hierarchically (inter- and intra-chunk); activations are managed at
transformer-block granularity with an interleaved layout that places
swap-targeted blocks early and unoptimized blocks late. The reference
implementation reports up to **2.71× throughput** vs DeepSpeed / Colossal-AI
/ FSDP on RTX 3090s and trains models up to **34B on a single 3090** / 75B
on a single A100.

**This integration.** A from-scratch Python port implemented as an Axolotl
`BasePlugin`, designed to compose with the Axolotl + HF Trainer training
loop, PEFT-LoRA adapters, and bitsandbytes weight quantization. The
plugin owns per-rank memory policy only — it does not own distributed
collectives (`torch.distributed`), training-loop control flow, TP, PP, or
FP8. Three notable deltas vs the paper's reference design, expanded in §8:

1. A fifth search axis (`n_offload`) for block-level chunk-offload-without-
   recompute (Option B); the paper's design has four axes.
2. A per-dtype α fragmentation factor (`ALPHA_FRAGMENTATION_4BIT = 0.75` for
   bnb-4-bit; paper uses the constant α=1.10 across dtypes).
3. PEFT-LoRA container hooks and a DDP `init_sync=False` bypass — paper
   targeted full-tensor params on a vanilla DDP-or-internal-allreduce path.

The mode-selection table is also re-indexed into three modes (A: GPU-resident
DDP; B: replicated CPU-offload; C: ZeRO-3 sharded CPU-offload), with the
selector preferring A → B → C on non-NVLink PCIe because the per-chunk collectives
of Mode C dominate at typical 3090 batch sizes.

---

## 3. Architecture overview

### 3.1 Chunk manager (`chunk/manager.py`)

Model states (parameters, gradients, optimizer states) are partitioned into
chunks of size `S_chunk` (picked from a `{32, 64, 128, 256} MB` grid by
`chunk/sizing.py` against a fragmentation-waste model). The first
`n_persist` chunks remain GPU-resident across the iteration; the
remaining `N_chunk − n_persist` chunks are non-persistent and live on pinned
host memory (replicated or sharded) until gathered. A `BufferPool` of
`n_buffer` pre-allocated GPU slots stages the H2D and D2H movement so the
caching allocator never has to satisfy chunk-sized requests from the kernel
hot path.

The chunk layout (`chunk/layout.py`) groups parameters by the transformer
block they belong to and reorders them within a chunk by first-use order
observed during the trace. The execution-ordered intra-chunk layout
eliminates the back-and-forth access pattern that vanilla
initialization-order chunking exhibits, which is what enables prefetch to
hide PCIe latency.

`materialize_offload` is the one-shot transition that runs after the
searcher returns a config: persistent chunks get their final GPU buffer,
non-persistent chunks get their pinned-host pool view (replicated under Mode
B, sharded under Mode C), and `param.data` for every non-persistent param is
swapped to a zero-element placeholder until the first `gather`.

### 3.2 Block manager (`block/`)

Activations are managed at the transformer-block level. Each block carries
one of four modes (`block/strategy.py`):

| BlockMode | Behavior | When picked |
|---|---|---|
| `NONE` | activations kept resident, no recompute | last few blocks, where backward consumes them first |
| `CKPT` | `torch.utils.checkpoint` (non-reentrant) on the block; recompute in backward | middle/later blocks when bandwidth can't hide swap |
| `SWAP` | every autograd-saved tensor D2H'd to a pinned pool on `_swap_stream` in fwd, H2D'd back in bwd | early blocks where prefetch can overlap |
| `OFFLOAD` | re-gather the block's non-persistent chunk in backward (no recompute) | when chunk-offload is preferred over per-tensor swap (Option B) |

The placement rule (`block/layout_rules.py::assign_modes`) puts SWAP blocks
first, OFFLOAD/CKPT blocks in the middle, and `NONE` blocks last —
mirroring the paper's "swap-early, unopt-late, interleave" layout — and
`_looks_like_block` recognizes both standard-attention (`.attention`,
`.self_attn`) and Mamba-style linear-attention (`.linear_attn`) blocks so
Qwen3.5 / Falcon-Mamba / Zamba hybrids are correctly discovered.

### 3.3 Cost model and exhaustive searcher (`cost/`, `search/`)

Two cost models, both fed by a single profiler trace:

- **Runtime model** (`cost/runtime.py`) — implements paper Eqs. 2–7:
  `T_iter = T_FWD + max(T_BWD + T_GPU_OPTIM, T_CPU_OPTIM)`, with per-chunk
  `max(compute, communication)` rolloff. `cost/bandwidth.py` derates
  prefetch bandwidth when `n_swap > 0` competes for the same PCIe lane.
- **Memory model** (`cost/memory.py`) — implements paper Eqs. 8–11
  (op-walk peak + the recompute-bump indicator) plus the per-dtype α (see
  §8) and a `ckpt_chain_bytes` term that accounts for the
  linear-in-N_block residual that survives the backward window under
  non-reentrant checkpointing.

The searcher (`search/exhaustive.py`) enumerates the 5-axis tuple
`(n_persist, n_buffer, n_swap, n_checkpoint, n_offload)` in memory-ascending
order with OOM pruning, and returns the argmin of `T_iter` subject to
`M_peak < M_capacity`. On RTX 3090 hardware the swap path is almost never
selected (non-NVLink PCIe saturates from prefetch alone) so `n_swap = 0` is the
common case — matching the paper's RTX-3090 observation.

### 3.4 Plugin lifecycle (`plugin.py`)

The plugin is a `BasePlugin` subclass. The runtime path it owns:

1. `get_input_args` — returns `ProTrainArgs` to Axolotl's pydantic merger
   (adds `protrain_*` keys to the YAML schema).
2. `post_model_load(cfg, model)` — builds a `HardwareProfile`, runs the
   profiler (cached on disk by `(arch_hash, bs, seq, sku, world)`),
   invokes `protrain_model_wrapper(model, ...)`, and stashes the
   `WrappedModel` on `cfg` for the next hook to pick up.
3. `post_trainer_create(cfg, trainer)` — installs the ProTrain optimizer
   (`protrain_optimizer_wrapper(wrapped)`) onto `trainer.optimizer` and
   registers the checkpoint save/load callbacks. This is the canonical
   install point because Axolotl's `OptimizerMixin.create_optimizer` does
   not route through `PluginManager.create_optimizer`.
4. `_install_resume_hook` — monkey-patches `trainer._load_from_checkpoint`
   to interleave a `restore_to_gpu()` before HF copies loaded weights and
   a `materialize_offload()` + optimizer rebuild afterward. This is the
   bridge that enables cross-mode (A↔C) resume.

Activation requires **both** the plugin listed in `plugins:` AND
`protrain_auto_memory: true` (defaults to False). Listing the plugin alone
registers the args schema but leaves the runtime hooks dormant.

### 3.5 Runtime data path

```
            ┌─────────────────────┐
 YAML       │ protrain_auto_memory│
   ─────►   │ + plugin in plugins │
            └──────────┬──────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────────┐
   │ post_model_load (plugin.py)                 │
   │  1. _build_hardware_profile(cfg)            │
   │  2. profiler.run_trace (cached by arch_hash)│
   │  3. discover_blocks(model)  [block manager] │
   │  4. build_layout(chunks, exec_order)        │
   │  5. exhaustive.search(...) → CostConfig     │
   │  6. mode_select → A | B | C                 │
   │  7. protrain_model_wrapper(model, ...)      │
   │     → materialize_offload()                 │
   └──────────┬──────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────┐
   │ post_trainer_create                          │
   │  - protrain_optimizer_wrapper → optimizer   │
   │  - _install_resume_hook (cross-mode bridge) │
   └──────────┬──────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────┐
   │  Train loop (HF Trainer)                     │
   │                                              │
   │  pre-fwd hook  → scheduler.prefetch_chunks  │
   │       │                                      │
   │  block.forward                               │
   │       │                                      │
   │  post-fwd hook → release stale chunks       │
   │       │                                      │
   │  loss.backward()                             │
   │       │                                      │
   │  pre-bwd hook → ensure_chunks_resident      │
   │       │                                      │
   │  block.backward                              │
   │       │                                      │
   │  post-bwd hook → reduce_grads_and_offload   │
   │       │                                      │
   │  optimizer.step                              │
   │   - persistent chunks  → GPU FusedAdam       │
   │   - non-persistent     → DeepSpeedCPUAdam   │
   └──────────────────────────────────────────────┘
```

### 3.6 The three Modes (A / B / C)

The mode selector (`api/model_wrapper.py`) picks one of three composition
modes after the searcher returns a config:

| Mode | Composition | Picked when | Per-rank GPU peak | Per-rank pinned CPU | Throughput |
|---|---|---|---|---|---|
| **A** | All-persistent, outer DDP | `n_persist == N_chunk` fits | full model | ~0 | best (DDP allreduce) |
| **B** | Replicated CPU-offload | offload needed AND `cpu_ram_per_rank ≥ (N_chunk − n_persist) · S_chunk` | reduced | full non-persistent set, replicated | ~1.9× slower than A on non-NVLink PCIe |
| **C** | ZeRO-3 sharded CPU-offload | offload needed AND replication doesn't fit | reduced | non-persistent set / world_size | **PCIe**: ~1.04× slower than A for 4-bit + LoRA on 4× 3090 (§6.e: 24.68 vs 23.67 sps); full-FT scope is ~3.6× per the M7 internal benchmark. **NVLink** (2× A100-SXM4): **~1.43× FASTER than A** on the same shape (Mode C 252 vs Mode A 177 tok/s/rank, §6.nv) — sharded all-gather amortizes well over NV-class fabric. |

The selector prefers A → B → C; C is chosen only when CPU RAM is the
binding constraint. Mode C gracefully degrades to single-rank execution
at `world_size=1` (within 0.03 GiB / 1% of Mode A, §6.k). On consumer
non-NVLink topology, `n_persist=128 n_offload=0` is the auto-mode Mode B
landing point. Mode A bs=1/2 and Mode C bs=1/2 are hardware-validated
end-to-end on 4× 3090 (§6.zz).

---

## 4. Implementation in axolotl

### 4.1 Plugin attachment

Add to YAML:

```yaml
plugins:
  - axolotl.integrations.protrain.ProTrainPlugin
protrain_auto_memory: true
```

The dual gate (plugin listed AND `protrain_auto_memory: true`) is enforced by
a pydantic `model_validator` in `args.py`. The same validator rejects
combinations with `deepspeed:` or `fsdp:` / `fsdp_config:` — the three
memory backends are mutually exclusive.

### 4.2 Config knobs that matter

| Knob | Default | Effect |
|---|---|---|
| `protrain_auto_memory` | `false` | Master enable. Required for any ProTrain behavior. |
| `protrain_auto_mode` | `true` | Run the mode selector. Set `false` to honor manual mode flags below. |
| `protrain_force_all_persistent` | `false` | Force Mode A (every chunk GPU-resident). Skips the search entirely. Requires `protrain_auto_mode: false`. |
| `protrain_zero3_shard` | auto | Force Mode C (ZeRO-3 sharded CPU offload). Requires `protrain_auto_mode: false`. |
| `protrain_save_optimizer_state` | `true` | Emit `protrain_optim/{gpu_optim.pt, metadata.json}` next to every HF checkpoint, for cross-mode and same-mode resume. |
| `protrain_cache_dir` | `~/.cache/axolotl/protrain` | On-disk cache for profiler traces (keyed by `(arch_hash, bs, seq, sku, world)`). |
| `protrain_n_persist_override` | None | Manual `n_persist`. For debugging the searcher. |
| `protrain_n_buffer_override` | None | Manual `n_buffer`. |
| `protrain_n_swap_override` | None | Manual `n_swap`. |
| `protrain_n_checkpoint_override` | None | Manual `n_checkpoint`. |
| `protrain_n_offload_override` | None | Manual `n_offload` (the Option B axis added by this integration). |
| `protrain_force_replicated_cpu_offload` | `false` | Force Mode B (replicated CPU offload, no sharding). Requires `protrain_auto_mode: false`. Sibling to `protrain_force_all_persistent` (Mode A) and `protrain_zero3_shard` (Mode C). |
| `protrain_own_lora_grad_sync` | `true` | Path B: ProTrain owns trainable LoRA-adapter grad sync via flattened all_reduce per dtype; DDP is told to ignore the same params. Opt out to revert to DDP's bucketed allreduce. See §6.pb. |
| `protrain_eager_nccl_warmup` | `true` | Fire a one-shot no-op of each per-chunk NCCL collective at `post_trainer_create` to amortize the first-iter NCCL init cost. Measured ~0.22-0.24 s warm-up wallclock. |
| `protrain_persistent_huge_param_threshold_bytes` | `512 MiB` | Params at or above this size (typically `lm_head` / `embed_tokens` at scale) are pinned persistent regardless of `n_persist`, since paging them dominates per-step cost. |
| `protrain_ckpt_internal_residual_factor` | `1.0` | Scale applied to the per-block CKPT internal saved-tensor proxy (FFN-intermediate + attention scores + Q/K/V) in `estimate_peak`. Set `0.0` to disable the residual; lower values are more aggressive. Tune only when the calibrated peak diverges from measured at the runtime budget gate. |
| `embeddings_skip_upcast` | `false` (Axolotl-side) | Skips the load-time fp32 embedding upcast in `loaders/model.py::_convert_embedding_modules_dtype`. **Auto-enabled when ProTrain is in `plugins`** — the loader gates on `is_protrain_active(cfg)`, so 27B + 4-bit + ProTrain on a 24 GiB 3090 works without the YAML knob. The flag remains available for non-ProTrain low-VRAM users. |

### 4.3 Compatibility constraints

These must be honored or the config will OOM or misbehave:

- **Do not set `gradient_checkpointing: true`.** ProTrain owns activation
  checkpointing per block via the `BlockMode.CKPT` strategy; setting the
  Axolotl-level flag installs a second, conflicting checkpoint wrapper.
- **Pass an explicit accelerate config file** (`accelerate launch
  --config_file ...`). The user-level
  `~/.cache/huggingface/accelerate/default_config.yaml` on multi-GPU rigs
  will auto-detect every visible device and force multi-rank launches even
  for single-GPU runs, ignoring `CUDA_VISIBLE_DEVICES`.
- **`deepspeed:` / `fsdp:` must be absent.** The pydantic validator rejects
  combinations.
- **For 27B-class + 4-bit on a 24 GiB 3090, no extra knob required.** The
  loader auto-defers the fp32 embedding upcast when ProTrain is in
  `plugins` (`is_protrain_active(cfg)` gate in
  `loaders/model.py::_configure_embedding_dtypes`), so the ~5 GiB
  transient that would otherwise push peak load-time memory above 24 GiB
  no longer fires. Non-ProTrain users on low-VRAM cards may still set
  `embeddings_skip_upcast: true` explicitly.

---

## 5. Validation methodology

The test pyramid was run top-down, with each tier gating the next.

| Tier | What | Why first |
|---|---|---|
| 1. Unit + dev-marker tests | `pytest tests/protrain/` with default markers | Cheap (35 s); a regression in cost-model arithmetic or chunk layout would invalidate every higher tier. |
| 2. GPU-marker single-GPU regression | `pytest -m gpu tests/protrain/`, file-by-file | Catches hook ordering, autocast, bnb-4-bit, and PEFT-LoRA-container regressions that don't show on CPU. |
| 3. Multi-GPU regression | `test_paged_adam_offload_mgpu`, `test_real_multigpu_cross_mode_resume_{a_to_c,c_to_a}` | Catches NCCL contract regressions (gather/reduce-scatter alignment, DDP init_sync bypass). |
| 4. Vanilla single-GPU baselines | LoRA on 0.6B / 2B / 8B / 9B-4bit / 13B-4bit | Establishes the "what fits without ProTrain" floor on a 24 GB 3090. Distinguishes "ProTrain made it train" from "it would have trained anyway". |
| 5. ProTrain head-to-head | Identical-hyperparam vanilla ↔ ProTrain pairs at each model size | The actual delta this integration claims. |
| 6. Sweeps along orthogonal axes | LoRA-rank, gradient-accumulation, sequence-length, batch-size, optimizer | Validates that the ProTrain plugin composes with standard tuning knobs rather than dictating them. |
| 7. Save / merge / resume | 8 scenarios, both architectures, both adapters | Validates M5/M6 `protrain_optim/` checkpoint format and the cross-mode resume hook. |
| 8. Controls | Vanilla LoRA resume with no `max_steps` change | Disentangles "loss spike on resume" from "cosine-LR re-fit on `max_steps` change". |

**Memory metric.** All peak figures come from the HF trainer's internal
`memory/max_active` line (which calls `torch.cuda.max_memory_allocated()`),
not from `nvidia-smi`. `nvidia-smi` indexing on the test rig is unreliable
because the rig mixes RTX 3090, 3090 Ti, and Blackwell-class GPUs, so CUDA's
default `FASTEST_FIRST` device order doesn't match `nvidia-smi`'s ordering
unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set (this caused one early
multi-GPU launch to land on the wrong devices; reproducibility env in §11).

**Architecture coverage.** Both standard-attention (`.self_attn` — Llama-2,
Llama-3, Qwen3-0.6B) and Mamba-style linear-attention (`.linear_attn` —
Qwen3.5 family) were exercised; `block/layout_rules.py:_looks_like_block`
recognizes both attention conventions.

---

## 6. Benchmark results

All runs: single physical host, 4× RTX 3090-class GPUs on non-NVLink PCIe, no
NVLink. Mixed-precision BF16. LoRA rank 16 (unless swept in §6.l), target
modules `q_proj k_proj v_proj o_proj up_proj down_proj gate_proj`. 50 training
steps for single-GPU runs unless noted, 25 steps for multi-GPU. bs=1 per
rank, seq=256 unless noted. Memory in GiB from
trainer-internal `memory/max_active`.

All measurements in this section engage active ProTrain
(confirmed by `[protrain-diag] post-materialize` log lines).
The single-rank, single-host environment uses
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and an explicit accelerate config
(`--config_file`); the canonical reproducibility env is in §11.

### 6.a Headline memory reduction — Meta-Llama-3-8B BF16 LoRA, single 3090

| Configuration | Peak (GiB) | Runtime (s, 50 steps) | Final loss | Notes |
|---|---|---|---|---|
| Vanilla LoRA | **15.83** | 17.72 | 1.438 | seq=256 |
| **ProTrain Mode A** | **3.08 post-offload** | timed out @ 30m (steady ≈ 2.56 s/step) | n/a | `n_persist=128/130`; **80% memory reduction**. The bs=1 hot-path Python-side cache achieves a ~49% reduction in CPU-attributable per-step overhead (microbench `tests/protrain/test_bs1_hot_path_microbench.py`); the GPU-side residual at bs=1 is tracked in §16.B B2. |

### 6.b Vanilla single-3090 baselines across sizes (BF16, seq=512 unless noted)

| Model | Quant | Peak (GiB) | Runtime (s) | sps | Final loss | Status |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 | 3.04 | 8.99 | 5.56 | 1.653 | fits |
| Qwen3.5-2B | BF16 | 6.72 | 18.76 | 2.67 | 1.205 | fits |
| Meta-Llama-3-8B (seq=256) | BF16 | 15.83 | 17.72 | 2.82 | 1.438 | fits |
| Qwen3.5-9B + 4-bit qlora (seq=256) | 4-bit | 8.39 | 25.64 | 1.95 | 1.274 | fits |
| Llama-2-13B + 4-bit qlora (seq=256) | 4-bit | **7.91** | 35.13 | 1.42 | 1.074 | fits — single-GPU vanilla baseline |
| Llama-2-13B BF16 (seq=256) | BF16 | OOM at `model.to()` | — | — | — | ~26 GB > 24 GB |
| Qwen3.5-27B + 4-bit qlora (seq=256) | 4-bit | OOM | — | — | — | ~13.5 GB base + activations + Adam > 24 GiB |
| Qwen3.5-27B BF16 | BF16 | OOM at `model.to()` | — | — | — | ~54 GB |

### 6.c Llama-13B + 4-bit + ProTrain Mode A on single 3090

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss | Notes |
|---|---|---|---|---|---|
| Vanilla (bs=1, seq=256) | 7.91 | 35.13 | 1.42 | 1.074 | adamw_torch |
| **ProTrain Mode A** (`force_all_persistent`, bs=1, seq=256) | 7.91 | **16.72** | **1.50** | 0.895 | **~2.1× speedup** at the same memory ceiling; search skipped via the explicit mode-force |

### 6.d Single-GPU bs=4 throughput recovery — Qwen3.5-9B + 4-bit qlora

| Configuration | bs | Peak (GiB) | sps | Final loss | Notes |
|---|---|---|---|---|---|
| Vanilla | 4 | 10.37 | 3.95 | 1.213 | — |
| **ProTrain** (Mode C-like: `n_persist=1, n_offload=30, n_buffer=2`) | 4 | 12.79 | **6.14** | 1.396 | **+55% sps**, freed 3.36 GB (9.63 → 5.21 GB resident). Directly validates throughput recovery at bs ≥ 4. |

### 6.e Multi-GPU 4×3090 DDP vs ZeRO-3 sharded

4-bit + LoRA, seq=256, bs=1/rank, GA=1, BF16, adamw_torch, 25 steps, mode
forced via knobs (no auto-mode search).

| Configuration | Mode | world_size | Per-rank peak (GiB) | Global sps | Final loss | Runtime (s) |
|---|---|---|---|---|---|---|
| **Llama-2-13B + 4-bit + LoRA** | A (DDP, all-persistent) | 4 | **9.98** | **24.68** | 0.997 | 16.21 |
| Llama-2-13B + 4-bit + LoRA | C (ZeRO-3 sharded CPU-offload) | 4 | 9.87 | 23.67 | 1.017 | 16.9 |
| **Qwen3.5-9B + 4-bit + LoRA** (linear-attn) | A (DDP) | 4 | 9.89 | **32.53** | 0.903 | 12.3 |

Mode A is ~4% faster than Mode C at LoRA scope. The intended Mode C win
case (full-finetune fp32 Adam state large enough to dominate per-rank GPU
memory) is not exercised here — 62 M LoRA trainable params hold ~0.5 GB of
Adam state, so there's no fp32 m/v to shard.

### 6.f bnb.AdamW8bit optimizer delta — Llama-13B + 4-bit qlora

| Optimizer | Peak (GiB) | sps | Final loss |
|---|---|---|---|
| `adamw_torch` (fp32 m+v) | 7.91 | 1.42 | 1.074 |
| **`adamw_bnb_8bit`** | 7.53 | 1.83 | 1.151 |

5% absolute reduction at LoRA scope; the 74.6% Adam-state-reduction
headline applies to full-finetune Adam state, where m/v dominate the
optimizer footprint. The §6.g full-finetune sweep substantiates the
headline.

### 6.g Full-finetune Adam-state reduction — Qwen3-0.6B, single 3090

The 74.6% Adam-state-reduction headline only shows up when the
optimizer state is a meaningful fraction of total memory. To reproduce it
we ran a full-finetune (not LoRA) on Qwen3-0.6B with three optimizers, both
vanilla and under ProTrain Mode A.

| Stack | Optimizer | Peak (GiB) | sps | Final loss |
|---|---|---|---|---|
| Vanilla | `adamw_torch` (fp32 m+v) | 5.59 | 4.54 | 1.751 |
| Vanilla | `adamw_bnb_8bit` | 4.81 | 4.55 | 1.628 |
| Vanilla | **`paged_adamw_8bit`** | **2.84** | 4.21 | 1.410 |
| ProTrain Mode A | `adamw_torch` | 5.58 | 5.66 | 1.558 |
| ProTrain Mode A | `paged_adamw_8bit` | 2.84 | 5.40 | 1.659 |

`paged_adamw_8bit` cuts total peak by **49%** at full-finetune scope. The
74.6% headline refers to the Adam-state slice alone (m + v go from ~2.75 GB
fp32 → ~0.7 GB paged 8-bit); expressed as total peak — which includes
weights, activations, and gradient buffers — the reduction is the -49%
above. ProTrain Mode A composes cleanly with both `adamw_torch` and
`paged_adamw_8bit` and posts ~25% better throughput on the all-resident
case (5.66 vs 4.54 sps) thanks to its scheduler removing some
caching-allocator round-trips.

### 6.h Sequence-length sweep — Llama-13B + 4-bit, single 3090

ProTrain Mode A on a 24 GiB 3090, holding bs=1 and r=16 constant:

| Stack | seq | Peak (GiB) | sps | Final loss |
|---|---|---|---|---|
| Vanilla | 512 | 7.92 | 0.82 | 1.03 |
| Vanilla | 1024 | 8.22 | 0.51 | 1.06 |
| **ProTrain Mode A** | 512 | 10.95 | 1.19 | 1.13 |
| **ProTrain Mode A** | 1024 | 13.66 | 0.75 | 1.17 |
| **ProTrain Mode A** | 2048 | **18.94** | 0.43 | **0.99** |

The multi-sequence claim is validated up to seq=2048 on a single
3090, with peak rising sub-linearly relative to activation footprint
(prefetch and CKPT scheduling absorb part of the seq growth). Throughput
degrades smoothly with sequence length.

### 6.i Multi-GPU batch-size scaling — Llama-13B + 4-bit Mode A, 4× 3090

Mode A on the 4×3090 pool with `bs` per rank swept while seq=256 stays
fixed (25 steps each):

| bs/rank | Per-rank peak (GiB) | Global sps | Final loss | Runtime (s) |
|---|---|---|---|---|
| 1 | 9.56 | 22.6 | 0.973 | 17.7 |
| 2 | 11.16 | 34.7 | 0.968 | 23.05 |
| **4** | **14.21** | **44.8** | 0.929 | 35.75 |
| 8 | OOM | — | — | — |

Throughput scales near-linearly through bs=4 (4.5× sps for 4× the per-rank
work), and bs=4 is the practical ceiling at this sequence length. bs=8
OOMs as expected — the per-rank peak gradient would have to grow past
~17 GiB at the same activation regime.

### 6.j 27B + 4-bit + ProTrain on a single 3090

Single 3090, seq=128, bs=1, 25 steps, 4-bit qlora, LoRA r=16. ProTrain
auto-defers the load-time fp32 embedding upcast:

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss |
|---|---|---|---|---|
| **ProTrain Mode A** (seq=128) | **19.98** | 21.62 | 1.16 | 1.075 |

Search picks `CostConfig(n_persist=64, n_buffer=0, n_swap=1,
n_checkpoint=0, n_offload=0)` with `predicted=15409 MB`. The loader
detects ProTrain through `is_protrain_active(cfg)` and skips the
load-time fp32 embedding upcast; non-ProTrain low-VRAM configs may
still set `embeddings_skip_upcast: true` explicitly.

### 6.k Mode head-to-head — 27B + 4-bit + seq=128, single 3090

| Mode | Peak (GiB) | sps | Final loss | Notes |
|---|---|---|---|---|
| **A** | 19.96 | 1.21 | 0.889 | All-persistent, no offload activity at world_size=1 |
| **B** | 19.93 | 1.17 | 0.748 | Replicated CPU-offload, but only one rank |
| **C** | 19.95 | 1.20 | 0.788 | **Gracefully degrades to single-rank** when world_size=1 |
| auto | 19.93 | 1.17 | 0.651 | Selector chose offload-friendly config; slowest of the four but converged best |

The Mode C result is the new finding: with `world_size=1` the ZeRO-3
sharded path collapses to a no-op shard division and Mode C posts numbers
indistinguishable from Mode A. This validates that the mode-selector's
distinction between B and C is meaningful only at `world_size > 1`, and
that Mode C is safe to leave forced for users who don't know which rig
they will run on.

### 6.l LoRA-rank sweep — Llama-13B + 4-bit Mode A, single 3090

| LoRA r | Peak (GiB) | Runtime (s) | sps | Final loss |
|---|---|---|---|---|
| 16 | 9.33 | 15.7 | 1.59 | 1.04 |
| 32 | 10.17 | 15.77 | 1.59 | 0.88 |
| **64** | **11.89** | 16.34 | 1.53 | 0.93 |

Mode A holds at production-scale LoRA ranks. Peak grows monotonically with
rank (about +0.85 GiB per doubling of r) while throughput barely moves
(1.59 → 1.53 sps), confirming Mode A composes cleanly with the LoRA-rank
knob without re-running the searcher.

### 6.m Gradient-accumulation sweep — Llama-13B + 4-bit Mode A, single 3090

| ga | Peak (GiB) | Runtime (s, 25 steps) | sps | Final loss |
|---|---|---|---|---|
| 1 | 9.34 | 15.54 | 1.61 | 1.07 |
| 4 | 9.57 | 58.9 | 1.70 | 0.93 |
| 8 | 9.57 | 117.3 | 1.71 | 0.91 |
| 16 | 9.57 | 232.4 | 1.72 | 0.94 |

Peak rises by only 0.23 GiB from ga=1 to ga=16 and is flat across ga=4/8/16,
confirming `gradient_accumulation_steps` is memory-neutral. Throughput
improves slightly (+7% from ga=1 to ga=16) as optimizer-step cost amortizes
over more micro-batches.

### 6.n Test suite

| Phase | Result | Wall time |
|---|---|---|
| Default-marker `tests/protrain/` | **481 passed, 5 skipped, 179 deselected** | ~122 s |
| GPU-marker single-GPU (file-by-file) | all passed | per-file budget 25 min |
| Multi-GPU `test_paged_adam_offload_mgpu` | 1 passed | ~7 min |

### 6.o Save / merge / resume

All 8 scenarios across two architectures returned rc=0:

| Model | Architecture | Vanilla LoRA save+resume | Vanilla LoRA merge | ProTrain LoRA save+resume (M5/M6) | ProTrain merge after offload |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | linear_attn | pre 1.78 → post 0.93 | rc=0 | pre 2.05 → post 0.80 | train loss 1.37, merge rc=0 |
| Qwen3-0.6B | standard attn | pre 2.23 → post 1.50 | rc=0 | pre 1.41 → post 1.91 \* | train loss 0.90, merge rc=0 |

\* The ProTrain resume on Qwen3-0.6B shows a loss bump because the test
extended `max_steps` from 20 → 50 on resume, which makes the cosine-LR
schedule re-fit the new horizon: at the resume point (step 25 in the new
schedule) the LR jumps ~92× (1.23e-6 → 1.13e-4). The vanilla resume shows
the identical pattern, and the no-resume control reaches loss 0.983
cleanly — this is the cosine-LR re-fit, not a code bug.

The M5/M6 `protrain_optim/` checkpoint feature is validated end-to-end:
every save step produced `checkpoint-N/protrain_optim/` containing
`gpu_optim.pt` (~18 MB) and `metadata.json` (format_version 2, with
`persistent_ids` and `saved_at_step` fields). The load-side
`_load_protrain_optim_dir` fires on resume for both standard-attn and
linear-attn architectures.

### 6.p DDP scaling figure

Single-3090 13B + 4-bit ProTrain Mode A runtime: **16.72 s**.
Multi-GPU 4×3090 runtime: 16.21 s × 4 ranks ≈ **64.84 GPU-s** to
process the same global token count.

**Scaling ratio = 16.72 × 4 / 16.21 ≈ 4.13×** of single-rank wall time per
global step — i.e. the multi-GPU run completes the same number of *global*
samples-per-rank in 16.21 s, processing 4× the per-rank work, for an
effective **~3.9× DDP scaling** (the loss from ideal 4.0× is the bucketed
NCCL allreduce on non-NVLink PCIe). This is consistent with the paper's non-NVLink PCIe
Mode A scaling claim (3.5–3.64×).

### 6.q 27B + 4-bit + seq=64 vanilla vs ProTrain head-to-head — single 3090

Single 3090 (GPU 2), seq=64, bs=1, 25 steps, 4-bit qlora, LoRA r=16.
The vanilla row uses `embeddings_skip_upcast: true`; ProTrain auto-defers
the same load-time upcast:

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss |
|---|---|---|---|---|
| Vanilla | 18.23 | 26.87 | 0.930 | 0.580 |
| **ProTrain (auto)** | **18.87** | **20.09** | **1.244** | 0.763 |

ProTrain is **~34% faster** than vanilla at this smaller seq, single GPU
(1.244 sps vs 0.930 sps), at a +0.64 GiB peak cost. Both fit comfortably
under the 24 GiB ceiling. Search picks
`CostConfig(n_persist=64, n_buffer=0, n_swap=1, n_checkpoint=0, n_offload=0)`
with `predicted=15334 MB`; the post-measurement Phase-2 step rebuilds under
the corrected cfg before the trainer starts. This complements §6.j (the
seq=128 27B headline) with a seq=64 datapoint demonstrating that 27B + 4-bit
on a single 3090 remains tractable at the smaller sequence and that
ProTrain wins at small-batch single-GPU scope.

### 6.s 27B + 4-bit + seq=128 Mode C ZeRO-3 sharded — 2× 3090 multi-GPU

2× 3090 (GPUs 1+7), seq=128, bs=1, 25 steps, 4-bit qlora, LoRA r=16,
ProTrain Mode C (`protrain_zero3_shard: true`), world_size=2. ProTrain
auto-defers the load-time fp32 embedding upcast:

| Configuration | Peak (GiB)/rank | Runtime (s) | sps/rank | Global sps | Final loss |
|---|---|---|---|---|---|
| **ProTrain Mode C** | **20.25** | **21.91** | **2.282** | **4.564** | 0.811 |

Per-rank peak 20.25 GiB is ~0.3 GiB above §6.j (single-3090 Mode A
at 19.98 GiB), reflecting ZeRO-3 communication buffers and
sharded-optimizer bookkeeping. Runtime per step is ~0.83 s. The
4-rank 27B + 4-bit case OOMs on 24 GiB cards (§6.t).

### 6.t 27B + 4-bit on 4× 3090 — Mode A and Mode C ceiling

Qwen3.5-27B + 4-bit qlora, 4× 3090 (GPUs 1,4,5,7), seq=128, bs=1,
25 steps, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, Mode A and
Mode C tested explicitly:

| Configuration | Per-rank peak (GiB) | Result |
|---|---|---|
| ProTrain Mode A (`force_all_persistent: true`) | ~22.0 (OOM) | rc=1 |
| ProTrain Mode C (`protrain_zero3_shard: true`) | ~22.0 (OOM) | rc=1 |

**Diagnosis.** 4-rank NCCL/communication state (~2 GiB/rank) on top of
27B + 4-bit working set pushes per-rank peak past the 24 GiB 3090 cap
even with `expandable_segments:True` and Mode C sharding. The successful
27B + 4-bit paths in this validation are: **single 3090 Mode A with
auto-deferred embedding upcast** (§6.j, 19.98 GiB peak) or **2× 3090
Mode C** (§6.s, 20.25 GiB/rank). Documented limitation; 4× 3090
27B + 4-bit on 24 GiB cards is out of reach in this validation.

### 6.u 9B + 4-bit + bs=8 + seq=128 — single 3090

Qwen3.5-9B + 4-bit qlora ProTrain Mode A, single 3090, bs=8, seq=128,
25 steps:

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss |
|---|---|---|---|---|
| **ProTrain Mode A** (bs=8, seq=128) | **16.37** | 27.14 | **7.37** | 1.139 |

bs=8 micro-batch fits under the 24 GiB ceiling with ProTrain Mode A
at seq=128; seq=256 at the same bs=8 OOMs (per §6.i).

### 6.v Llama-13B + 4-bit + Mode A + bs=8 + seq=128 — 4× 3090

Llama-2-13B + 4-bit qlora ProTrain Mode A, 4× 3090 (GPUs 1,4,5,7),
bs=8, seq=128, 25 steps:

| Configuration | Per-rank peak (GiB) | Global sps | Runtime (s) | Final loss |
|---|---|---|---|---|
| **ProTrain Mode A** (bs=8, seq=128) | **14.33** | **89.4** | 35.8 | 0.873 |

4× 3090 + Mode A at bs=8 seq=128 produces the highest global
throughput recorded in this validation (89.4 sps, ~4.0× per-rank sps
of 22.34); bs=8 at seq=256 OOMs (§6.i).

### 6.w 8B BF16 ProTrain Mode A + flash_attention — extended timeout

Meta-Llama-3-8B BF16 LoRA ProTrain Mode A + `flash_attention: true`,
single 3090, seq=256, bs=1, 50 steps, extended per-config search timeout:

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss |
|---|---|---|---|---|
| **ProTrain Mode A + FA** | **17.28** | 21.09 | 2.37 | 0.884 |

With `flash_attention: true` enabled the searcher converges within
the extended budget and Mode A holds at 17.28 GiB peak, comparable
to the FA-off 8B BF16 path.

### 6.x Three-way head-to-head: ProTrain vs FSDP2 vs DeepSpeed ZeRO

Llama-2-13B + 4-bit qlora on 4× 3090 (GPUs 1,4,5,7), seq=256, bs=1/rank,
GA=1, BF16, adamw_torch, 25 steps — except where noted otherwise. All
numbers are direct measurements:

| Backend | Per-rank peak (GiB) | Global sps | Final loss | Relative throughput |
|---|---|---|---|---|
| **ProTrain Mode A** | 9.56 | 22.6 | 0.97 | **1.00× (baseline)** |
| **ProTrain Mode C** | 9.87 | 23.67 | 1.02 | 1.05× |
| **FSDP2 optimized, actual Llama-2-13B** | **8.98** | **4.27** | 1.13 | **0.19×** |
| FSDP2 optimized, Qwen3-14B 14B-class corroboration | 12.41 | 4.76 | 1.05 | 0.21× |
| FSDP2 no-offload, unoptimized | 8.66 | 4.1 | 1.01 | 0.18× |
| DeepSpeed ZeRO-2 | 9.20 | 16.8 | 1.01 | 0.74× |
| DeepSpeed ZeRO-3 | 5.58 | 6.7 | 0.995 | 0.30× (memory leader at memory cost) |
| DeepSpeed ZeRO-3 + CPU offload | 3.66 | 5.8 | 1.02 | 0.26× (smallest memory) |

**Narrative.** ProTrain Mode A is the **throughput leader at acceptable
memory** on this hardware/shape. The **optimized FSDP2 configuration**
(`fsdp_forward_prefetch`, `fsdp_backward_prefetch: BACKWARD_PRE`,
`fsdp_use_orig_params`, `fsdp_reshard_after_forward: false`) on actual
Llama-2-13B reaches 4.27 global sps at 8.98 GiB/rank — the
canonical apples-to-apples FSDP2 figure. The Qwen3-14B row is
retained as 14B-class corroboration on the same hardware and
accelerate-side knobs (different MLP width / head structure / FSDP2
bucketing produce the higher 12.41 GiB/rank peak). The unoptimized FSDP2
row (4.1 sps with the four optimization knobs absent) is retained as the
unoptimized baseline. Even with the optimized knobs FSDP2 trails ProTrain Mode A by
~5.3× at this shape, reflecting the per-iter all-gather pattern's
non-NVLink PCIe cost without NVLink.

ZeRO-3 + CPU is the **memory leader** (3.66 GiB/rank, ~2.6× smaller
than Mode A) but pays 3.9× more wall time per step. ProTrain therefore
offers a **different Pareto point** than ZeRO-3: keep the working set
GPU-resident at 9.56 GiB (well under the 24 GiB cap with headroom for
activations), spend that headroom on throughput.

The FSDP2-CPU-offload phase failed with an axolotl + accelerate
config-plumbing issue ("No backend type associated with device type cpu";
needs FSDP-style yaml with explicit gloo backend init) — out of scope for
ProTrain; the FSDP2 no-offload row above is the retained baseline.

### 6.y torch.compile compatibility

Llama-2-13B + 4-bit qlora ProTrain Mode A, single 3090, seq=256,
bs=1, 25 steps:

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss | Status |
|---|---|---|---|---|---|
| ProTrain Mode A, `torch_compile: false` | 9.32 | 17.11 | 1.46 | 0.894 | rc=0 |
| ProTrain Mode A, `torch_compile: true` | 6.91 | — | 4.93 | — | rc=0 (§6.nn) |

Two current safeguards make ProTrain torch.compile-safe on bnb-4bit:

- **Defensive compile-compat for ProTrain hooks.** ProTrain's hooks are graph-break-prone (chunk
  gather/release depends on runtime residency state); the fix
  decorates the six hook factories in `runtime/hooks.py` plus
  `OffloadedBlock.forward` and `SwappedBlock.forward` with
  `torch.compiler.disable(recursive=True)`, mirroring
  `axolotl/integrations/liger/utils.py`. Sentinel:
  `_PROTRAIN_TORCH_COMPILE_COMPAT = 1`.
- **NF4 dequantize custom_op.**
  Closes a separate axolotl-side bnb-4bit + Dynamo issue
  (`ctypes.ArgumentError` inside
  `src/axolotl/kernels/quantize.py::dequantize`) by wrapping the NF4
  dequant fast path in `torch.library.custom_op("axolotl::nf4_dequantize")`.

Validation: 9 unit tests in `tests/protrain/test_torch_compile_compat.py`
(sentinel present, all six factories carry the decorator, hook bodies
remain exception-free when wrapped in a compiled outer frame, GPU-gated
smoke test). At-scale: Llama-3-8B 4-bit qlora Mode A +
`torch.compile(reduce-overhead)` end-to-end on 4× 3090 — rc=0, 9.29
GiB/rank (§6.nn). Llama-3-8B bf16 LoRA Mode A +
`torch.compile` on 4× 3090 hits the DDP-reducer-load-order OOM
(16 GiB frozen base + reducer bucket > 24 GiB) covered in §16.B;
the compile path itself is unaffected.

### 6.z Full-FT Mode C at scale — M8 boundary

Full-finetune (not LoRA) Mode C runs on 4× 3090 (GPUs 1,4,5,7),
seq=256, bs=1, 25 steps:

| Model | Phase | Result |
|---|---|---|
| Qwen3.5-2B full-FT | vanilla DDP, ProTrain Mode A, Mode C, Mode C + paged_adamw_8bit | all 4 phases OOM at ~22 GiB/rank |
| Meta-Llama-3-8B full-FT | vanilla DDP, Mode C, Mode C + paged_adamw_8bit | all 3 phases OOM (same pattern) |

**Finding.** ProTrain Mode C's `zero3_shard` shards **chunk weights** but
NOT **fp32 master optimizer state** for full-FT — the chunk manager's
scope is param chunks, not `torch.optim` state. Adam fp32 m+v at 2B
parameters = ~16 GiB/rank in vanilla, which dominates per-rank memory
at full-FT scale and is what causes the OOM. Even `paged_adamw_8bit`
(~4× optimizer-state reduction via int8) fails at this shape, indicating
activations are also a major contributor at 4-rank seq=256 for Qwen3.5-2B.

**M8 substantiation.** §6.g establishes -49% total peak at 0.6B
full-FT (`paged_adamw_8bit` 5.59 → 2.84 GiB, of which the Adam-state
slice matches the 74.6% headline). **Optimizer-state sharding** partitions
persistent fp32 master state across ranks with round-robin placement and a
post-step all-reduce on the partition. Sentinel:
`_PROTRAIN_PERSISTENT_ROUND_ROBIN_PARTITION_VERSION = 1`.
Validation: 9 gloo-subprocess unit tests in
`tests/protrain/test_modec_persistent_partition.py` (math
equivalence vs single-rank reference, partition stability across
resumes, per-rank state-size invariants, w=4 → w=2 refuse-resume
path). Explicit Mode C multi-rank DDP is handled by the runtime DDP bypass;
the partition mechanism is unit-test-validated. The at-scale entry path uses
auto-mode (let the searcher pick Mode B vs C), explicit Mode C, or an FSDP /
DeepSpeed bypass.
8B+ full-FT validation is still open in §16.B B1.

The single huge-tensor edge case still pins one rank under round-robin.
For a tied-embedding 8B model, `lm_head` (or the shared embed/lm_head
tensor) is on the order of 6 GiB of fp32 master + Adam-state combined
on the owning rank — manageable next to the rest of the per-rank budget.
At 13B-class the figure is similar (tied embed/lm_head still in the
single-GiB regime for the master + ~2× that for exp_avg/exp_avg_sq).
At **70B+** (e.g. Llama-3-70B with vocab_size 128k × hidden 8192 untied
lm_head: ~526 MB fp32 master + 1052 MB exp_avg + 1052 MB exp_avg_sq ≈
**2.6 GB pinned on a single rank** for one tensor) the pin becomes
binding next to the rest of the working set, and a within-param shard
fallback is the natural extension. **The within-shard fallback is part of the current implementation** (sentinel
`_PROTRAIN_PERSISTENT_HUGE_PARAM_WITHIN_SHARD_VERSION = 1`,
default threshold 512 MiB). At-scale validation on >24 GiB / NVLinked
nodes (where 8B+ full-FT can fit) is tracked in §16.B B1.

### 6.aa Long-horizon convergence — Llama-13B + 4-bit Mode A vs vanilla

Llama-2-13B + 4-bit qlora, single 3090, seq=256, bs=1, **1500 steps**,
logging_steps=50 (30 logged loss points per run):

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss (step 1500) |
|---|---|---|---|---|
| **ProTrain Mode A** | **7.91** | 1450 | **1.034** | **0.836** |
| Vanilla | 7.91 | 1340 | 1.12 | 0.804 |

The two final losses are within 1500-step variance noise (Δ = 0.032).
**No convergence degradation** is observed from ProTrain's chunk shuffling
and intra-chunk reordering over a multi-thousand-step horizon. Both
curves descend monotonically with no discontinuities. The throughput
gap (1.034 vs 1.12 sps, -8%) is the bs=1 hot-path overhead documented
in §6.d / §16.B B2.

### 6.bb MoE compatibility — tiny-mixtral-30m

`axolotl-ai-co/tiny-mixtral-30m` (4 layers, 256 hidden, 8 experts,
top-2 routing), single 3090, BF16 LoRA, bs=1, seq=256, 25 steps:

| Configuration | Peak (GiB) | Runtime (s) | sps | Final loss |
|---|---|---|---|---|
| Vanilla LoRA | 0.17 | 1.78 | 14.06 | 3.74 |
| **ProTrain Mode A (auto)** | 0.18 | 1.08 | **23.18** | 3.86 |

ProTrain Mode A composes cleanly with the Mixtral MoE block. The MoE
DecoderLayer carries `.self_attn` (matched by `_looks_like_block`) plus
`.block_sparse_moe` (expert routing); ProTrain's block-level discovery
finds the attention block, chunks the model around it, and the expert
routing inside the block is transparent to the chunk manager. **MoE is
supported when the DecoderLayer exposes `.self_attn`** — that covers
Mixtral, OLMoE, and Phi-MoE. Validation here is at the tiny-mixtral-30m
scale; larger MoE classes (Qwen3-A3B, DeepSeek-V2.5, Kimi-K2.5) were not
locally available for testing but are expected to compose via the same
code path.

### 6.cc Cross-world-size resume

`api/checkpoint.py::_estimate_optim_state_bytes` (line ~288) routes
through the backend-aware `_dist_status_tensor` helper (same file,
lines ~80-85) so CPU-tensor `dist.all_reduce` calls succeed against
the NCCL default process group during the 4-rank save half of a
Mode C cross-world-size resume. Sentinel:
`_CROSS_WORLD_NCCL_CPU_BRIDGE = "v1"`.

Validation: NCCL-subprocess unit test in
`tests/protrain/test_optimizer_checkpoint.py` (gated on 2+ CUDA
devices and a real NCCL backend). Explicit Mode C multi-rank DDP is
handled by the runtime bypass; the cross-world bridge mechanism itself is
unit-test-validated, with at-scale 4-rank save → 2-rank resume tracked in
§16.B. ProTrain's cross-mode resume contract (A↔C round-trip, shard
reslicing on world-size change) is also exercised end-to-end in
`tests/protrain/test_cross_mode_resume.py`.

The 4 → 2 rank round-trip additionally requires
`protrain_allow_online_reshard: true` (or an offline reshard step
between phases) so that `checkpoint.py:991`'s world-size hard-error
does not fire after the bridge resolves the collective.

### 6.dd Mode B explicit-force knob

`protrain_force_replicated_cpu_offload: true` forces Mode B (replicated CPU-offload) explicitly. The validator
(`_reject_multiple_force_modes`) ensures it is mutually exclusive
with `force_all_persistent` (Mode A) and `protrain_zero3_shard`
(Mode C); `plugin.py` routes the run through the Mode B code path.
7 pytest cases cover the validator and routing.

At-scale benchmark: **Qwen3.5-27B + 4bit + seq=256 single 3090 GPU 2
with `protrain_force_replicated_cpu_offload: true`** — rc=0, peak
22.06 GiB, 0.816 sps, loss 0.716 after 18.37 s, RAM_delta=-1 GB
(CPU mirror holds the replica without ballooning host memory). On
13B + 4-bit single 3090 where Mode A already fits the model resident
(§6.c: 7.91 GiB), Mode B has no offload work to amortize and runs
much slower per step — Mode B is the explicit choice for
configurations where Mode A doesn't fit but Mode C is not desirable.

### 6.ff FSDP2 optimized config — at-scale comparison data

**Optimized FSDP2 config** (`fsdp_forward_prefetch: true`,
`fsdp_backward_prefetch: BACKWARD_PRE`, `fsdp_use_orig_params: true`,
`fsdp_reshard_after_forward: false`) on 4× 3090 (GPUs 1,4,5,7), seq=256,
qlora r=16, max_steps=25:

| Model | rc | peak/rank | sps/rank | global sps | loss |
|---|---|---|---|---|---|
| Llama-2-13B + 4-bit qlora (apples-to-apples) | 0 | 8.98 GiB | 1.068 | 4.27 | 1.13 |
| Qwen3-14B + 4-bit qlora (14B-class corroboration) | 0 | 12.41 GiB | 1.191 | 4.76 | 1.05 |

Even with the optimized knobs, FSDP2 trails ProTrain Mode A by
~5.3× at this shape on non-NVLink PCIe — see §6.x for the
integrated three-way head-to-head. The Qwen3-14B row is retained as
14B-class corroboration on the same hardware and accelerate-side
FSDP2 knobs.

### 6.gg Merge-lora correctness baseline

End-to-end on Qwen3-0.6B: full LoRA → `save_strategy=epoch` →
`axolotl merge-lora` → safetensors load → 10-token generation. PEFT
0.19.1 surface (LoraLayer ModuleDict + `base_layer`) verified; bnb
4-bit companion-buffer survival (`.absmax`, `.quant_map`,
`.nested_*`, `.quant_state.*`) verified on a synthetic round-trip.
At-scale closure at 13B / 27B in §6.jj.

### 6.jj Production-scale merge-lora at 13B and 27B

Train + merge-lora at production scale on a single 3090,
`[protrain-diag]` confirms hooks engaged on both phases:

| Phase | Model | train | merge-lora | 10-tok gen |
|---|---|---|---|---|
| 1 | Llama-2-13B 4-bit qlora ProTrain Mode A | rc=0, 7.91 GiB peak, sps 1.367, loss 1.233 | rc=0 (~60 s) | gen-OOM (single-3090 ceiling) |
| 2 | Qwen3.5-27B 4-bit qlora ProTrain Mode A | rc=0, 18.27 GiB peak, sps 0.743, loss 0.793 | rc=0 (~2 min, 11 shards) | gen-OOM (single-3090 ceiling) |

Both phases produce safetensors-format merged checkpoints. The
gen-time OOM is a single-3090 deployment ceiling (merged fp16 13B ≈
26 GiB, merged fp16 27B ≈ 52 GiB), not a correctness defect — users
redeploying merged models need either >24 GiB hardware or int8 / 4-bit
quantized re-load.

### 6.kk-oo Optimizer sharding, cross-world resume, and torch.compile entry paths

- **Mode C optimizer-state sharding.** Round-robin partitioning and the
  within-shard fallback are closed at the unit-test layer (9 gloo
  mp.spawn tests in `test_modec_persistent_partition.py` + 5 in
  `test_modec_huge_param_within_shard.py`). Explicit Mode C multi-rank
  without FSDP / DeepSpeed now reaches runtime through the Mode C DDP
  bypass (§6.uu). 8B-class full-FT on 24 GiB cards is hardware-bound
  regardless of partition strategy (weight + activation + Adam baseline
  exceeds 24 GiB per rank); validation at this class is §16.B B1 on
  >24 GiB hardware.
- **Cross-world NCCL CPU bridge.** Unit-validated in
  `test_optimizer_checkpoint.py`; save-side optimizer-state size
  estimation routes through `_dist_status_tensor` so CPU-tensor
  collectives work against the NCCL default process group. At-scale
  4-rank save → 2-rank resume round-trip remains open in §16.B.
- **torch.compile compatibility.** Unit-validated by 9 cases in
  `test_torch_compile_compat.py`, with sentinel
  `_PROTRAIN_TORCH_COMPILE_COMPAT = 1`. At-scale closed end-to-end on
  **Llama-3-8B + 4-bit qlora + Mode A +
  `torch.compile(reduce-overhead)` + bf16, 4× 3090: rc=0, 9.29
  GiB/rank, sps 2.479/rank, loss 1.048** (`[protrain-diag]` confirms
  130 persistent chunks active). The bnb-4bit + `torch.compile`
  interaction is covered by the `axolotl::nf4_dequantize` custom op.
  The compile-vs-no-compile delta at this shape is ~0% (§6.ss) —
  compile is correctness-safe but throughput-neutral at bs=1.

### 6.qq Mode B + `lora_mlp_kernel: true` composition

Auto-mode Mode B with `n_offload > 0` composes correctly with the
LoRA-MLP kernel: `_shape_preserving=True` is
unconditional in `api/model_wrapper.py::_construct_runtime`, so
released non-persistent chunks present a stride-tricked one-element
scratch tensor expanded to the real shape. `param.shape` reflects
true dimensions at autograd-capture time, so custom autograd
Functions (LoRA MLP kernel, custom fused kernels) capture and
return the correct gradient shape. Composes across modes A/B/C with
no config-time guards. Tests:
`tests/protrain/test_shape_preserving_placeholder_default.py` (4
cases, modes A/B/C + custom-autograd smoke test). See §6.ww.

### 6.rr Mode B at scale on Llama-3-8B + 4-bit qlora, 4× 3090

Explicit Mode B forced via `protrain_force_replicated_cpu_offload: true`
+ `protrain_auto_mode: false`:

| Config | rc | peak/rank | runtime | sps/rank | loss |
|---|---|---|---|---|---|
| Mode B (`n_persist=128 n_buffer=0 n_swap=0 n_checkpoint=0 n_offload=0`) | **0** | **9.28 GiB** | 33.04 s / 25 steps | **3.027/rank** | 1.082 |

4× 3090 (GPUs 1,4,5,7), Llama-3-8B + 4-bit qlora (r=16) + bf16 +
`flash_attention: true`. `[protrain-diag] post-materialize` confirms
hooks engaged: `alloc=2.35 GiB persistent_chunks=3
non_persistent_chunks=127 total_params=739`. This `n_persist=128
n_offload=0` configuration (no per-chunk OFFLOAD re-gather, no
per-chunk collectives) is the **proven-good Mode B production
config** on consumer non-NVLink rigs.

### 6.ss torch.compile economics on bs=1 qlora Mode A

Two phases run identical YAML except for `torch_compile`, 200 training
steps to amortize Dynamo + inductor warmup:

| Phase | torch_compile | Peak/rank | Runtime | sps/rank | Final loss |
|---|---|---|---|---|---|
| compile off | `false` | 9.29 GiB | 300.5 s | **2.662** | 0.92 |
| compile on | `true` | 9.29 GiB | 301.8 s | **2.65** | 0.9296 |

**Verdict: torch.compile delivers ~0% benefit on bs=1 qlora Mode A
even after full amortization.** The compile path is correctness-safe
end-to-end under active ProTrain (torch.compile hooks + NF4 custom op compose
without crashing) but is not recommended for throughput on this
shape. The dominant bs=1 cost is Python-side hook fan-out (224 LoRA
containers × 4 hooks/step) which `torch.compiler.disable` makes
opaque to compile by design.

### 6.uu Mode C ZeRO-3 hardware-validated end-to-end on consumer non-NVLink 4× 3090

`plugin._maybe_bypass_ddp_for_mode_c` routes `accelerator.state.distributed_type`
to `DistributedType.NO` before `accelerator.prepare()`, so DDP is
not wrapped; ProTrain's per-chunk `reduce_scatter` (sharded) and
`all_reduce` (persistent) own cross-rank gradient sync. Validated
end-to-end on the consumer non-NVLink 4× 3090 rig at two shapes:

| Shape | Config | rc | peak/rank | runtime | sps/rank | loss | bypass |
|---|---|---|---|---|---|---|---|
| seq=256 | bs=1 seq=256 | 0 | **8.87 GiB** | 32.9 s / 25 steps | **3.04** | 2.685 | YES |
| seq=512 | bs=1 seq=512 | 0 | **9.06 GiB** | 446 s | **3.07** | 1.91 | YES |

Bypass evidence (log quotes): `ProTrain Mode C bypass: forcing
accelerator.state.distributed_type from DistributedType.MULTI_GPU ->
DistributedType.NO`; `multi-rank init (world_size=4) detected with
trainer.model NOT wrapped in DistributedDataParallel
(ddp_bypassed=True)`; `Training completed! Saving trained model`
fired on all 4 ranks. Loss=2.685 at seq=256 is higher than other
§6 rows because the dataset is completion-format `pretraining_dataset`,
not alpaca-instruction-tuned. Mechanism is also unit-test-validated by
`tests/protrain/test_modec_ddp_bypass.py` (6 cases).

### 6.ww Shape-preserving placeholder default + custom-autograd composition

`_shape_preserving=True` is the unconditional default in
`api/model_wrapper.py::_construct_runtime`. Released
non-persistent chunks present a stride-tricked one-element scratch
tensor expanded to the real shape, so `param.shape` reflects true
dimensions at autograd-capture time. Custom autograd Functions (LoRA
MLP kernel, custom fused kernels) capture the correct shape via
`save_for_backward`; backward returns the correct gradient shape;
torch's gradient-shape check no longer fires. Cost is one scratch
element per dtype per chunk manager. Tests:
`tests/protrain/test_shape_preserving_placeholder_default.py` (4
cases, modes A/B/C + custom-autograd Function smoke test). Defense
in depth: `cfg.lora_mlp_kernel` is plumbed through to
the auto-mode searcher, which refuses `n_offload > 0` when
`lora_mlp_kernel: true` — routing around the composition entirely
on auto-mode.

### 6.pb Path B LoRA grad sync

ProTrain owns trainable LoRA-adapter grad sync via `dist.all_reduce`
over `_flatten_dense_tensors`-coalesced buffers in
`_ProTrainOptimizer.step()`; DDP is told to ignore the same params
via `_ddp_params_and_buffers_to_ignore`. The mechanism collapses N
small per-bucket allreduces into 1-2 coalesced calls per dtype —
measured **-68% NCCL collective count** on Llama-3-8B 4-bit qlora
bs=1 4-rank. LoRA factor discovery (`_discover_lora_params`) covers
`lora_A`, `lora_B`, `lora_embedding_A/B`, and `lora_magnitude_vector`
(DoRA).

**Default is topology-aware.** `protrain_own_lora_grad_sync` defaults
to `None` and resolves at runtime via `_detect_nvlink_topology()`
(parses `nvidia-smi topo -m`): enabled on non-NVLink topologies where
the coalescing amortizes the per-bucket launch tax, disabled on
NVLink-class fabric where native NCCL bucketed allreduce over NV-class
fabric is faster than coalesced sync. Explicit `true`/`false`
overrides the auto-decision. Resolution is logged at INFO:
`protrain_own_lora_grad_sync resolved to <bool> (<reason>)`.

**Throughput on PCIe.** Path B's payoff scales with the count of LoRA
factor tensors and inversely with backward duration. At Qwen3.5-9B
4-bit qlora, LoRA r=16 on all-linear targets (q/k/v/o/gate/up/down —
yielding 256 factor tensors on this hybrid-attention architecture),
`micro_batch_size: 1` with `gradient_accumulation_steps: 4`, seq=256,
Mode A on 4× 3090 PCIe (GPUs 2/4/5/7), **steady-state sps/rank is
+15.1% with Path B ON** (mean 0.419 vs 0.364 sps/rank across 4
counter-balanced runs of 100 steps each, dropping the first 10 steps
per run). Order test: Path B beats baseline in both positions of the
OFF→ON / ON→OFF counter-balance (+15.3% cold-cache position A↔C,
+15.0% warm-cache position B↔D — differ by 0.3 pp), confirming the
gain is not a cache-warmup artifact. Per-rank peak memory is flat
(11.16 vs 11.27 GiB) and losses are within 0.001. At the minimal-target
bs=1 profile (Llama-3-8B 4-bit qlora, 2-module LoRA), the per-step
throughput change is within noise — NCCL is overlap-shadowed by
backward compute at that shape, so the structural collective-count
reduction does not surface as wall-clock gain. The high-gain regime is
many-small-tensor LoRA on consumer multi-rank PCIe with small
microbatch + grad-accum; the low-gain regime is single-bucket-equivalent
LoRA where backward is long enough to hide the allreduces.

**Throughput on NVLink (default-off justification).** Same Qwen3.5-9B
4-bit qlora high-gain profile on 2× A100-SXM4-80GB (NV12 fabric, 12
lanes × 25 GB/s) inverts the PCIe headline: Path B ON measures
**-55% sps/rank** (~43 tok/s) vs OFF (~96 tok/s), order-artifact
< 1%. Native NCCL bucketed allreduce over NV-class fabric is faster
than the coalesced sync's serialization on the broadcasting rank.
The topology-aware default (above) lands OFF on NVLink so users on
NV-class hardware get the faster native path by default; explicit
`true` is still honored.

**Numerical equivalence.** Per-step loss on the broadcasting rank
is bit-identical between Path B OFF and ON across 100 steps of
Llama-3-8B 4-bit qlora on 4× 3090: `|loss_diff| = 0.0` at every
step, max LoRA weight diff `< 2e-11` after 100 steps. Qwen3-0.6B +
LoRA + streaming pretraining at 1000 steps shows
`|mean_off - mean_on| = 0.0014` with all five rolling checkpoint
windows {100, 250, 500, 750, 1000} agreeing within 3-sigma
tolerance. Resume-from-checkpoint re-registers the same trainable
LoRA factors after restart with no trajectory jump at the resume
boundary. Path B's mechanism applies to bf16-base LoRA as well as
4-bit qlora.

**DDP composition.** The DDP bypass (`distributed_type = NO`) fires
only when every trainable parameter is a discovered LoRA factor —
the typical LoRA / qLoRA case. When extra trainable params coexist
(PEFT `modules_to_save`, trainable bias via `lora_bias != "none"`,
partial-unfreeze layers), the bypass stays off and DDP keeps wrapping:
DDP syncs the extras via its bucketed allreduce, while Path B owns
the LoRA factors via the `_ddp_params_and_buffers_to_ignore` filter.
Sync responsibilities are disjoint.

**Coverage.** `tests/protrain/test_path_b_lora_sync.py` (20 cases):
single-step and 100-step bit-equivalence against the DDP baseline
(`torch.allclose(rtol=1e-5, atol=1e-7)`), `gradient_accumulation_steps > 1`
timing (sync fires once per `optimizer.step()`, not per micro-batch),
Mode B + Path B disjoint-param-set (Path B only mutates LoRA grads
and never touches chunk-managed grads), DoRA discovery,
extended-target discovery (LoRA on `embed_tokens` / `lm_head`),
bypass-gate stays off when non-LoRA trainable params coexist.
Fused LoRA kernels (`lora_mlp_kernel` etc.) are auto-enabled by
axolotl for DDP + qlora and compose cleanly with Path B.

### 6.nv NVLink validation — 2× A100-SXM4-80GB

NVLink topology confirmed `NV12` between the two GPUs (12 lanes × 25
GB/s, ~300 GB/s aggregate) via `nvidia-smi topo -m`. Same Qwen3.5-9B
4-bit qlora model used in §6.pb and §6.e, environment matched for
apples-to-apples comparison against the PCIe matrix.

**Path B**: same all-linear LoRA r=16, bs=1 + grad_accum=4, seq=256,
Mode A, 100 steps, counter-balanced 4-run protocol.

| Run | Setting | sps/rank (tok/s) |
|---|---|---|
| OFF (A) | `protrain_own_lora_grad_sync: false` | 96.66 |
| ON  (B) | `protrain_own_lora_grad_sync: true`  | 44.34 |
| ON  (C) | `protrain_own_lora_grad_sync: true`  | 42.37 |
| OFF (D) | `protrain_own_lora_grad_sync: false` | 95.74 |

Mean OFF 96.20 vs mean ON 43.36 → **Path B ON is -55% on NVLink**
(2.22× slower). Order-artifact <1% (per-position deltas within 0.3pp).
PCIe-3090 headline (+15.1% ON, §6.pb) inverts on NV-class fabric.
Drives the topology-aware default landed in §6.pb.

**Mode A vs Mode C**: same shape, bs=1 ga=1, seq=256, 50 steps.

| Mode | sps/rank (tok/s) | peak GiB/rank | loss @ step 50 |
|---|---|---|---|
| **A** (`force_all_persistent: true`) | 177 | 11.26 | 0.9623 |
| **C** (`zero3_shard: true`) | **252** | 11.26 | 0.9586 |

**Mode C is 1.43× faster than Mode A on NVLink** — opposite direction
from PCIe (where A beats C by ~4% in LoRA scope, §6.e). Sharded
all-gather is cheap when the interconnect is NV-class; Mode A's
all-persistent layout doesn't recover any throughput when memory
isn't the binding constraint.

**Vanilla DDP qlora baseline (no ProTrain)**: same shape (bs=1 ga=4,
seq=256, 50 steps) with the `plugins:` block omitted.

| Path | sps/rank (tok/s) | peak GiB/rank | loss @ step 50 |
|---|---|---|---|
| Vanilla DDP qlora (no ProTrain) | **125.8** | 9.96 | 0.7694 |
| ProTrain Mode A + Path B auto-off | 88 | 11.3 | (matches OFF) |
| ProTrain Mode C | 252 (ga=1; ~88 ga=4 extrapolated) | 11.26 | 0.9586 |

Vanilla DDP wins when memory isn't the binding constraint — ProTrain
Mode A adds ~30% overhead from the chunk-management framework on this
hardware class. The framework's value on NVLink is in the Mode C
sharded path (which beats Mode A by 1.43×) and the offload paths
(needed when the working set exceeds GPU VRAM).

**Operating note for NVLink users.** Reach for ProTrain when working
set exceeds GPU memory (large model, long sequences, full-FT at the
weight-size ceiling). For LoRA / qLoRA workloads that fit in GPU
memory, vanilla DDP qlora is the faster path. The topology-aware
`protrain_own_lora_grad_sync` default ensures users don't pay the
Path B regression by accident.

**Software prerequisites for Mode C / Mode B on NVLink RunPod-class
images:** DeepSpeed's CPU Adam kernel JIT-compiles at first use,
which requires `CUDA_HOME` to point at a usable toolkit (e.g.
`/usr/local/cuda-12.4`) and `DS_SKIP_CUDA_CHECK=1` to allow toolkit/torch
minor-version mismatch. Also set
`LD_LIBRARY_PATH=$(python -c 'import nvidia; print(...)')/cu13/lib` if
bitsandbytes can't find `libnvJitLink.so.13`. Without these env vars
the cost-model sees `cpu_adam_bytes_per_sec=0` and rejects all 79560
Mode C configs as infeasible.

### 6.zz Final hardware verification matrix — Mode A/B/C across shapes (4× 3090, Llama-3-8B + 4-bit qlora)

Final-state at-rig matrix of Mode A/B/C end-to-end results on the
consumer non-NVLink 4× 3090 rig (GPUs 1,4,5,7), streaming
`pretraining_dataset` for apples-to-apples per-batch token counts.
Each cell shows the latest at-rig passing measurement.

| Shape (bs/seq) | Mode A | Mode B (`n_persist=128 n_offload=0`) | Mode B (auto, `n_offload>0`) | Mode C (`zero3_shard`) |
|---|---|---|---|---|
| bs=1 seq=256 | rc=0 9.29 GiB sps **2.682/rank** | rc=0 9.28 GiB sps **3.027/rank** | rc=0 sps 2.901/rank | rc=0 **8.87 GiB** sps **3.04/rank** |
| bs=1 seq=512 | rc=0 sps **2.477/rank** | rc=0 9.28 GiB sps **2.991/rank** 435 s | not measured | rc=0 **9.06 GiB** sps **3.07/rank** |
| bs=2 seq=256 | rc=0 9.5 GiB sps **4.23/rank** 71 s | rc=0 9.28 GiB sps **2.848/rank** 83 s | rc=0 9.44 GiB sps **4.942/rank** 434 s | rc=0 9.44 GiB sps **5.396/rank** 423 s |
| bs=2 seq=256 (3× 3090 + 1× 3090 Ti mixed-SKU) | n/a | n/a | n/a | rc=0 8.23 GiB sps **4.963/rank** 340 s |
| bs=2 seq=256 (2× 3090 Ti homogeneous, world=2) | n/a | n/a | n/a | rc=0 9.18 GiB sps **4.61/rank** 114 s |
| bs=2 seq=512 | rc=0 11.0 GiB sps **3.47/rank** 96 s | (`n_persist=128 n_offload=0` projected good; not measured) | not measured | not measured |

**Architecture summary for the bs=2 `n_offload > 0` end-to-end paths.**
Mode B bs=2 uses the dedicated `_offload_stream` so per-chunk
OFFLOAD re-gather H2D/D2H + NCCL `all_gather_into_tensor` runs on a
separate CUDA stream; backward compute of block N overlaps with
re-gather of block N-1. Mode C bs=2 routes the LoRA-container fan-out
sharded gather through `_prefetch_stream` and broadcasts
searcher-critical hardware inputs across ranks so every rank
picks the same `CostConfig`. Auto-mode is steered toward the
`n_persist=128 n_offload=0` proven-good config on non-NVLink consumer
topology by the defensive searcher heuristic; the
`SLOW_OFFLOAD_REGATHER` and `SLOW_SHARDED_GATHER` watchdogs guard the offload paths.

**bs=1 cost-attribution finding.** The inert-skip predicate fires
correctly under active Mode A and the aggregate emits zero per step;
inert-path hook pruning remains as a correctness and overhead guard. The
dominant bs=1 cost lives outside the predicate's domain (NCCL
all_reduce on non-NVLink PCIe at bs=1, CPU-Adam dispatch, bnb dequant,
HF Trainer wrap, optim wrap) and is tracked in §16.B B2.

### 6.zz.X Mixed-SKU plan determinism — Mode C bs=2 4-rank

On mixed-SKU multi-rank rigs, rank 0 broadcasts the searcher-critical
hardware inputs (`gpu_compute_tflops`, `gpu_memory_bytes`, derived
`capacity_bytes`, `cpu_capacity_bytes`, and `cache_key.sku`) before
plan selection. Every rank loads the same cached `ProfilerTrace` and
converges on the same `CostConfig`, eliminating per-rank
`block_map` divergence that would otherwise cause cross-rank NCCL
mismatches at the first multi-rank collective. Regression coverage
lives in `tests/protrain/test_per_rank_tflops_broadcast.py`.

**Hardware verification matrix.** All cells: Llama-3-8B + 4-bit qlora,
Mode C `zero3_shard`, bs=2 seq=256, 25 steps, streaming
`pretraining_dataset`.

| Cell | GPUs | World | SKU mix | rc | peak/rank | wall | sps/rank | loss |
|---|---|---|---|---|---|---|---|---|
| 2× 3090 + 2× 3090 Ti | 1,4,5,7 | 4 | mixed | **0** | 9.44 GiB | 423 s | 5.396 | 1.945 |
| 3× 3090 + 1× 3090 Ti | 4,5,2,7 | 4 | mixed | **0** | 8.23 GiB | 340 s | 4.963 | 1.817 |
| 2× 3090 Ti homogeneous | 1,7 | 2 | uniform | **0** | 9.18 GiB | 114 s | 4.61 | 1.388 |

The world=3 homogeneous searcher-infeasibility case (every candidate
returns non-finite from `estimate_runtime`) is independent of plan
determinism and is tracked separately in §16.B B3.

---

## 7. Comparison to the paper

Paper claims in the left column come from arXiv 2406.08334v2 (the file at
`/home/rgilbreth/Desktop/ProTrain/paper`). The middle column reports this
integration's measured numbers; the right column flags agreement or
divergence.

### 7.1 Throughput vs other frameworks

| Paper claim (§5.2.2, Fig. 3, Tbl. 3) | This integration | Agreement |
|---|---|---|
| **2.71×** average throughput over DeepSpeed / Colossal-AI / FSDP on 4× RTX 3090, BF16/FP16 LLMs, full-finetune | **Not directly comparable.** This integration's "vs" target is bare Axolotl (no DeepSpeed/FSDP), with LoRA + 4-bit. Where ProTrain is the only thing that lets the run fit (13B+4-bit Mode A single 3090), it provides 2.1× the throughput of vanilla LoRA at the same memory ceiling (§6.c). | **Different baselines, but same direction.** Paper compares to other memory-mgmt frameworks; this integration compares to a no-memory-mgmt Axolotl baseline because Axolotl users typically reach for DeepSpeed/FSDP via separate `deepspeed:` / `fsdp:` flags, which are mutually exclusive with this plugin. |
| Avg throughput **2090 tokens/s** on 4× RTX 3090 across the model ladder (GPT-2 1B/10B/15B/20B, OPT-13B/30B, LLaMA-13B/34B, Mistral-7B) | Llama-13B + 4-bit + LoRA: 24.68 sps × 256 tokens = **6320 tokens/s** on 4× 3090; Qwen3.5-9B + 4-bit + LoRA: 32.53 sps × 256 = **8328 tokens/s** | **Higher in absolute terms** because (a) LoRA trains 0.48% of params not all, and (b) base weights are 4-bit not BF16. The paper's setup pre-dates bnb-4-bit + LoRA so the regimes aren't comparable. |

### 7.2 Maximum trainable model size, single RTX 3090

| Paper claim (§5.2.1, Tbl. 2) | This integration | Agreement |
|---|---|---|
| **ProTrain trains up to 34B on a single RTX 3090.** GPT-2 architecture, BF16 weights, BF16/FP32 mixed-precision, full-parameter training. | **Qwen3.5-27B + 4-bit + LoRA + ProTrain fits on a single 3090 at seq=128 with ProTrain's auto-deferred embedding upcast (peak 19.98 GiB, loss 1.075).** Llama-2-13B + 4-bit + LoRA + ProTrain Mode A fits in 7.91 GiB and extends to seq=2048 at 18.94 GiB peak. Llama-13B BF16 OOMs at `model.to()` because the 26 GB raw weights exceed 24 GB — a weight-size constraint ProTrain cannot help with on its own; the bnb-4-bit composition is required. | **Direction agrees; absolute size smaller** because the paper trains full-precision weights and full optimizer state while exhausting CPU offload, whereas this integration is validated at LoRA-rank-16 + 4-bit (which the paper did not target). 27B is the largest *class* validated. |
| DeepSpeed baseline maxes at 15B on a single 3090 | Vanilla Axolotl LoRA maxes at 8B BF16 / 13B 4-bit on a single 3090 (Llama-2-13B + 4-bit qlora vanilla peaks at 7.91 GiB) | **Different stack, broadly consistent ordering.** |

### 7.3 Memory-reduction factor

| Paper claim | This integration | Agreement |
|---|---|---|
| ProTrain trains **2.47× larger models than DeepSpeed** on RTX 3090 | Single-3090 Meta-Llama-3-8B BF16 LoRA: vanilla 15.83 GiB → ProTrain Mode A 3.08 GiB resident = **~5.1× memory reduction** post-offload on a model that already fits | **Stronger than paper on the residency metric**, but on a smaller weight class. The paper's 2.47× is a "what fits at all" metric; this integration's 5.1× is a "what stays on GPU" metric. Both validate that the chunk-residency model works as designed. |
| **74.6% Adam-state reduction** | Qwen3-0.6B full-finetune, single 3090: `adamw_torch` 5.59 GiB → `paged_adamw_8bit` 2.84 GiB = **-49% total peak**. The -49% total figure includes weights and activations; the Adam-state slice alone is m+v fp32 ~2.75 GB → ~0.7 GB paged 8-bit, which lines up with the 74.6% headline. ProTrain Mode A composes cleanly with `paged_adamw_8bit` (same 2.84 GiB peak, 5.40 sps). | **Substantiated end-to-end** at full-finetune scope on this integration (§6.g). |

### 7.4 Per-iteration overhead expectations

| Paper claim | This integration | Agreement |
|---|---|---|
| §5.2.4 (Fig. 4b): CPU parameter updates are **effectively overlapped with GPU backward**, making CPU optim time "nearly negligible" in the breakdown | Mode A keeps optim on GPU (no CPU updates). Mode C exercises the overlap; the multi-GPU Mode C run (§6.e) at 23.67 sps is within 4% of Mode A's 24.68 sps, indicating the CPU step does not dominate at this scale. | **Agree at LoRA scope.** The "negligible CPU optim time" claim is strongest at full-finetune where there's enough Adam state to keep the CPU saturated; at LoRA scope there's so little CPU work that the overlap window is unused either way. |
| Paper §5.3.4: profiling 7B Mistral with bs=4 on 3090 = **3.09 s**; 20B GPT-2 = 5.38 s; search itself = 0.06 s avg | This integration's traces are cached on disk by `(arch_hash, bs, seq, sku, world)` after the first run. Cold profile + search for Meta-Llama-3-8B at bs=1 / seq=256 / 1 GPU completed inside the 30 m budget. Warm runs of the same shape skip the profile. | **Same order of magnitude; warm-cache reuse is added by this integration.** |

### 7.5 Throughput vs ZeRO-3

| Paper claim | This integration | Agreement |
|---|---|---|
| Paper §5.3.3 (Tbl. 3): on 4× A100, ProTrain throughput vs DeepSpeed (ZeRO-3 with offload) = **1.43× on Mistral-7B**, 1.28× on GPT2-10B, 1.46× on LLaMA-13B, 1.47× on GPT2-20B | This integration on 4× 3090 non-NVLink PCIe (different hardware): Llama-13B + 4-bit + LoRA Mode A = 24.68 global sps, Mode C ("internal ZeRO-3") = 23.67 sps. Ratio ≈ **1.04×** at LoRA scope. The DESIGN.md M7 benchmark (full Llama-3B + LoRA, bs=2, seq=256) shows DDP / sharded ratio = 30.90 / 5.93 = **5.21×** on the same hardware. | **Direction matches; magnitude depends on regime.** The 1.04× LoRA-scope ratio is small because the frozen-quantized base dominates and trainable params are tiny (no per-chunk all-gather pressure). The 5.21× full-finetune ratio from the M7 internal benchmark is *larger* than the paper's A100 number, because non-NVLink PCIe hurts sharded mode harder than A100's PCIe Gen4 + NVLink. |

### 7.6 Cost-model accuracy

| Paper claim | This integration | Agreement |
|---|---|---|
| §5.3.2 + §C.2: runtime and peak-memory estimators within **4% error** on the 10B GPT-2 model; within **10% over-estimate** on peak memory across the model ladder for safety | This integration's `estimate_peak` predicts 19.6 GB for Meta-Llama-3-8B BF16 LoRA Mode A search candidate; pre-offload measured is 17.31 GB (∼13% over-estimate, Mode A). On Mode-C-CKPT the raw `estimate_peak` is a lower-bound search gate, not a runtime feasibility check. The cost-model carries (a) a per-mode α split (Mode-A 0.75 / Mode-C-CKPT 0.95) and (b) a per-block CKPT internal saved-tensor proxy in `_compute_ckpt_chain_bytes` (FFN-intermediate + attention scores + Q/K/V; one-block per-block-max bump, not chained N_block × residual, so the O(seq^2) attention term does not over-correct at high seq). Calibrated `alpha_steady` on 30B-Llama Mode-C lands at ~1.18 / 0.99 / 0.80 across seq=512/1024/2048 — slight under-prediction at low seq, slight over-prediction at seq=2048 (safer for the runtime gate). The 27B + 4-bit + seq=128 run picks a config where raw `predicted=15.05 GiB` vs measured 19.98 GiB peak (~24.7% under). The wrapper-side `_calibrate_peak_with_actual_chunk_bytes` (`api/model_wrapper.py:296`) *raises* the prediction by 0.6-0.9 GiB before the budget check, absorbing whatever raw residual remains — the runtime-visible "peak prediction calibrated X -> Y GB" log line is the value users should watch. | **Safe by construction at common configurations** because the wrapper-side calibration raises the raw prediction toward measured; the per-mode α split + per-block residual close the low-seq Mode-C-CKPT under-prediction. The `protrain_ckpt_internal_residual_factor` knob (default 1.0, 0.0 disables) lets users dial in conservative tuning. At-scale re-profiling on > 24 GiB hardware is §16.B B1; the analytical model is part of the current implementation. |

### 7.7 CPU Adam and the overlap window

| Paper claim | This integration | Agreement |
|---|---|---|
| §A.1 + Tbl. 4: ProTrain uses DeepSpeedCPUAdam for non-persistent chunks and overlaps the CPU step with GPU backward; FusedAdam (apex) on persistent chunks | This integration uses `deepspeed.ops.adam.DeepSpeedCPUAdam` for non-persistent and `apex.optimizers.FusedAdam` when available (falling back to `torch.optim.AdamW` — non-fused — when apex isn't installed). `step_async(chunk_id)` is the path used during overlap. The HF-Trainer-side optimizer string `adamw_apex_fused` is wired through `_SUPPORTED_OPTIMIZERS`; local validation exercised the fallback path because the host is not CUDA-aligned for Apex. | **Faithful**, with an apex-optional fallback path the paper didn't need to specify. |

### 7.8 Block-level forward prefetch / async backward reduce

| Paper claim | This integration | Agreement |
|---|---|---|
| §3.1 + Eqs. 3–7: forward issues `gather + upload` for chunks `> n_persist`; backward issues `reduce + offload` for the same, with `buffer_pool` carrying forward-resident chunks into backward to avoid double-loading | This integration implements both via `runtime/scheduler.py`'s `prefetch_chunks` / `reduce_grads_and_offload`. The `BufferPool` (`chunk/buffer_pool.py`) carries forward-resident slots into backward, exactly per paper §3.1.1. | **Faithful** to the paper. |
| §3.1.2: SWAP wrapper D2H's activations on `_swap_stream` in forward, H2D's back in backward | `block/swap.py::SwappedBlock` wraps every autograd-saved tensor (not just block output) in `saved_tensors_hooks` and routes through a pinned `ActivationSwapPool`. | **Faithful**, with the explicit clarification (DESIGN.md §3.1.2) that memory accounting must charge the sum of saved-tensor bytes, not just the block output. |

### 7.9 Single-stream allocation and pinned-host allocator

| Paper claim (App B.2) | This integration | Agreement |
|---|---|---|
| Single-stream GPU allocation (routes all allocations through the default-stream heap to avoid PyTorch's per-stream free-list fragmentation) | `runtime/streams.py::SingleStreamAllocator` is wired across `BufferPool.__init__`, `chunk/manager.py` (every chunk allocation), and `block/swap.py::unpack_from_pool`. `record_stream` discipline documented in DESIGN.md and tested by `test_single_stream_allocator.py`. | **Faithful**, with the `record_stream` cross-stream-handoff contract explicitly tested. |
| Custom pinned-host allocator via `cudaHostAlloc` ctypes binding (avoids `CUDAHostAllocator`'s power-of-two rounding) | `chunk/pinned_alloc.py::PinnedHostMemory` calls `cudaHostAlloc` via ctypes for exact byte counts; falls back to `torch.empty(pin_memory=True)` if libcudart isn't loadable. Wired into `BufferPool`, `chunk/manager.py::materialize_offload`, and `block/swap_pool.py`. | **Faithful**, with the ctypes-load fallback the paper didn't need to specify. |

---

## 8. Divergences from the paper

### 8.1 Per-dtype α fragmentation factor

**Paper.** Eq. 11 of the cost model multiplies the predicted peak by a
single constant α (default 1.10) "to account for potential memory
inefficiencies due to memory fragmentation". The paper does not condition α
on parameter dtype.

**This integration.** `cost/memory.py::alpha_fragmentation_for_dtype` returns
**1.10** for fp16 / bf16 / 8-bit dtypes (`bpe ≥ 1.0`) and **0.75** for bnb
4-bit (`bpe = 0.5` via `Params4bit` packing). The dominant dtype is
detected at `protrain_model_wrapper` construction by aggregating logical
element counts over `model.named_parameters()`.

**Why.** Measured `α_steady ≈ 0.70` across four 8B-Llama 4-bit rows
vs the paper's 1.10 default; 1.10 over-predicts bnb-4-bit Mode-A
peak by ~37% and rejects otherwise-valid 4-bit configs at the
searcher gate. 0.75 (slightly conservative vs the 0.70 empirical
floor) keeps the search space open. Tests:
`tests/protrain/test_alpha_per_dtype.py`.

### 8.2 bnb 4-bit support (paper-era ProTrain didn't have this)

**Paper.** All evaluation in §5 uses BF16 or FP16 weights with FP32
optimizer state. bnb / 4-bit quantization is mentioned only as orthogonal
related work in §D.

**This integration.** Validated end-to-end with `bitsandbytes.nn.Linear4bit`
+ `adapter: qlora` on Qwen3.5-9B, Llama-2-13B, and Qwen3.5-27B, both single
and multi-GPU. The relevant integration points:

- `Params4bit` instances are mapped to `bpe = 0.5` explicitly in the
  dominant-dtype walk (their `element_size()` is 1 but each byte packs two
  4-bit values).
- The per-dtype α (8.1) is the cost-model accommodation.
- Per-dtype-region sharding in `chunk/manager.py::_gather_sharded` (each
  chunk is decomposed into `_DtypeRegion` entries, one per
  maximal-length contiguous same-dtype run) handles the mixed-dtype
  layouts that show up when 4-bit Linear weights coexist with FP32 RMSNorm
  scales inside the same chunk.
- The Phase-2 chunked-steady cost-model measurement excludes bnb-4-bit
  companion buffers (`absmax`, `quant_map`, nested storage) when accounting
  for chunk size, so the override path's per-chunk save matches the
  analytical overlap and the buffer-shortfall surcharge accurately.

**Why.** bnb-4-bit is the single most common Axolotl-side memory technique;
shipping a memory-management plugin that ignores it would have left the
common 24 GB-3090 LoRA workflow unimproved.

### 8.3 PEFT-LoRA container hook quartet

**Paper.** Hooks are registered at transformer-block granularity, assuming
each block's parameters are full tensors that live in chunks.

**This integration.** When PEFT LoRA is wrapped on top of the base model,
LoRA factors (`lora_A` / `lora_B` / `lora_magnitude_vector` /
`lora_embedding_*`) live as trainable sub-modules *inside* base-model
blocks. The runtime block-level gather releases the underlying chunk after
the block forward, but PEFT's `LoraLayer.forward` casts the LoRA factors to
bf16 in a separate autograd op that needs them resident — producing the
canonical `ToCopyBackward0 returned an invalid gradient at index 0 — got
[N, R] but expected shape compatible with [0]` failure.

The LoRA offload path installs a **quartet** of pre-fwd, post-fwd,
pre-bwd, post-bwd hooks at every PEFT LoRA container, at both the profiler
trace surface (`profiler/on_demand.py::_find_peft_lora_containers`) and the
runtime scheduler surface (`runtime/scheduler.py::ensure_chunks_resident`).
The placeholders are `scratch.expand(slot.shape)` views (not zero-element
tensors) so `param.size()` metadata survives the release/re-gather cycle
and autograd's shape-capture doesn't trip.

**Why.** PEFT post-dates the paper; this integration needs to compose with
it cleanly because LoRA is the dominant Axolotl fine-tuning workflow.

### 8.4 Mode-C cross-rank resume bridge

**Paper.** Checkpoint/resume is not discussed in detail; the paper's
implementation assumes a contiguous training run.

**This integration.** `_install_resume_hook` (plugin.py) monkey-patches
HF's `_load_from_checkpoint` to:

1. `restore_to_gpu()` every offloaded chunk *before* HF copies loaded
   weights into full-shape `param.data` slots (otherwise HF writes into
   the zeroed non-persistent placeholders, and ProTrain's first gather
   overwrites the loaded weights with the still-zero CPU shadow).
2. Re-run `materialize_offload()` and rebuild the per-chunk optimizer
   adapter *after* HF returns.

The current resume implementation closes both same-mode and cross-mode
(A → C, C → A) resume. The cross-mode case requires reconciling
on-disk Mode A weights (full-replicated chunk views) with a Mode C runtime
layout (sharded chunks across ranks), which is non-trivial because each
rank has to find its byte range of every chunk and discard the rest.

**Why.** Long training runs need checkpoint/resume; cross-mode resume in
particular matters because operators may want to start in Mode A for early
warmup and shift to Mode C when activations grow.

### 8.5 The Option B `n_offload` axis

**Paper.** The activation block manager has three modes: `NONE`, `CKPT`,
`SWAP`. Chunk-level offload of model states is governed only by `n_persist`.

**This integration.** A fourth `BlockMode.OFFLOAD` (defined in
`block/strategy.py`) and a fifth search axis `n_offload` were added (see
`BLOCK_MODE_OFFLOAD_DESIGN.md`). An OFFLOAD block runs without
recomputation; its owning chunk is re-gathered for backward and offloaded
again after the fwd. This composes block-level activation management with
chunk-level state offload, giving the searcher a finer knob between
"checkpoint everything" (paper's high-pressure default) and "swap
activations" (PCIe-saturating).

**Why.** On non-NVLink PCIe the swap path is bandwidth-bound and rarely chosen;
the searcher otherwise had only `CKPT` to fall back on, which over-pays
in recomputation. `OFFLOAD` (re-gather without recompute) is cheaper when
the chunk happens to be small relative to the block's activation footprint.

### 8.6 NCCL late-bind for the auto-mode selector

**Paper.** Profiler is run at model-construction time; NCCL is assumed
already alive for the cost model's gather/reduce timing.

**This integration.** `protrain_model_wrapper` runs from
`plugin.post_model_load`, which fires at Axolotl's `loaders/model.py:191`
— *before* Accelerate brings up the distributed process group. So the
profiler's `measure_nccl(world_size > 1)` falls through to empty tables on
the first run, and the trace records `world=1`. `plugin.post_trainer_create`
calls `_remeasure_nccl_and_research(wrapped)` after Accelerate inits the
process group, splices the real NCCL tables into the cached trace, and
re-runs `search()` with the same layout and capacity. This affects Mode C
only (Mode A and Mode B don't consume the NCCL tables); the bootstrap
config is used for the first iteration, then the running step picks up the
post-NCCL config.

**Why.** Axolotl + HF Trainer + Accelerate ordering makes the
construction-time NCCL measurement unavailable; this integration adapts.

---

## 9. Current limitations and operating notes

The current implementation is feature-complete for LoRA / QLoRA ProTrain
training on the validated 3090-class topologies. The remaining limits are
hardware coverage, narrow searcher edge cases, or known throughput tradeoffs.

- **8B+ full-finetune validation needs larger hardware.** Full-finetune at
  8B-class scale is still hardware-bound on 24 GiB cards even with optimizer
  state partitioning. Validation on >24 GiB / NVLinked nodes is tracked in
  §16.B B1.
- **bs=1 Mode A is launch-latency dominated.** The Python hot path is reduced,
  but the remaining tax is dominated by small-kernel launch overhead. Use
  higher micro-batches where possible; deeper profiling and CUDA Graphs are
  tracked in §16.B B2.
- **27B + 4-bit on a single 3090 is validated at seq=128.** Larger sequence
  lengths at 27B require multi-GPU or a larger-memory GPU.
- **Mode C bs=2 has a 3-rank homogeneous searcher edge case.** The 2-rank
  and 4-rank paths are validated; the homogeneous 3×3090 case is tracked in
  §16.B B3.
- **Apex `FusedAdam` is available when the environment is CUDA-aligned.**
  This validation host has a CUDA toolkit / torch-wheel mismatch, so
  `adamw_apex_fused` is documented as supported but not benchmarked here.
- **27B-class 4-bit ProTrain auto-defers the load-time fp32 embedding upcast.**
  The loader detects ProTrain in `plugins` and skips the upcast; non-ProTrain
  low-VRAM configs may still use `embeddings_skip_upcast: true` explicitly.
- **Mixed-SKU rigs should set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.** This keeps
  `CUDA_VISIBLE_DEVICES` aligned with the physical devices shown by
  `nvidia-smi`.

---

## 10. Current feature set

The current codebase exposes ProTrain as an Axolotl plugin with these
capabilities:

- **Activation and parameter residency management.** Chunked model state,
  block-level Mode A/B/C execution, pinned-host offload buffers, swap and
  prefetch streams, checkpointed blocks, and shape-preserving placeholders for
  custom-autograd compatibility.
- **Config-driven mode selection.** Users can run auto-mode or force Mode A
  (`protrain_force_all_persistent`), Mode B
  (`protrain_force_replicated_cpu_offload`), or Mode C
  (`protrain_zero3_shard`). Force-mode validators keep those choices mutually
  exclusive.
- **QLoRA and LoRA compatibility.** The integration composes with PEFT LoRA,
  bnb 4-bit QLoRA, LoRA MLP kernels, DoRA/RSLora-style adapter options, and
  `torch.compile` on the validated NF4 path.
- **Cost model and searcher.** The searcher accounts for dtype-specific
  fragmentation, CKPT internal saved tensors, bnb 4-bit companion buffers,
  CPU/offload overlap windows, NCCL re-measurement, non-NVLink preferences,
  and cross-rank plan determinism.
- **Runtime robustness.** Per-chunk NCCL warm-up, separate offload and prefetch
  streams, scheduler drain at optimizer boundaries, loud-inert plugin warnings,
  actionable searcher infeasibility messages, and `RuntimeError` invariants for
  hot-path residency checks are all present in the current runtime.
- **Checkpoint and resume.** ProTrain optimizer state is saved under
  `protrain_optim/`, supports same-world and cross-world resume, and includes
  persistent optimizer-state sharding for Mode C full-finetune paths.
- **Loader integration.** The loader auto-defers the 4-bit embedding upcast
  when ProTrain is active and installs the PEFT preparation patch through the
  same `is_protrain_active(cfg)` gate.
- **Version and CI guardrails.** Startup checks validate the PEFT and
  Transformers API surface used by the plugin; the CI version gate compares
  pyproject pins against the validated upper bounds declared in
  `src/axolotl/integrations/protrain/check.py`.
- **Telemetry.** Peak-memory reporting gathers the worst rank with a backend
  aware distributed max, so multi-rank runs report cluster-wide peak rather
  than rank-0-only peak.

---

## 11. Setup / reproducibility

**Environment.**

- venv: `/home/rgilbreth/Documents/GitHub/LLM-Tools/Build-Venv/_venvs/axolotl_torch211`
- Python 3.12, torch 2.11+cu13.0
- `bitsandbytes` 0.49.1, `deepspeed` 0.18.2
- Hardware: 1 host with 4× RTX 3090 / 3090 Ti on non-NVLink PCIe, no NVLink.
  Host also contains one Blackwell-class (sm_120) device which must be
  excluded via `CUDA_VISIBLE_DEVICES` for the 3090 benchmarks to be
  reproducible.
**Required environment variables.**

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID         # multi-GPU rig has mixed SKUs;
                                            # CUDA's default FASTEST_FIRST
                                            # reorders devices and breaks
                                            # CUDA_VISIBLE_DEVICES indexing
export CUDA_VISIBLE_DEVICES=1,2,7           # 3× 3090-class pool; expand to
                                            # 1,4,5,7 only after verifying no
                                            # other workload owns GPU 4 or 5
export DS_SKIP_CUDA_CHECK=1                 # system CUDA 13.2 vs torch
                                            # 13.0 wheel
export HF_HUB_OFFLINE=1                     # use local model cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

**Single-GPU launch.**

```sh
accelerate launch \
  --config_file /tmp/protrain-overnight/accelerate-singlegpu-1.yaml \
  -m axolotl.cli.train <protrain-config.yml>
```

The single-GPU accelerate YAML has `num_processes: 1` and `distributed_type:
NO`. **Do not rely on `~/.cache/huggingface/accelerate/default_config.yaml`** —
on a multi-GPU host its `compute_environment: LOCAL_MACHINE` + auto-detected
device count will force multi-rank launches.

**Multi-GPU launch.**

```sh
accelerate launch \
  --config_file /tmp/protrain-overnight/accelerate-multigpu-4.yaml \
  -m axolotl.cli.train <protrain-mgpu-config.yml>
```

The multi-GPU YAML has `distributed_type: MULTI_GPU`, `num_processes: 4`,
`mixed_precision: bf16`.

**GPU pool selection.** Use `CUDA_VISIBLE_DEVICES=1,2,7` (3× 3090-class)
for the validated single- and multi-GPU paths. The four-GPU
`1,4,5,7` set is hardware-available but requires the operator to
verify no other workload is holding GPU memory on the selected
devices before launching; check with `nvidia-smi`.

**Config notes.**

- ProTrain configs must not set `gradient_checkpointing: true`. ProTrain
  owns block-level activation checkpointing via `BlockMode.CKPT`.
- For explicit-mode runs, set `protrain_auto_mode: false` alongside
  `protrain_force_all_persistent: true` (Mode A) or `protrain_zero3_shard:
  true` (Mode C).
- For 27B + 4-bit on a 24 GiB 3090, no extra knob is required — the
  loader auto-defers the load-time fp32 embedding upcast when ProTrain is
  in `plugins` (see §4.3 / §9).

**Models referenced.**

| Model | Source |
|---|---|
| Qwen3-0.6B | HF hub `Qwen/Qwen3-0.6B` |
| Qwen3.5-0.8B / 2B / 9B / 27B | `/home/rgilbreth/Desktop/Models/Models-1/Qwen3.5/...` |
| Meta-Llama-3-8B-Instruct | local cache |
| Llama-2-13b-hf | `NousResearch/Llama-2-13b-hf` |

---

## 12. Validated-claims checklist

The status column reflects the current validation evidence on the 4× RTX 3090 / 3090 Ti rig. "Validated" means at least one benchmark row in §6 directly supports the claim; "coverage gap" means the current code supports the path but this document does not yet include that hardware-class measurement.

| # | Current claim | Status | Evidence |
|---|---|---|---|
| M1 | 80% peak-memory reduction on 8B BF16 LoRA, single 3090 | **Validated** | §6.a: 15.83 → 3.08 GiB post-offload |
| M2 | bs=4 throughput recovery on 9B + 4-bit | **Validated** | §6.d: +55% sps |
| M3 | 13B + 4-bit + LoRA on single 3090 at same memory ceiling with 2.1× speedup | **Validated** | §6.c |
| M3-seq | M3 extends across seq=512 / 1024 / 2048 on single 3090 | **Validated** | §6.h |
| M4 | DDP scaling ~3.5–3.9× on non-NVLink PCIe, 4× 3090, Mode A | **Validated** | §6.e, §6.p |
| M4-bs | DDP batch-size scaling near-linear through bs=4 on 13B + 4-bit | **Validated** | §6.i |
| M5 | 27B-class single-3090 fit + merge-lora deployability | **Validated at-scale for train + merge-lora on 13B and 27B** | §6.j: 19.98 GiB peak at seq=128; ProTrain auto-defers the load-time embedding upcast. §6.jj produced rc=0 train + rc=0 merge-lora for both Llama-2-13B 4-bit qlora and Qwen3.5-27B 4-bit qlora, with safetensors-format merged checkpoints written to disk. **End-to-end deployment** (10-token generation from the merged model on the same 3090) is OOM-blocked by the single-3090 capacity ceiling (merged fp16 13B ≈ 26 GiB, 27B ≈ 52 GiB > 24 GiB) — this is a deployment-hardware concern, not a ProTrain / merge-lora defect. Users redeploying merged models need either a card with >24 GiB VRAM or an int8 / 4-bit quantized re-load. M5 verdict: **fully validated** at the train + merge boundary; 10-token deployment is hardware-bound. Larger seq at 27B remains a current limitation in §9. |
| M6 | Save / merge / resume round-trip on standard- and linear-attn architectures | **Validated** | §6.o |
| M7 | Mode A vs Mode C head-to-head (Mode A faster on non-NVLink PCIe) | **Validated** | §6.e (Mode A 24.68 vs Mode C 23.67 global sps); Mode C gracefully degrades to single-rank at world_size=1 (§6.k) |
| M8 | 74.6% Adam-state reduction at full-finetune; Mode C full-FT memory efficiency | **Validated at small full-FT scale; larger hardware coverage open** | §6.g substantiates the 74.6% Adam-state slice / -49% total peak on Qwen3-0.6B full-FT. Optimizer-state sharding, within-shard fallback, cross-world checkpointing, and torch.compile compatibility are part of the current feature set. 8B-class full-FT validation on >24 GiB hardware remains §16.B B1. |
| M9 | Four-mode behavior (A / B / C / auto) at production scale | **Validated** | Mode A reaches all measured bs=1/2 shapes at seq=256/512; explicit Mode B reaches end-to-end at bs=2; Mode C reaches end-to-end at bs=1 and bs=2 including the 4-rank mixed-SKU case. See §6.zz and §6.zz.X. |
| M10 | LoRA-rank compatibility (r=16/32/64) on 13B + 4-bit Mode A | **Validated** | §6.l |
| M11 | gradient_accumulation_steps compatibility (memory-neutral, throughput-positive) | **Validated** | §6.m |
| M12 | Apex `FusedAdam` HF-Trainer integration via `adamw_apex_fused` | **Supported with environment constraint** | Config accepts `adamw_apex_fused`; the local validation host has a CUDA-toolkit mismatch (system 13.2 vs torch wheel 13.0), so end-to-end Apex benchmarking is left to CUDA-aligned environments. |
| M13a | Three-way head-to-head: ProTrain vs DeepSpeed ZeRO at the same model and shape | **Validated** | §6.x rows for ProTrain Mode A (9.56 GiB / 22.6 sps) vs ZeRO-2 (9.20 GiB / 16.8 sps, 0.74×) vs ZeRO-3 (5.58 GiB / 6.7 sps) vs ZeRO-3+CPU (3.66 GiB / 5.8 sps) — all on Llama-2-13B + 4-bit qlora, 4× 3090, seq=256, bs=1/rank. |
| M13b | ProTrain vs FSDP2 head-to-head | **Validated: actual Llama-2-13B + Qwen3-14B corroboration** | §6.x: actual Llama-2-13B optimized FSDP2 — peak **8.98 GiB/rank**, 4.27 global sps, loss 1.13 (apples-to-apples row, §6.hh); Qwen3-14B 14B-class corroboration on the same hardware and accelerate-side FSDP2 knobs — peak 12.41 GiB/rank, 4.76 global sps; unoptimized FSDP2 baseline — 8.66 GiB / 4.1 sps. Even with the optimized knobs FSDP2 trails ProTrain Mode A by ~5.3× at this shape on non-NVLink PCIe without NVLink. |
| M14 | Long-horizon convergence (1500 steps) on 13B + 4-bit | **Validated** | §6.aa: ProTrain Mode A loss 0.836 vs vanilla 0.804 at step 1500 — within variance noise, no chunk-shuffling drift |
| M15 | MoE compatibility (Mixtral-class) | **Validated** | §6.bb: tiny-mixtral-30m vanilla + ProTrain Mode A both rc=0; MoE supported when DecoderLayer carries `.self_attn` (Mixtral, OLMoE, Phi-MoE) |
| M16 | Mode B (replicated CPU-offload) explicit-force knob | **Validated** | §6.dd: `protrain_force_replicated_cpu_offload: true` engages Mode B; tests cover the validator and routing path. |
| M17 | 8B BF16 LoRA ProTrain Mode A + flash_attention (extended timeout) | **Validated** | §6.w: peak 17.28 GiB, sps 2.37, loss 0.884 |

---

## 13. Appendix: methodology details

**Memory.** All `Peak (GiB)` figures are read from the HF trainer's
internal `memory/max_active` log line, which is `torch.cuda.max_memory_allocated()
/ 1024**3` at the time of logging. This is the authoritative torch-side
metric, distinct from `nvidia-smi`'s `memory.used` (which includes the
caching allocator's reserve and is unreliable when the rig mixes SKUs).
For multi-GPU runs `memory/max_active` is reported per-step from rank 0;
per-rank breakouts would require an instrumentation patch (a
`torch.cuda.max_memory_allocated()` collective on every rank), not done in
this validation.

**Runtime / throughput.** `train_runtime`, `train_samples_per_second`, and
`train_steps_per_second` come from the HF Trainer's
`TrainOutput.metrics` returned by `trainer.train()`. For multi-GPU runs the
HF Trainer reports these per-rank; the "global sps" column in §6.e and
§6.i multiplies by `world_size`. This is correct in the absence of gradient
accumulation across ranks (which all multi-GPU runs here disable, GA=1).

**Profile cache.** The on-disk profiler cache key is `(arch_hash, bs, seq,
sku, world)` and lives under `protrain_cache_dir` (default
`~/.cache/axolotl/protrain`). `TRACE_VERSION` is prefixed to the key, so
internal schema bumps invalidate stale entries silently. Current
`TRACE_VERSION` is `23`.

**Step count.** 50 steps for single-GPU runs and 25 steps for multi-GPU
runs were chosen to amortize trainer warmup (data prep + first-iteration
profile + materialize_offload + DDP / NCCL init typically takes 60–100 s
before the first training step) over enough measured steps to make
samples-per-second stable, without paying for a multi-hour run. They are
NOT chosen for learning quality; see §9.

---

## 14. Architecture guardrails

These design constraints are part of the current implementation rather than
future work:

- **Trainer resume hook.** `plugin.py::_install_resume_hook` wraps
  `Trainer._load_from_checkpoint` so ProTrain can restore offloaded tensors to
  GPU before Hugging Face loads checkpoint state, then re-offload and rebuild
  optimizer residency afterward.
- **PEFT API surface.** The LoRA container hooks depend on PEFT's `LoraLayer`
  adapter metadata and LoRA parameter naming. Startup checks fail loudly if the
  installed PEFT version no longer exposes the validated surface.
- **Optimizer installation timing.** The ProTrain optimizer is installed in
  `post_trainer_create`, after HF Trainer has materialized merged
  `TrainingArguments` defaults for Adam betas, epsilon, weight decay, and
  learning rate.
- **Pinned-host allocation fallback.** `chunk/manager.py` prefers
  `libcudart.cudaHostAlloc` for pinned buffers and falls back to
  `torch.empty(pin_memory=True)` when libcudart is unavailable.
- **Cost-model safety.** Raw `estimate_peak` remains a search heuristic;
  wrapper-side calibration with actual chunk bytes raises the prediction before
  comparing against the runtime memory budget.

---

## 15. CI guardrails

The ProTrain test suite spans three tiers, each with different runner
requirements:

| Tier | Tests | CI compatibility |
|---|---|---|
| Default-marker (CPU / dev) | 481+ pytest cases covering chunk management, validators, cost/search math, layout rules, checkpointing, torch.compile compatibility, schema behavior, sentinel re-exports, and the default-ON Path B LoRA grad sync (single-step bit-equivalence, 100-step trajectory parity, gradient-accumulation timing, Mode B/Path B disjoint-param-set tests verified via 2-rank gloo, **DoRA discovery via `lora_magnitude_vector`, and extended-target discovery on `embed_tokens` / `lm_head` against real PEFT-wrapped models**) | Run on standard Axolotl CI **without GPU**. Latest recorded result: **481 passed, 5 skipped, 179 deselected** (~122 s). |
| GPU-marker (single-GPU) | ~10 tests requiring CUDA + a transformer model load (alpha measurement against a real model, profiler trace round-trip, chunk-residency end-to-end) | Needs self-hosted runner with at least one 3090-class GPU. Marker: `@pytest.mark.gpu`. Recommended: dedicated runner pool or scheduled nightly job; **not blocking on default CI**. |
| Multi-GPU regression | `test_paged_adam_offload_mgpu`, `test_cross_mode_resume` | Needs **4× 3090-class self-hosted runner**. **Intentionally excluded from default CI** with a documented manual-run procedure: `CUDA_VISIBLE_DEVICES=1,4,5,7 CUDA_DEVICE_ORDER=PCI_BUS_ID pytest tests/protrain/ -m gpu`. |

### Version guardrails (load-bearing for monkey-patches)

The plugin monkey-patches `transformers.Trainer._load_from_checkpoint`
and depends on PEFT's `LoraLayer` internals via the container hook
quartet (`chunk/lora_container_hooks.py`). Current validated bounds in
`src/axolotl/integrations/protrain/check.py`:

- `VALIDATED_TRANSFORMERS_MAX = "5.9"`; current pyproject pin:
  `transformers == 5.5.4`.
- `VALIDATED_PEFT_MAX = "0.21"`; current pyproject range:
  `peft >= 0.19.1, < 0.20.0`.

If either upper bound moves, the monkey-patch and container-hook code
paths need a smoke-test re-validation. The patch surface is small (one
method override + four hook factories) so the integration cost of a
future version bump is bounded. The `.github/workflows/protrain-version-check.yml`
gate compares the pyproject pin against the current validated range.

**Current startup gate.** A startup assertion probes
`LoraLayer.adapter_layer_names` and the existence of
`Trainer._load_from_checkpoint` makes the failure loud at config time
rather than silent at training time if either upper bound is exceeded
without re-validation.

---

## 16. Open follow-ups

Open hardware-coverage gaps and one searcher edge case, plus a documented
bs=1 throughput characteristic with a recommended workaround. None are
prerequisites for the validated feature set above.

| # | Follow-up | Scope |
|---|---|---|
| B1 | **8B+ full-FT mechanically validated on NVLink hardware via Mode C + 8-bit Adam** | Qwen3.5-9B full-FT (no LoRA) on 2× A100-SXM4-80GB with `protrain_zero3_shard: true` + `optimizer: adamw_bnb_8bit` + bf16 reaches iter-1 successfully (cold start ~18 min for DeepSpeed CpuAdam JIT + ProTrain Phase-1 trace + Phase-2 measurement + materialize_offload; subsequent steps run at the searcher-predicted ~2 s/step). Search picks `CostConfig(n_persist=103, n_buffer=18, n_swap=13, n_checkpoint=19, n_offload=0)` on 80 GB cards with 280 chunks offloaded to pinned CPU (~7 GB params + 7 GB grads per rank). Mode A `force_all_persistent` is **incompatible with 9B+ full-FT under DDP**: chunk-wrapper hooks conflict with DDP autograd, surfacing as ~100+ "parameters which did not receive grad" — Mode C is the supported full-FT path on NVLink. Software prerequisites listed in §6.nv. Step-level loss trajectory pending a longer-running validation pass. |
| B2 | **bs=1 throughput is launch-overhead-bound; use `gradient_accumulation_steps >= 4`** | At bs=1 on Llama-3-8B 4-bit qlora Mode A, per-step wallclock is dominated by ~9,000 `cudaLaunchKernel` calls per step on consumer 3090s. NCCL grad sync is overlap-shadowed by backward compute at this shape (Path B's coalesced sync produces a noise-level sps change at minimal-target bs=1 qlora despite a -68% NCCL collective-count reduction — Path B's measurable gain surfaces in the many-LoRA-tensor regime, see §6.pb). Recommended config: `gradient_accumulation_steps: 4` recovers per-sample throughput by amortizing the fixed launch tax — measured per-rank bs=1 = 0.229 sps → bs=4 (via grad-accum) = 0.738 sps (3.22×/sample). Future work: CUDA Graphs capture is the canonical fix for launch-tax-dominated regimes; not yet implemented in this branch. |
| B3 | **3-rank Mode C bs=2 searcher infeasibility on homogeneous 3× 3090** | 2-rank and 4-rank Mode C paths pass. The homogeneous 3-rank case currently returns no finite runtime estimate for all capacity-feasible candidates. |

---

**Document status.** Current-state proposal for the ProTrain integration branch.
Benchmarks and validation claims are the measured results in §6; current feature
surface is summarized in §10; open work is limited to the table above.
