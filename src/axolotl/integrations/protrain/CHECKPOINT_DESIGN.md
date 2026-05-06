# ProTrain Optimizer Checkpoint/Resume — Design Note (v2)

**Status:** historical design note; Phase 1 implementation has landed (see `api/checkpoint.py` and plugin wiring). Phase 2 (DDP + ZeRO-3) is documented in `CHECKPOINT_DESIGN_PHASE2.md` and has also shipped.
**Scope:** Item 3 from the paper-fidelity follow-up plan
**Branch base:** `myfork/protrain-paper-fidelity` @ `99afc31c`

This is **v2** of the design note. v1 underestimated the
HF Trainer / Accelerate hostility to ProTrain's optimizer-state shape.
The reviewer's corrections (recorded in §1.7–§1.9) tightened the
scope: Phase 1 is now **single-rank, non-ZeRO only**, with a custom
ProTrain save/load hook rather than relying on HF's stock path.

---

## 0. Where we stand today

`_ProTrainOptimizer.state_dict` and `.load_state_dict` raise
`NotImplementedError` (`api/optim_wrapper.py:116-126`). At runtime
those methods are silently overridden by the plugin
(`plugin.py:491-520`):

- `state_dict` is patched to return a hollow `{"state": {},
  "param_groups": [...]}` shell.
- `load_state_dict` is patched to a no-op.

The patch comment explicitly names two callers — both are unconditional:
1. **HF Trainer** at checkpoint save (silenced today via
   `save_only_model=True` from `get_training_args`, plugin.py:302-314).
2. **Accelerate at `prepare` time** for device-placement
   (`move_to_device(state_dict, ...)` → `load_state_dict(state_dict)`
   round-trip). NOT silenced — it fires every run.

So today, "checkpointing works" — but the optimizer state is **not
persisted** (resumed runs cold-start every momentum buffer), and any
real implementation has to coexist with the Accelerate `prepare`
round-trip on every run, not just at save time.

---

## 1. Key facts that shape the design

These were verified before writing this note. If any of these turn out
wrong in implementation, revisit the design.

### 1.1 DeepSpeedCPUAdam state IS round-trippable via standard torch APIs

This was the originally flagged risk. Verified empirically:

- `DeepSpeedCPUAdam` inherits `state_dict` / `load_state_dict` directly
  from `torch.optim.Optimizer` — no override (MRO check).
- Inside `step()`, the kernel writes `exp_avg`, `exp_avg_sq`, and
  `step` into `self.state[p]` as ordinary CPU torch tensors
  (cpu_adam.py:144-160):
  ```python
  state['step'] = 0
  state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
  state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
  # ...
  self.ds_opt_adam.adam_update(self.opt_id, state['step'], ...,
                               state['exp_avg'], state['exp_avg_sq'])
  ```
- The C++ extension (`ds_opt_adam`) mutates these tensors **in place**.
  No opaque internal state.

**Implication:** No custom per-chunk state-extraction layer needed.
`inner_optim.state_dict()` is enough.

### 1.2 GPU-side optimizer is a vanilla torch optimizer

`GpuFusedAdamAdapter` wraps `apex.optimizers.FusedAdam` (or falls back
to `torch.optim.AdamW`). State_dict round-trips with no special handling.

### 1.3 The optimizer is a two-tier facade

`_ProTrainOptimizer` owns:
- `self._gpu_optim: GpuFusedAdamAdapter | None` — one optimizer over all
  persistent params
- `self._cpu_optim: CpuFusedAdamAdapter | None` — adapter that owns a
  `dict[ChunkId, DeepSpeedCPUAdam]` (one inner optimizer per
  non-persistent chunk; `chunk/optim.py:88-121`)

Saved state has to be **two-tier** (one GPU optimizer + N CPU
optimizers keyed by ChunkId), not flat.

### 1.4 The chunk partition is deterministic given fixed search output

Layout is built from (model arch, profiler trace, S_chunk, block spans)
and is reproducible. Persistent IDs are derived from `n_persist` plus a
**non-block force-pin pass** (`model_wrapper.py:824-832`) — chunks
holding non-block params (e.g., `lm_head`) are pinned to persistent
even if they fall outside `[0, n_persist)`. The recently landed
`ec65f68f` fix made routing key off the **set** of persistent IDs, so
non-contiguous persistent sets are handled correctly.

**Implication for save metadata:** persisting only `n_persist` is
insufficient — the effective persistent set after the non-block
expansion is what determines which inner optimizer owns which params.
We save the full **`persistent_ids: list[int]`** (the post-expansion
effective set), not just `n_persist`.

### 1.5 Hooks must be reinstalled before load

`materialize_offload` installs per-param `post_accumulate_grad_hook`
closures over chunk IDs and slot pointers (`manager.py:838-851`).
These closures cannot be pickled. The resume flow must call
`materialize_offload()` during wrapper construction (which it already
does) **before** any attempt to load optimizer state.

### 1.6 ZeRO-3 sharded path: CPU optimizer is built over per-rank shard_params

In sharded mode, `cpu_params_per_chunk_for_optim[cid]` contains
`shard_param` objects — one flat `nn.Parameter` per dtype region
holding only that rank's slice (`model_wrapper.py:918-926`,
`manager.py:753-836`). Per-rank optimizer state is naturally
rank-local. Per-rank save / per-rank load is the natural shape.

But **getting per-rank save/load actually wired through HF Trainer is
non-trivial** (see §1.8). That is what pushes ZeRO-3 to Phase 2.

### 1.7 Accelerate `prepare` round-trip fires on every run

This is the structural reason the existing no-op patch exists. From
plugin.py:491-502:
> HF Trainer and Accelerate both call ``state_dict`` unconditionally —
> HF at checkpoint save (silenced via ``save_only_model=True`` in
> ``get_training_args``) and Accelerate at ``prepare`` time for
> device-placement (NOT silenced).

The round-trip is:
1. Accelerate calls `optim.state_dict()` to get the current state.
2. Walks the dict and `.to(device)`s every tensor.
3. Calls `optim.load_state_dict(moved_dict)` to put it back.

For ProTrain this is hostile in two specific ways:
- **CPU adam state must NOT be moved to GPU.** Big-model momentums
  (fp32 × 2 × N) are exactly the memory ProTrain offloaded to keep
  out of HBM. Letting Accelerate stage them on GPU defeats the
  optimizer.
- **Two-tier routing must survive the round-trip.** A naive flat
  state_dict loses the chunk_id partitioning; load needs to know which
  inner optimizer each tensor belongs to.

Two ways to coexist (pick one in §8):
- **Option P (preferred — patch stays):** keep the no-op patch active
  for the lifetime of the optimizer. Save/load goes through a
  ProTrain-specific hook (see §1.8) that bypasses
  `optim.state_dict()`. Accelerate's prepare is unaffected because
  state_dict still returns the empty shell.
- **Option Q (intercept the round-trip):** make the real `state_dict`
  emit CPU-resident tensors (which `.to(device)` would balloon HBM)
  and the real `load_state_dict` re-route by chunk_id and move CPU
  pieces back to CPU. Survives Accelerate's call but pays a real HBM
  spike during prepare.

**Recommendation:** Option P. The no-op patch is correct for the
prepare lifecycle. Don't fight it; route real save/load through a
separate path.

### 1.8 HF Trainer save/load is hostile to ProTrain's state shape

Three specific facts:

1. **HF saves a single `optimizer.pt`** under
   `args.output_dir/checkpoint-N/` from the rank where
   `args.should_save` is True (rank-0 in the standard path, see
   `Trainer._save_checkpoint`). This is a single `torch.save(
   optimizer.state_dict(), 'optimizer.pt')` blob.
2. **HF loads with `map_location=self.args.device`** when world_size > 1
   (and frequently with `device` even single-rank, depending on
   version). This pulls every saved tensor onto GPU at load time —
   directly hostile to CPU-offloaded adam state.
3. **HF's save path doesn't know about per-chunk or per-rank
   structure.** FSDP and DeepSpeed both opt out of the standard path
   and provide their own checkpoint engines (DeepSpeed has its own
   checkpoint writer; FSDP has `FullStateDictConfig` /
   `ShardedStateDictConfig` orchestration). ProTrain has nothing
   equivalent today.

**Implication:** Phase 1 must implement a **custom ProTrain save/load
hook** rather than relying on HF's stock path. Verified against the
installed transformers version, the HF `TrainerCallback` API exposes
`on_save` (post-checkpoint-write) but **does NOT have an
`on_load_checkpoint` hook**. `on_train_begin` fires AFTER
`Trainer._load_optimizer_and_scheduler` runs, so it is also too late
for the load path.

The integration shape is therefore split:
- **Save**: register a `TrainerCallback` whose `on_save` writes our
  per-chunk shard directory beside HF's standard checkpoint dir.
- **Load**: monkey-patch `trainer._load_optimizer_and_scheduler` in
  `post_trainer_create`, wrapping the original to also detect and load
  from `protrain_optim/` if present. This sits exactly where HF expects
  the optimizer-load to happen (before `on_train_begin`) and is
  symmetric with the existing `optim.state_dict` / `load_state_dict`
  monkey-patches in plugin.py:519-520.

### 1.9 Multi-rank single-blob writes are wrong even for "replicated" mode

DDP / replicated-only mode might naively look like "rank-0 saves
everything" — but ProTrain's state is partitioned per-chunk, and the
inner CPU adams hold CPU tensors that must not be staged onto GPU at
load. So even multi-rank replicated needs the custom save/load path.

**Implication:** Phase 1 ships **single-rank only**. Multi-rank
replicated AND ZeRO-3 sharded both need the custom save/load path
fully designed; both go to Phase 2.

---

## 2. Phase 1: single-rank, non-ZeRO

This is the ship target for Phase 1: **single-rank training** (no DDP,
no ZeRO-3). Multi-rank in any form ships in Phase 2.

### 2.1 What we save

Save format goes to `output_dir/checkpoint-N/protrain_optim/` (a
sub-directory beside HF's standard `optimizer.pt` slot, which we leave
empty / disabled).

```text
protrain_optim/
  metadata.json               # see schema below
  gpu_optim.pt                # standard torch.save of inner GPU optimizer state_dict (or absent)
  cpu_optim/
    chunk_0.pt                # one file per non-persistent chunk
    chunk_3.pt
    chunk_5.pt
    ...
```

`metadata.json`:
```text
{
  "format_version": 1,
  "protrain_layout_signature": "<sha256 of layout fingerprint>",
  "protrain_persistent_ids": [0, 1, 2, ..., 129],   // EFFECTIVE set after non-block expansion
  "protrain_n_buffer": <int>,
  "protrain_world_size": 1,
  "protrain_zero3_shard": false,
  "param_groups_meta": [
    {"lr": ..., "betas": ..., "eps": ..., "weight_decay": ...}
  ],
  "saved_at_step": <int>,
  "torch_version": "...",
  "axolotl_version": "..."
}
```

Notes:
- **`protrain_persistent_ids` is the effective set**, not `n_persist`.
  That captures the non-block force-pin expansion in §1.4. This is what
  Option A from §8.1 pins on resume.
- One file per non-persistent chunk → enables streaming save (no
  84GB-in-RAM blob). Each file is `torch.save(inner_optim.state_dict(),
  ...)`.
- `gpu_optim.pt` may be absent if no chunks are persistent.
- `cpu_optim/` may be empty if every chunk is persistent.
- `metadata.json` is JSON, not a pickle, so it can be inspected with
  `cat`/`jq` for debugging.

### 2.2 What we DON'T save

- Per-param hooks — reinstalled by `materialize_offload` on resume.
- CPU shard buffers (`_cpu_slots`, `_chunk_shards`) — reconstructed by
  `materialize_offload` on resume from the model's GPU params.
- Profiler trace — already cached separately under
  `~/.cache/protrain/profiler/`.
- Search results / cost-model state — out of scope here, tracked as a
  separate concern.

### 2.3 How save fires

A `ProTrainOptimizerCheckpointCallback(TrainerCallback)` is registered
via plugin during `post_trainer_create`. It implements:

- **`on_save(args, state, control, **kwargs)`**: triggered after HF
  Trainer writes its standard checkpoint files. Reads the optimizer
  off the trainer (via `kwargs['optimizer']` or stored ref), checks
  the `protrain_save_optimizer_state` config. If false → skip. If true
  → write to `args.output_dir/checkpoint-{state.global_step}/protrain_optim/`.
- **Resume/load path:** the shipped implementation hooks into
  `Trainer._load_optimizer_and_scheduler` via a monkey-patch / override
  installed in `post_trainer_create`. The previously-considered
  `on_load_checkpoint` callback variant is not used.

Inside the callback's save:
```text
1. Compute current layout signature; build metadata dict.
2. mkdir protrain_optim/, write metadata.json.
3. If self._gpu_optim is not None:
     torch.save(self._gpu_optim._optim.state_dict(), 'gpu_optim.pt')
4. For chunk_id, inner in self._cpu_optim._optims.items():
     mkdir cpu_optim/
     torch.save(inner.state_dict(), f'cpu_optim/chunk_{chunk_id}.pt')
```

Each per-chunk write is bounded by chunk size (default `S_chunk` ~
hundreds of MB), so peak RAM during save is one chunk's optimizer
state, not the whole model's.

### 2.4 How load fires

Load is triggered by HF Trainer's `_load_optimizer_and_scheduler`,
which the plugin wraps via monkey-patch in `post_trainer_create`
(no `on_load_checkpoint` callback exists).

```text
1. Read metadata.json. Validate format_version == 1.
2. Validate world_size == 1 (Phase 1 single-rank guard). Else error.
3. Validate zero3_shard == False. Else error.
4. Compare persistent_ids against the current run's effective set:
   - If different AND Option A in effect (§8.1): hard error,
     suggest passing the saved set as override.
   - (Option B not in scope for Phase 1.)
5. Compare layout_signature: hard error on mismatch.
6. If gpu_optim.pt exists: torch.load(map_location='cpu'),
   then self._gpu_optim._optim.load_state_dict(loaded). Inner load
   handles device placement.
7. For each chunk_*.pt under cpu_optim/:
     parse chunk_id from filename
     loaded = torch.load(file, map_location='cpu')   # CPU on purpose
     self._cpu_optim._optims[chunk_id].load_state_dict(loaded)
8. Validate param_groups_meta against current optimizer defaults;
   warn (don't error) on lr/wd drift.
```

**Key explicit choice:** all `torch.load` calls use `map_location='cpu'`.
We never let HF's `map_location=device` infect this path. After load,
each inner optimizer's `load_state_dict` will place its tensors
correctly (GPU adam on GPU, CPU adam on CPU).

### 2.5 Plugin layer changes

Three changes to `plugin.py`:

1. **`get_training_args`** (lines 302-314): unchanged in behavior —
   continue to force `save_only_model=True` UNLESS
   `protrain_save_optimizer_state=True` AND a "size+runtime safe"
   precondition is met (see §2.7). When opt-in, return
   `{"save_only_model": False}` so HF tries to save (our callback
   then takes over the actual write). Keep `save_only_model=True` as
   the default.
2. **`post_trainer_create`** (lines 491-520): keep the no-op patches
   for `state_dict` / `load_state_dict`. These remain correct for the
   Accelerate `prepare` round-trip (§1.7, Option P). Real save/load
   does NOT go through these methods; it goes through the callback.
3. **Register `ProTrainOptimizerCheckpointCallback`** via
   `trainer.add_callback(...)` after the optimizer is installed.

The `_ProTrainOptimizer.state_dict` / `load_state_dict` in
`api/optim_wrapper.py` continue to raise `NotImplementedError` — they
are NEVER the right path. Document this in the docstring.

### 2.6 New YAML flag

`protrain_save_optimizer_state: bool = False` (default off).

Positive name (per §8.2). Save-only — does NOT conflate with load.
Load is implicit: if the checkpoint dir contains `protrain_optim/`,
the callback loads from it.

### 2.7 Save size & gating policy

A 7B-LoRA checkpoint's optimizer state is small (~tens of MB). A 7B
full-FT optimizer state is ~84 GB (fp32 × 2 buffers × ~14B numel).
We don't want to default-write 84 GB blobs.

**Gating logic before save:**
1. Compute `estimated_optim_state_bytes` by walking the inner adapter
   state dicts (`_gpu_optim._optim.state` and every
   `_cpu_optim._optims[*].state`), summing each tensor's bytes
   (`numel × element_size`). This matches exactly what gets pickled
   to disk modulo Python object overhead. Walking the user-facing
   `optim.param_groups` instead would undercount: after
   `ChunkManager.materialize_offload` runs, every offloaded param's
   `.data` is replaced with an empty placeholder, so `p.numel()`
   returns 0 between training steps and the estimate would miss every
   offloaded chunk's optimizer state — producing silent 84 GB writes
   for 7B full-FT.
2. Compare against `protrain_optim_save_max_bytes` (default
   `2 * 1024**3`, i.e., 2 GiB — small enough that LoRA always passes,
   full-FT never silently passes).
3. If estimate > max:
   - If `protrain_optim_save_max_bytes` was explicitly set by user →
     proceed (they opted in).
   - Else → emit a loud WARN with the estimated size, instruct user to
     either set `protrain_optim_save_max_bytes` higher or accept that
     saves are skipped, and skip the save.
4. If estimate ≤ max: proceed.

This means the default behavior is: small models / LoRA checkpoint
their optimizer; big full-FT runs warn and don't write a giant blob
unless the user explicitly raises the threshold.

(Alternative design: implement true streaming save/load with disk
quotas, no gating threshold. More work. Phase 1 ships with the gate;
streaming is a follow-up.)

### 2.8 Failure modes & how to surface them

| Failure mode | Detection | Surface |
|---|---|---|
| World size != 1 on save or load | metadata field check | Hard error (Phase 1 scope) |
| ZeRO-3 active | metadata field check | Hard error (Phase 1 scope) |
| `persistent_ids` mismatch (Option A) | Set comparison | Hard error, suggest override |
| Layout signature mismatch | Hash comparison | Hard error, name differing fields |
| Inner-optimizer state shape mismatch | torch's own `load_state_dict` | Hard error, name the tensor |
| Saved `cpu_optim/chunk_N.pt` missing | File walk vs. set | Hard error, name the chunk |
| Saved chunk_id not present in current optimizer | Set diff | Hard error, suggest the layout-signature path |
| User changed lr/wd | `param_groups_meta` compare | Warn, log old vs new |
| Estimate > save-size threshold | Pre-save gate | Warn, skip save |
| `protrain_save_optimizer_state=False` | Config check | Skip save silently (current behavior) |
| Format version unknown | metadata field check | Hard error, name versions |

### 2.9 Edge cases worth calling out before code

1. **Empty-state load.** If user saves before any `step()` ran, every
   inner state_dict is empty. Load should accept silently.
2. **Persistent-only configs.** When `force_all_persistent=True`,
   `cpu_optim` is `None`. `cpu_optim/` directory should be empty.
3. **Mixed-precision optimizer state.** DeepSpeedCPUAdam stores
   momentums fp32 by default. Don't downcast on save.
4. **Concurrent saves.** Trainer's save can fire from a callback
   while a CPU adam step is in flight. The write must call
   `chunk_manager.wait_cpu_optim_all()` first to drain pending steps,
   so we don't snapshot half-stepped state.
5. **Save during phase-2 rebuild window.** Phase-2 measurement happens
   on cache miss during wrapper construction, *before* any training
   step. So the save callback never fires mid-rebuild. (If this ever
   changes, revisit.)

### 2.10 Phase 1 test plan

Tests live under `tests/protrain/test_optimizer_checkpoint.py` (new
file). Use existing `_tiny_model()` / `_build_chunk_manager()` helpers
from `tests/protrain/test_chunk_manager_offload.py` for consistency.

**Unit tests (fast, in fast suite):**

| Test | What it proves |
|---|---|
| `test_state_dict_round_trip_persistent_only` | All-persistent: save → load on a fresh wrapper reproduces inner-state bit-identical |
| `test_state_dict_round_trip_with_offload` | Mixed config: both GPU and CPU inner state survive round-trip |
| `test_save_format_layout_one_file_per_chunk` | Save produces metadata.json + gpu_optim.pt + cpu_optim/chunk_*.pt with the right names |
| `test_save_uses_map_location_cpu_on_load` | Mock torch.load, verify map_location='cpu' is passed every call |
| `test_load_rejects_world_size_mismatch` | metadata.world_size=2 with current=1 → RuntimeError |
| `test_load_rejects_zero3_mismatch` | metadata.zero3_shard=true with current=false → RuntimeError |
| `test_load_rejects_persistent_ids_mismatch` | metadata.persistent_ids != current effective set → RuntimeError |
| `test_load_rejects_layout_signature_mismatch` | metadata.layout_signature differs → RuntimeError |
| `test_load_warns_on_lr_change` | Change lr between save/load → log warning, load succeeds |
| `test_load_handles_empty_state` | Save before any step → load on fresh succeeds, inner states empty |
| `test_load_rejects_missing_chunk_file` | Tamper with cpu_optim/, remove a file → RuntimeError naming the chunk |
| `test_save_gate_blocks_when_estimate_exceeds_max` | Estimated bytes > max → save skipped, warn logged |
| `test_save_gate_proceeds_when_user_overrides_max` | User explicitly raises max → save proceeds |
| `test_accelerate_prepare_round_trip_unaffected` | Real implementation does NOT break the existing prepare round-trip (no-op patches still active) |
| `test_save_drains_cpu_optim_before_snapshot` | Save callback calls wait_cpu_optim_all() before reading state_dict |

**Integration test (slow suite):**

| Test | What it proves |
|---|---|
| `test_7b_lora_resume_matches_continuous` | Train 7B-LoRA 5 steps with checkpoint at step 3 → resume → final loss matches reference 5-step continuous run, tolerance 1e-3 on loss |

The integration test guards on world_size==1 to keep it Phase 1.

### 2.11 What's NOT in Phase 1

- Multi-rank replicated mode (DDP) — Phase 2
- ZeRO-3 sharded mode — Phase 2
- Migration across persistent-set changes (Option B from v1) — deferred
- True streaming save/load (no in-memory chunk dict at all) — deferred,
  the per-chunk file layout already bounds peak RAM but per-chunk write
  itself is in-memory
- Saving search results / cost-model state alongside the optimizer —
  separate concern

---

## 3. Phase 2: multi-rank (replicated AND ZeRO-3 sharded)

**Phase 2 has its own design note: `CHECKPOINT_DESIGN_PHASE2.md`.**
Read that doc for the detailed schema, save/load flows, validation
matrix, and test plan covering DDP-replicated and ZeRO-3 sharded
modes.

Phase 2 is **not** "Phase 1 with sharded tensors." Both multi-rank
replicated AND ZeRO-3 sharded require multi-rank save/load
coordination (per-rank shard files for sharded mode, rank-0-only
writes for replicated mode, dist.barrier framing, broadcast-of-gate-
decision for cross-rank consistency, region-layout metadata for the
sharded reload contract). The Phase 2 doc lays out the file-naming
convention, schema bump (v1 → v2 with forward compat), and the
~12-test ship gate.

---

## 4. Recommended schema (TL;DR)

Phase 1, on disk under `output_dir/checkpoint-N/protrain_optim/`:

```text
metadata.json:
{
  "format_version": 1,
  "protrain_layout_signature": str,        # sha256 of layout fingerprint
  "protrain_persistent_ids": list[int],    # EFFECTIVE set after non-block expansion
  "protrain_n_buffer": int,
  "protrain_world_size": 1,                # Phase 1 = always 1
  "protrain_zero3_shard": false,           # Phase 1 = always false
  "param_groups_meta": list[dict],         # lr/betas/eps/wd
  "saved_at_step": int,
  "torch_version": str,
  "axolotl_version": str
}

gpu_optim.pt:                              # may be absent
  torch.save(self._gpu_optim._optim.state_dict(), ...)

cpu_optim/chunk_<N>.pt:                    # one per non-persistent chunk; cpu_optim/ may be empty
  torch.save(self._cpu_optim._optims[N].state_dict(), ...)
```

Phase 2 extends with `saving_rank: int` and `protrain_save_mode:
"replicated" | "sharded"` in metadata, a `regions_per_chunk` mapping
(stringified `ChunkId` → ordered list of region descriptor objects with
fields `chunk_offset`, `region_bytes`, `region_bytes_padded`,
`shard_bytes`, and `dtype`; see `CHECKPOINT_DESIGN_PHASE2.md` §3.2 for
the full schema diff), and `cpu_optim/chunk_<N>_rank_<R>.pt` naming for
the sharded mode (Phase 1 files keep the legacy `cpu_optim/chunk_<N>.pt`
layout).

`format_version` bumps when fields change. Phase 1 is v1; Phase 2 is v2.

---

## 5. Recommended load ordering (TL;DR)

Phase 1:
1. Wrapper built (incl. `materialize_offload`, hooks live).
2. `_ProTrainOptimizer` constructed (empty inner states).
3. Trainer attaches optimizer, no-op patches stay active for the
   Accelerate `prepare` round-trip.
4. ProTrain's `trainer._load_optimizer_and_scheduler` monkey-patch runs:
   read metadata, validate single-rank + non-ZeRO + persistent_ids match,
   then load each shard with `map_location='cpu'` and call inner
   `load_state_dict`.
5. First step proceeds with restored momentums.

---

## 6. Failure modes catalog (TL;DR)

| Failure | Phase | Surface |
|---|---|---|
| Schema version unknown | Both | Hard error |
| World size != 1 | Phase 1 | Hard error |
| ZeRO-3 mismatch | Phase 1 | Hard error |
| Layout signature mismatch | Both | Hard error |
| `persistent_ids` mismatch | Both | Hard error, suggest override |
| Region layout mismatch | Phase 2 | Hard error |
| Inner state_dict tensor shape mismatch | Both | Hard error (torch raises) |
| Missing per-chunk file | Both | Hard error |
| Hyperparam (lr/wd) drift | Both | Warn, continue |
| Empty saved state | Both | Accept silently |
| Estimate > save threshold | Both | Warn, skip save |
| `protrain_save_optimizer_state=False` | Both | Skip save silently |

---

## 7. Minimum viable test set (TL;DR)

Phase 1 ship gate:
- `test_state_dict_round_trip_persistent_only`
- `test_state_dict_round_trip_with_offload`
- `test_save_format_layout_one_file_per_chunk`
- `test_save_uses_map_location_cpu_on_load`
- `test_load_rejects_world_size_mismatch`
- `test_load_rejects_zero3_mismatch`
- `test_load_rejects_persistent_ids_mismatch`
- `test_load_rejects_layout_signature_mismatch`
- `test_save_gate_blocks_when_estimate_exceeds_max`
- `test_accelerate_prepare_round_trip_unaffected`
- `test_save_drains_cpu_optim_before_snapshot`
- `test_7b_lora_resume_matches_continuous` (slow suite)

Phase 2 has its own ship gate / test plan; see
`CHECKPOINT_DESIGN_PHASE2.md` and the Phase 2 test set in
`tests/protrain/test_optimizer_checkpoint.py` +
`tests/protrain/test_world_size_reshard.py` for the captured suite.

---

## 8. Historical design notes (v2-shipped decisions)

These were open questions during the v1 → v2 design pass; all five
are now resolved and shipped. Captured here so readers don't dig
through git history to find why each knob landed the way it did.

1. **Save-size gate threshold default — DECISION:**
   `protrain_optim_save_max_bytes = 2 GiB` is the shipped default.
   Blocks unintentional 84 GB writes for full-FT (typical 70B-class
   model) and lets every LoRA pass without configuration. Operators
   can override in YAML when full-FT saves are intentional.

2. **Callback hook vs. trainer override — DECISION:** the save side
   uses `TrainerCallback.on_save`; the load side patches
   `Trainer._load_optimizer_and_scheduler` because HF does not
   expose a reliable `on_load_checkpoint` hook (see §1.8 for the
   monkey-patch contract).

3. **`save_only_model` flip precondition — DECISION:** `save_only_model =
   True` by default. Flipped to `False` only when
   `protrain_save_optimizer_state=True` AND the size gate passes,
   evaluated **at every save call** (not at config time). The
   per-call check pays a single broadcast in exchange for surfacing
   skip decisions exactly when they happen — operators see the warning
   in the same `on_save` log line that produced the decision.

4. **Streaming — DEFERRED.** Phase 1 shipped the size gate; streaming
   is a follow-up. The two-step shape held up under v2 review and the
   gate alone covers the dominant failure mode (silent oversized
   saves), so streaming is out of scope for the v2 ship and tracked
   separately.

5. **Accelerate `prepare` coexistence — DECISION:** Option P
   (instance-level `state_dict` / `load_state_dict` no-op patches
   stay; real save/load routes through the dedicated
   `_save_protrain_optim_dir` / load-hook path). Option Q (real
   `state_dict` as the only path, paying the prepare-time HBM spike)
   was rejected because the spike crosses 24 GiB on the target 7B
   workloads.

---

## 9. Resolved decisions (from v2 corrections)

- **`n_persist` migration on resume:** Option A (pin saved
  partition). Save the **effective `persistent_ids`** set, not just
  `n_persist`, so the non-block force-pin pass is captured.
- **YAML flag name:** `protrain_save_optimizer_state` (positive,
  save-only; does not conflate with load).
- **Default `save_only_model` flip:** No global flip. `True` stays the
  default. Flip to `False` only when `protrain_save_optimizer_state=True`
  AND the size+runtime path is safe.
- **Phase scoping:** Phase 1 = single-rank, non-ZeRO only. Phase 2 =
  multi-rank (DDP) AND ZeRO-3 sharded; both need per-rank save/load
  control and warrant their own design pass.
- **Streaming default:** Don't default to in-memory writes for full-FT
  scale. Implement gating first; streaming comes later or as Phase 1.5.

---

*Historical note: this document captures the pre-implementation design
decisions for Phase 1 (single-rank, non-ZeRO). Phase 1 has shipped;
Phase 2 (DDP + ZeRO-3) is covered in `CHECKPOINT_DESIGN_PHASE2.md`.
Retained for context on why the current code looks the way it does.*
