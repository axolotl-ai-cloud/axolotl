# ProTrain Optimizer Checkpoint/Resume — Phase 2 Design Note

**Status:** implemented (M5 + Mode-C Phase 2 shipped on branch `protrain-optim-checkpoint-phase2-mode-c`)
**Scope:** multi-rank replicated (DDP) AND ZeRO-3 sharded checkpoint/resume
**Builds on:** Phase 1 single-rank, non-ZeRO checkpoint/resume documented in `CHECKPOINT_DESIGN.md` (callback wiring, atomic save, manifest schema)

Phase 1 is single-rank by hard-coded guard. Phase 2 lifts that guard
in two distinct configurations that need different handling:

* **Mode-B (replicated CPU-offload, DDP):** every rank holds the full
  optimizer state for the full chunk set. State is identical across
  ranks (modulo numerical noise) because DDP all-reduces grads before
  the per-param hooks fire CPU adam.
* **Mode-C (ZeRO-3 sharded CPU-offload):** each rank holds only its
  slice of each non-persistent chunk's regions; persistent (GPU)
  optimizer state remains replicated.

These differ enough in save/load shape that the design treats them as
two distinct flows under one umbrella callback.

---

## 0. What carries over from Phase 1

Recap of decisions Phase 1 already made that Phase 2 inherits
unchanged:

* Save side is a `TrainerCallback.on_save`. HF's `on_save` fires on
  every rank (verified in `_maybe_log_save_evaluate` line 48 of the
  trainer source — `_save_checkpoint` and `on_save` both run
  unconditionally; rank-0-only writes inside `_save_checkpoint` are
  gated by `args.should_save` per-block).
* Load side is a monkey-patched `trainer._load_optimizer_and_scheduler`
  — HF has no `on_load_checkpoint` callback, and `on_train_begin`
  fires after the load slot. The patch is per-rank (each rank's
  trainer gets its own).
* `optim.state_dict` / `optim.load_state_dict` no-op patches stay
  active to coexist with Accelerate's `prepare` round-trip.
* `map_location='cpu'` discipline for every `torch.load` call —
  defeats HF's hostile `map_location=device` default.
* The save-size gate (`protrain_optim_save_max_bytes`, default 2 GiB)
  applies the same way; per-rank estimate counts the rank's own state.
* Schema versioning via `format_version` — Phase 2 bumps to **v2**.
* All save/load files live under
  `{checkpoint_dir}/protrain_optim/`. Per-rank file naming distinguishes
  shards (see §2.1, §3.1).
* `protrain_save_optimizer_state` flag stays. Phase 2 also introduces opt-in knobs `protrain_save_optim_verify_replicated` (replicated-state cross-rank verification), `protrain_allow_online_reshard` (online world-size resharding on load), and `protrain_optim_save_max_bytes` (per-rank save-size cap). Defaults preserve Phase 1 behavior.

---

## 1. Key facts that shape Phase 2

### 1.1 In Mode-B (DDP-replicated), every rank holds identical optimizer state

Verified from the runtime:

* `materialize_offload` runs on every rank, partitioning the same
  chunk set into the same persistent / non-persistent split.
* DDP all-reduces gradients before the per-param post-accumulate-grad
  hooks fire (`skip_internal_grad_reduce=True` in `post_trainer_create`
  when DDP composition is detected — see plugin.py:561-582).
* Per-rank CPU adam steps fire from those hooks with the same grad
  values, against the same starting weights, with the same
  hyperparams. So the resulting state is byte-identical across ranks.

**Implication:** Mode-B save can be **rank-0-only**. Other ranks skip
the write. On load, every rank reads the same files. This matches the
classic "DDP optimizer save" pattern.

There is one corner case to check: **floating-point determinism in the
C++ kernel**. DeepSpeedCPUAdam's `adam_update` kernel processes
elements deterministically per-thread, and same-input + same-seed must
produce same-output. We trust this (it's table stakes for DeepSpeed)
but a sanity check on cross-rank state equality during the first save
is cheap insurance — see §2.4.

### 1.2 In Mode-C (ZeRO-3 sharded), per-rank state is genuinely different

* `materialize_offload` partitions each non-persistent chunk into
  per-rank shards (one `shard_param` per dtype region per rank;
  `manager.py:753-836`).
* The CPU adam is built over those `shard_param` objects
  (`model_wrapper.py:918-926`). Each rank's CPU adam owns only its
  slice.
* Persistent (GPU) optimizer state is **NOT sharded** in ProTrain —
  the GPU FusedAdam in `_gpu_optim` is built over the full persistent
  param list on every rank.

**Implication:** Mode-C save needs per-rank shard files. Mode-C load
needs per-rank shard reads. Persistent state can still be saved
rank-0-only (or saved per-rank with cross-rank consistency check).

### 1.3 Region layout is part of the load contract for Mode-C

The sharded path's `_DtypeRegion` records (per chunk):
* `chunk_offset` — byte offset within chunk
* `region_bytes` — valid bytes in the region (un-padded)
* `region_bytes_padded` — padded bytes (rank-evenly-divisible)
* `shard_bytes` — bytes per rank for this region
* `dtype` — region's element dtype
* (the `shard_param` is rebuilt fresh on load, not persisted)

If the current run's region layout differs from the saved one
(different dtype mix, different total chunk_bytes after dtype-mixed
alignment, different world_size changing shard_bytes), the saved per-
rank shard tensors won't fit the rebuilt `shard_param`. Catching this
explicitly with a load-time check beats letting torch's
`load_state_dict` crash with a shape error 200 lines deep.

### 1.4 Cross-rank coordination on save needs `dist.barrier()`

The save flow per rank:
1. Drain in-flight CPU adam (`wait_cpu_optim_all` — already in Phase 1).
2. Compute estimate, validate scope (world_size > 1 or zero3_shard
   are now valid in Phase 2).
3. Write own files (rank-0: metadata + persistent state; sharded:
   own shard files).
4. `dist.barrier()` to make sure all rank shards are on disk before
   any caller (Trainer, downstream callbacks) trusts the directory
   structure.

The load flow is the inverse: barrier → all ranks have read their
shards → safe to proceed. But since each rank's load is independent
(no cross-rank file access), the barrier on load is a defensive
sanity check rather than a strict requirement.

### 1.5 HF Trainer's process_index and should_save are the right gates

* `args.process_index` — 0..world_size-1 per-rank ordinal.
* `args.should_save` — `True` only on rank-0 in DDP/FSDP modes.
* `args.world_size` — total ranks.

We use these directly. No need to re-derive from `torch.distributed`
inside the callback — HF's view is canonical for what HF will load
later.

---

## 2. Mode-B (DDP-replicated) save & load

### 2.1 On-disk layout

```text
{checkpoint_dir}/protrain_optim/
  metadata.json                         # rank-0 only
  gpu_optim.pt                          # rank-0 only (replicated state)
  cpu_optim/
    chunk_0.pt                          # rank-0 only
    chunk_3.pt
    ...
```

Same as Phase 1. No per-rank suffixes. No rank stamps in filenames.

### 2.2 metadata.json (v2)

```text
{
  "format_version": 2,
  "protrain_layout_signature": str,
  "protrain_persistent_ids": list[int],
  "protrain_n_buffer": int,
  "protrain_world_size": int,           # may be > 1 in Phase 2
  "protrain_zero3_shard": false,        # Mode-B = false; Mode-C = true
  "protrain_save_mode": "replicated",   # NEW: "replicated" or "sharded"
  "param_groups_meta": list[dict],
  "saved_at_step": int,
  "torch_version": str,
  "estimated_optim_state_bytes": int,
  "saving_rank": 0
}
```

`protrain_save_mode` is a new explicit field. Could be derived from
`zero3_shard`, but storing it explicitly makes a grep/jq inspection
unambiguous and lets a future shape (e.g., partial-rank save) coexist.

### 2.3 Save flow — Mode-B

```text
1. All ranks: drain wait_cpu_optim_all().
2. All ranks: compute estimate, check scope (zero3_shard==False here).
3. If args.process_index == 0:
     a. Compute layout signature.
     b. Write metadata.json with protrain_save_mode="replicated".
     c. Write gpu_optim.pt.
     d. Write cpu_optim/chunk_<N>.pt for each non-persistent chunk.
4. Other ranks: NO writes.
5. dist.barrier() — make sure rank-0's writes are flushed before any
   downstream code touches the dir.
```

### 2.4 Cross-rank consistency check (one-time, optional)

The first save in a run can do a one-time cross-rank state-equality
check to catch the corner case where DDP determinism doesn't hold
(numerical drift, manual user override, etc.):

```text
on first save of a run:
  for each non-persistent chunk:
    h_local = sha256(rank's inner state_dict bytes)
    gathered = dist.all_gather_object(h_local)
    if not all-equal(gathered):
      raise RuntimeError(
        "Mode-B precondition violated: optimizer state diverges "
        "across ranks. Refusing to save (rank-0's state would not "
        "represent the cluster). World ranks reporting different "
        "hashes: ..."
      )
```

This is **opt-in via a separate flag** (`protrain_save_optim_verify_replicated`,
default False) because it's expensive (full state hash, all_gather).
On a clean DDP run it always passes; we offer it for paranoid
operators but don't pay the cost by default.

The flag is **Mode-B only**. The callback gate skips the check on
Mode-C and on single-rank runs: under Mode-C every rank holds a
genuinely different shard, so cross-rank hashing would always
report divergence and falsely abort the save. Implementation: the
gate requires `verify_replicated and not done and world_size > 1
and not zero3_shard`.

### 2.5 Load flow — Mode-B

```text
1. All ranks: read metadata.json (every rank reads it; no broadcast
   needed — same file).
2. All ranks: validate
     - format_version == 2
     - protrain_save_mode in {"replicated", "sharded"} AND matches
       current zero3_shard
     - protrain_world_size: see §4.1 for the policy
     - layout signature matches
     - persistent_ids match
3. All ranks: load gpu_optim.pt with map_location='cpu' →
   gpu_optim._optim.load_state_dict(loaded).
4. All ranks: walk cpu_optim/, load each chunk_<N>.pt with
   map_location='cpu' → cpu_optim._optims[N].load_state_dict(loaded).
5. dist.barrier() (optional — defensive).
```

Same files read by every rank. No collective needed for state
distribution because the data on disk is already what every rank
needs.

---

## 3. Mode-C (ZeRO-3 sharded) save & load

### 3.1 On-disk layout

```text
{checkpoint_dir}/protrain_optim/
  metadata.json                         # rank-0 only
  gpu_optim.pt                          # rank-0 only (replicated GPU state)
  cpu_optim/
    chunk_0_rank_0.pt                   # each rank writes its own
    chunk_0_rank_1.pt
    chunk_3_rank_0.pt
    chunk_3_rank_1.pt
    ...
```

Filename pattern: `chunk_<N>_rank_<R>.pt`. This generalizes Phase 1's
`chunk_<N>.pt` — Phase 1 effectively had implicit rank=0 only.

### 3.2 metadata.json (v2 sharded extensions)

```text
{
  "format_version": 2,
  ... (all Mode-B fields) ...,
  "protrain_save_mode": "sharded",
  "protrain_zero3_shard": true,
  "regions_per_chunk": {
    "0": [
      {
        "chunk_offset": 0,
        "region_bytes": 1234,
        "region_bytes_padded": 1280,
        "shard_bytes": 320,
        "dtype": "torch.float16"
      },
      ...
    ],
    "3": [...]
  }
}
```

`regions_per_chunk` is the new field. Keys are stringified ChunkIds
(JSON only allows string keys); values are the region descriptors
captured at save time. On load, every rank verifies its current
chunk's regions match the saved descriptors exactly — this catches
dtype-mix changes, world-size-driven shard-bytes changes, and any
alignment differences.

### 3.3 Save flow — Mode-C

```text
1. All ranks: drain wait_cpu_optim_all().
2. All ranks: compute estimate, check scope (zero3_shard==True here).
3. If args.process_index == 0:
     - Compute layout signature.
     - Write metadata.json with protrain_save_mode="sharded" and
       regions_per_chunk[<cid>] = [{...}, ...] for every non-persistent
       chunk.
     - Write gpu_optim.pt (replicated GPU state — only rank-0 writes,
       since all ranks have the same persistent state).
4. All ranks: write own shard files
     - For each non-persistent chunk in self._cpu_optim._optims:
         path = cpu_optim/chunk_<N>_rank_<args.process_index>.pt
         torch.save(inner.state_dict(), path)
5. dist.barrier() — every rank must finish before the dir is
   considered complete.
```

### 3.4 Load flow — Mode-C

```text
1. All ranks: read metadata.json. Validate as in Mode-B, plus:
     - protrain_save_mode == "sharded"
     - regions_per_chunk matches the current run's region layout per
       chunk (chunk_offset, region_bytes, region_bytes_padded,
       shard_bytes, dtype) — exact match required.
2. All ranks: load gpu_optim.pt with map_location='cpu' →
   gpu_optim._optim.load_state_dict(loaded). (Replicated.)
3. All ranks: load own shard files
     - For each chunk in self._cpu_optim._optims:
         path = cpu_optim/chunk_<N>_rank_<args.process_index>.pt
         If file absent → hard error naming missing rank-shard.
         loaded = torch.load(path, map_location='cpu')
         cpu_optim._optims[N].load_state_dict(loaded)
4. dist.barrier() (optional defensive).
```

### 3.5 Region-layout match — what "exact match" means

Every field of every region in `regions_per_chunk[cid]` must equal the
current run's corresponding region's field, in order. Any of these
trip the hard error:
* Different number of regions per chunk (dtype-mix changed)
* Different dtype string at any region index
* Different `chunk_offset`, `region_bytes`, `region_bytes_padded`, or
  `shard_bytes`

Mismatch implies the loaded saved file's bytes won't fit the rebuilt
`shard_param` — fail loud with a useful message instead of a torch
shape mismatch deep in `load_state_dict`.

---

## 4. Cross-cutting validation rules

### 4.1 World-size mismatch policy

Three options:

| Option | Behavior | Tradeoff |
|---|---|---|
| **A** | Hard error if saved world_size ≠ current | Safest. User must resume with the same job shape. Awkward if hardware changes. |
| **B** | Allow Mode-B replicated load into different world_size | Replicated state is shape-independent of world_size, so this is mathematically fine. Different world_size only affects gradient distribution, not optimizer state. Reasonable for Mode-B. |
| **C** | Migration path for Mode-C: re-shard saved state on load when world_size changed | Originally rejected as "lots of code, not warranted for Phase 2's first ship." |

**Implemented (post-Phase-2-first-ship):** **Option B + opt-in
Option C.** Mode-B replicated + world_size change is harmless and
implemented as in the original recommendation. Mode-C now has two
recovery routes for cross-world-size resume; the user picks one
explicitly:

* **Default — offline:** the load path hard-errors on
  `saved_world != current_world` and points the user at
  `scripts/protrain/reshard_optim.py`. The CLI runs offline (no GPUs,
  no `torch.distributed`) and produces a fresh directory at the new
  world_size. The user then resumes against that directory.
* **Opt-in — online:** when the user sets
  `protrain_allow_online_reshard: True` in the ProTrain config, the
  same reshard logic runs in-process at load time. Rank-0 reshards
  into a temp dir under `<saved-protrain_optim>/.reshard_to_N<W>/`,
  every rank `dist.barrier()`s (the failure protocol mirrors the
  Mode-C save's lockstep `_broadcast_status_or_raise` so a rank-0
  reshard failure surfaces on every rank, not just rank-0), and the
  load proceeds against the temp dir as if it were a natively-saved-
  at-N=W checkpoint. Cleanup runs after a successful load; failures
  leave the temp dir for post-mortem inspection. **Off by default**
  because (i) silent automatic resharding can mask configuration
  drift the user might want to be told about, and (ii) writing files
  in (or under) the checkpoint dir as a side-effect of "load" is
  surprising — explicit opt-in keeps the surface conservative.

The reshard logic is a single source of truth shared by both routes:
`src/axolotl/integrations/protrain/api/reshard.py` exposes
`reshard_mode_c_shards(src_dir, dst_dir, target_world_size)`, which
the CLI loads via file-path-based `importlib` (preserving the "no
heavy axolotl imports" property that makes the CLI runnable on a
vanilla CPU host) and the load path imports normally.

The Phase 1 hard error stays for cases where
`saved.zero3_shard ≠ current.zero3_shard` or for save-mode
mismatches (replicated ↔ sharded — see §4.2).

### 4.2 Save-mode mismatch policy

Saved mode must match current mode. Concrete error matrix:

| Saved → Current | Result |
|---|---|
| replicated → replicated | OK |
| replicated → sharded | Hard error (sharding requires per-rank shard files; replicated save has none) |
| sharded → replicated | Hard error (rank-0 cannot reconstruct full state without all ranks' shards on disk in usable form) |
| sharded → sharded | OK if regions match per §3.5 |

### 4.3 Persistent_ids mismatch — same as Phase 1

Hard error. The auto-mode selector (Mode-A/B/C) plus the search may
pick a different `n_persist` between save and load runs, which
changes the chunk partition. Pin it via `protrain_n_persist_override`
to resume.

### 4.4 Estimate gate

A naive design would let each rank gate its own save against its
local estimate. That breaks Mode-C: if rank-0's estimate fits but
rank-1's estimate trips the cap, rank-1 silently skips writing its
`chunk_<N>_rank_1.pt` shards while rank-0 writes the metadata declaring
"saved" — a partial checkpoint that cannot be loaded. Even Mode-B is
fragile under hypothetical state divergence. The gate decision must
be cross-rank consistent.

**Implemented behavior:** rank-0 computes its local estimate and
**broadcasts** the skip-or-save decision via
`torch.distributed.broadcast_object_list`. All ranks act on rank-0's
decision — all save or none do. The metadata records
`estimated_optim_state_bytes` from rank-0's view.

The per-rank `_save_protrain_optim_dir` function still has its own
size-gate for legacy direct callers (Phase-1-style single-rank
tests). The callback path passes `_skip_size_gate=True` so the inner
gate is suppressed and rank-0's broadcast is the single source of
truth.

**Why this works:** rank-0's estimate is representative for Mode-B
(every rank has the same state by DDP determinism) and conservative
for Mode-C (rank-0 holds at most as much as any single rank's shard
slice — and in practice they hold the same shard size when regions
are evenly split). Simpler and cheaper than `all_gather_object`-ing
local decisions. Mode-C edge case where rank shards are wildly
unequal is exotic and can be handled in a follow-up.

**Rejected alternative:** gate locally per-rank, then
`all_gather_object` the decisions and refuse to write anything if
they diverge. Equivalent correctness but adds a round-trip and makes
the failure surface more confusing (every rank participates in a
collective just to discover none of them want to save).

---

## 5. Schema diff Phase 1 → Phase 2

```diff
  {
-   "format_version": 1,
+   "format_version": 2,
    "protrain_layout_signature": str,
    "protrain_persistent_ids": list[int],
    "protrain_n_buffer": int,
-   "protrain_world_size": 1,
+   "protrain_world_size": int,
-   "protrain_zero3_shard": false,
+   "protrain_zero3_shard": bool,
+   "protrain_save_mode": "replicated" | "sharded",
+   "saving_rank": int,
+   "regions_per_chunk": dict[str, list[dict]],   # sharded only
    ...
  }
```

Phase 1 saves under v1 are not auto-readable by Phase 2 code without
a forward-compat path. Two options:

* **Drop forward compat:** v1 saves error on v2 load with a clear
  "this save predates Phase 2; resume from a fresh run" message. User
  cost: any in-flight Phase-1 checkpoints can't be resumed under
  Phase-2 code.
* **Add forward compat:** v2 loader accepts v1 saves by inferring
  `protrain_save_mode="replicated"` and `saving_rank=0` and `world_size=1`
  from absent fields. Cheap to implement, friendly to users.

**Recommendation:** the second. Forward compat is ~10 lines.

---

## 6. Multi-rank save/load orchestration in the callback

Pseudocode for the v2 callback:

```python
class ProTrainOptimizerCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        optim = kwargs.get("optimizer")
        if not _is_protrain_optimizer(optim):
            return control

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if not os.path.isdir(checkpoint_dir):
            return control

        chunk_manager = optim._chunk_manager
        zero3_shard = bool(getattr(chunk_manager, "zero3_shard", False))
        rank = int(getattr(args, "process_index", 0))
        world_size = int(getattr(args, "world_size", 1))

        # Drain async CPU adam — every rank.
        chunk_manager.wait_cpu_optim_all()

        # Estimate gate — broadcast from rank-0 for cross-rank consistency.
        estimate = _estimate_optim_state_bytes(optim)
        skip_decision = [estimate > self._save_max_bytes]
        _broadcast_object_list_or_noop(skip_decision, src=0)
        if skip_decision[0]:
            return control

        target = os.path.join(checkpoint_dir, PROTRAIN_OPTIM_DIRNAME)
        # rank-0 makes the dir; others wait
        if rank == 0:
            os.makedirs(target, exist_ok=True)
        _barrier_or_noop()

        if zero3_shard:
            _save_phase2_sharded(optim, target, rank, world_size, state.global_step)
        else:
            if rank == 0:
                _save_phase2_replicated(optim, target, world_size, state.global_step)

        _barrier_or_noop()
        return control
```

Helpers:
* `_broadcast_object_list_or_noop` and `_barrier_or_noop` no-op on
  single-rank (preserve Phase 1 behavior).
* `_save_phase2_replicated` ≈ Phase 1's `_save_protrain_optim_dir`
  with `format_version=2`, `protrain_save_mode="replicated"`, and
  using HF's `world_size` instead of forcing 1.
* `_save_phase2_sharded`:
  * On rank-0: write metadata.json with regions_per_chunk + write
    gpu_optim.pt.
  * On all ranks: write `cpu_optim/chunk_<N>_rank_<R>.pt` for each
    non-persistent chunk in `self._cpu_optim._optims`.

Symmetric for load:

```python
def install_load_hook(trainer, optim):
    original = trainer._load_optimizer_and_scheduler
    def _patched(checkpoint):
        original(checkpoint)
        if checkpoint is None:
            return
        if not _is_protrain_optimizer(optim):
            return
        target = os.path.join(checkpoint, PROTRAIN_OPTIM_DIRNAME)
        if not os.path.isdir(target):
            return
        meta = _read_and_validate_metadata(target, optim, trainer.args)
        if meta["protrain_save_mode"] == "sharded":
            _load_phase2_sharded(optim, target, meta, trainer.args)
        else:
            _load_phase2_replicated(optim, target, meta)
        _barrier_or_noop()
    trainer._load_optimizer_and_scheduler = _patched
```

---

## 7. Phase 2 test plan

The Phase-2 test suite extends `tests/protrain/test_optimizer_checkpoint.py`
with multi-rank tests. We use **gloo backend** for the cross-rank
infrastructure tests so they don't need NCCL — gloo works on CPU and
exercises the same `dist.barrier` / `dist.broadcast_object_list` /
`dist.all_gather_object` paths. NCCL-only tests live in the slow lane.

### 7.1 Mode-B (replicated) — unit tests

| Test | Coverage |
|---|---|
| `test_replicated_save_only_rank_0_writes` | mp.spawn 2 gloo ranks, save, verify only one set of files (no rank suffix) |
| `test_replicated_load_succeeds_on_all_ranks` | All ranks read the same files into their own optimizers |
| `test_replicated_save_with_protrain_save_optim_verify_replicated_passes_on_clean_run` | The opt-in cross-rank consistency check passes when state is in fact identical |
| `test_replicated_save_with_protrain_save_optim_verify_replicated_catches_divergence` | Tamper with one rank's state pre-save → verify path errors with a clear message |
| `test_replicated_load_v1_checkpoint_is_forward_compat` | Phase-1 (v1) save loads cleanly into Phase-2 code as replicated mode |

### 7.2 Mode-C (sharded) — unit tests

| Test | Coverage |
|---|---|
| `test_sharded_save_writes_per_rank_shard_files` | Each rank writes `chunk_<N>_rank_<R>.pt`; rank-0 also writes metadata + gpu_optim.pt |
| `test_sharded_load_reads_per_rank_shard_files` | Each rank loads its own shard, asserts state matches what it had pre-save |
| `test_sharded_metadata_contains_regions_per_chunk` | metadata.json has the regions_per_chunk dict; entries match runtime DtypeRegion records |
| `test_sharded_load_rejects_region_count_mismatch` | Tamper metadata regions to add a fake region → hard error |
| `test_sharded_load_rejects_region_dtype_mismatch` | Tamper metadata regions dtype string → hard error |
| `test_sharded_load_rejects_missing_rank_shard` | Remove a `chunk_<N>_rank_<R>.pt` file → hard error naming the missing file |
| `test_sharded_load_rejects_world_size_change` | Save 2-rank, attempt 4-rank load → hard error |

### 7.3 Cross-cutting validation tests

| Test | Coverage |
|---|---|
| `test_load_rejects_save_mode_mismatch` | Saved replicated, current sharded → error; and inverse |
| `test_save_estimate_gate_decision_is_broadcast_from_rank_0` | Mock rank-0's estimate above threshold; verify all ranks skip save (not just rank-0) |
| `test_save_with_world_size_2_does_not_double_write` | mp.spawn 2 ranks; verify each non-persistent chunk has exactly one file in replicated mode |

### 7.4 Functional-equivalence tests (slow lane)

These need separate processes per arm to avoid the pinned-host
allocator issue from Phase 1. Use pytest-forked or subprocess.

| Test | Coverage |
|---|---|
| `test_sharded_resume_matches_continuous_2rank` | mp.spawn 2 ranks. Run N steps, save. New mp.spawn run loads, runs M steps. Compare to mp.spawn ref of N+M steps. Tolerance 1e-3 on loss. |
| `test_replicated_resume_matches_continuous_2rank` | Same shape but in replicated mode. |

### 7.5 Test infra notes

* **Helper:** an `mp_spawn` test wrapper that spawns N gloo processes,
  runs a function, and surfaces per-rank assertion failures cleanly.
  Existing `tests/protrain/test_chunk_manager_offload.py::test_sharded_restore_to_gpu_round_trip_2rank`
  (line 1058) shows the pattern — re-use that scaffolding.
* **Avoid pinned-host explosion:** every multi-rank test must exit
  the spawned process cleanly so its pinned-host allocations are
  reclaimed by OS process teardown. No two ChunkManagers in one
  spawned process if avoidable.

---

## 8. Open questions (resolved during implementation)

These were the design choices that needed direction before
implementation. They are recorded here for historical context; the
decisions below are what shipped on
`protrain-optim-checkpoint-phase2-mode-c`.

1. **World-size mismatch policy (§4.1).** Chose Option B: replicated
   world-size changes are allowed; sharded world-size changes are a
   hard error unless resolved via the offline reshard tool or the
   opt-in online reshard mechanism described in §4.1.

2. **Forward compat for v1 saves (§5).** Chose YES — the v2 loader
   accepts v1 saves as `replicated`/`world_size=1` in ~10 lines.

3. **Cross-rank state-equality check in Mode-B (§2.4).** Added the
   opt-in flag, default OFF (matching the recommendation in §2.4). The
   alternatives (no flag; default ON for first save) were rejected.

4. **Estimate-gate broadcast (§4.4).** Chose rank-0-decides +
   broadcast. The per-rank-decides + cross-rank assert alternative was
   rejected as logs-noisier without enough upside.

5. **Functional-equivalence test infra.** Drove the slow correctness
   tests via `subprocess.run` from inside a single test function (the
   dependency-free option) rather than adding pytest-forked as a test
   dep.

6. **`save_only_model` flip in multi-rank.** Phase 1 sets
   `save_only_model=False` so HF saves scheduler.pt + rng_state.pth.
   In Mode-C with HF Trainer's standard distributed checkpoint path,
   verified that HF's rng_state save coexists with our per-rank shard
   path without collision.

7. **Should Phase 2 land as a single PR, or split into Mode-B and
   Mode-C?** Landed as a single branch
   (`protrain-optim-checkpoint-phase2-mode-c`) covering both Mode-B
   and Mode-C rather than splitting for a faster Mode-B win.

---

## 9. Recommended schema (TL;DR)

```text
{checkpoint_dir}/protrain_optim/
  metadata.json                                   # rank-0 only
  gpu_optim.pt                                    # rank-0 only
  cpu_optim/
    chunk_<N>.pt                                  # replicated mode (rank-0)
    chunk_<N>_rank_<R>.pt                         # sharded mode (each rank)
```

`metadata.json` adds `format_version=2`, `protrain_save_mode`,
`saving_rank`, and (sharded only) `regions_per_chunk`.

---

## 10. Recommended load ordering (TL;DR)

1. ProTrain wrapper built (incl. `materialize_offload`, hooks live).
2. `_ProTrainOptimizer` constructed.
3. Per-rank trainer attaches optimizer; no-op `state_dict` patches
   stay active.
4. ProTrain load monkey-patch on `trainer._load_optimizer_and_scheduler`
   fires per-rank: read metadata → validate → load gpu_optim
   (replicated) → load own per-rank shards (sharded) or chunk files
   (replicated) → barrier (defensive).
5. First step proceeds with restored momentums on every rank.

---

## 11. Failure modes catalog (TL;DR additions over Phase 1)

| Failure | Detection | Surface |
|---|---|---|
| Saved Mode-B → current Mode-C | save_mode field check | Hard error (§4.2) |
| Saved Mode-C → current Mode-B | save_mode field check | Hard error (§4.2) |
| Region count differs | regions_per_chunk len compare | Hard error |
| Region dtype differs | regions_per_chunk[i].dtype compare | Hard error |
| Region offsets/sizes differ | per-field compare | Hard error |
| Per-rank shard file missing | os.path.isfile in load loop | Hard error naming chunk + rank |
| Mode-C world_size change | size compare on saved vs current | Hard error |
| Mode-B world_size change | tolerated under Option B | Pass (§4.1) |
| Cross-rank state divergence in Mode-B (with verify flag) | all_gather_object hash compare | Hard error (§2.4) |
| Estimate-gate skip decision diverges across ranks (without §4.4 broadcast) | all_gather_object decision compare | Hard error |
| Phase-1 v1 save loaded under Phase-2 code | format_version field | Pass with `replicated`/`world_size=1` defaults (§5) |

---

## 12. Minimum viable test set (TL;DR ship gate for Phase 2)

* `test_replicated_save_only_rank_0_writes`
* `test_replicated_load_succeeds_on_all_ranks`
* `test_replicated_load_v1_checkpoint_is_forward_compat`
* `test_sharded_save_writes_per_rank_shard_files`
* `test_sharded_load_reads_per_rank_shard_files`
* `test_sharded_metadata_contains_regions_per_chunk`
* `test_sharded_load_rejects_region_count_mismatch`
* `test_sharded_load_rejects_missing_rank_shard`
* `test_sharded_load_rejects_world_size_change`
* `test_load_rejects_save_mode_mismatch`
* `test_save_estimate_gate_decision_is_broadcast_from_rank_0`

The functional-equivalence tests (§7.4) are stretch goals, not ship
gates — they need separate-process infra and run on the slow lane.

---

*This design note was the prerequisite to the feature branch off
`protrain-optim-checkpoint` (Phase 1 landed first), shipped as
`protrain-optim-checkpoint-phase2-mode-c`. The §8 questions were
answered during implementation and the answers are recorded above.*
