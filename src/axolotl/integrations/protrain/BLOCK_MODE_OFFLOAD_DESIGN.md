# Block-Mode OFFLOAD — Design Note (Option B)

**Status:** complete. M1 (types + validator) and M2 (runtime hook) shipped in commit `8264f773`; M3 (scheduler integration) shipped in commit `a1ab8aff`; M4 (cost model + searcher) shipped in commit `ea20710a`; M5 (test enablement — re-enabled the 3 previously-skipped slow tests) shipped in commit `c7c155f7`. All milestones M1–M5 have landed; Option B is fully implemented — see [§7 Implementation roadmap](#7-implementation-roadmap) for the per-milestone summary.
**Scope:** extend the ProTrain runtime so a non-persistent chunk's owning block can run under `BlockMode.NONE` (no recompute) — i.e. the param chunk is gathered for forward, offloaded after forward, AND re-gathered for backward without invoking `torch.utils.checkpoint`.
**Builds on:** `DESIGN.md` (overall plugin), `CHECKPOINT_DESIGN.md` / `CHECKPOINT_DESIGN_PHASE2.md` (style template).
**Branch base:** `protrain-optim-checkpoint-phase2-mode-c` @ tip (per `MEMORY.md::protrain_branch_state`).

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Paper alignment](#2-paper-alignment)
3. [Proposed design](#3-proposed-design)
4. [Cost model implications](#4-cost-model-implications)
5. [Test matrix expansion](#5-test-matrix-expansion)
6. [Risks and open questions](#6-risks-and-open-questions)
7. [Implementation roadmap](#7-implementation-roadmap)
8. [Deferral / kill criteria](#8-deferral--kill-criteria)
9. [Glossary](#9-glossary)

---

## 1. Problem statement

### 1.1 The current contract

The runtime today enforces a strict invariant in
`search/exhaustive.py::block_map_runtime_admissible`:

> If a block owns any non-persistent parameter chunk, that block MUST
> use `BlockMode.CKPT`. NONE/SWAP are only legal when every chunk the
> block touches is persistent.

The reasoning — copied from the docstring — is correctness, not
performance:

> The forward scheduler releases non-persistent chunk storage after
> the block runs, and PyTorch's saved tensors for a normal NONE/SWAP
> block are not a safe persistence mechanism once `param.data` is
> rebound to the empty sentinel. CKPT blocks recompute their forward
> during backward, so the scheduler can re-gather chunks immediately
> before recompute.

In other words: if the block runs in NONE/SWAP, autograd retained
saved tensors that view the GPU buffer that was *just released* in
`post_block_forward`. When backward runs, those saved tensors point
into freed (or recycled) storage — silent corruption at best, segfault
at worst. The CKPT path sidesteps the problem because the recompute
function call re-builds the saved-tensor table fresh, and the
scheduler can re-gather chunks immediately before that call (see
`runtime/scheduler.py::ensure_block_resident`, wired through
`CheckpointedBlock.set_recompute_pre_hook`).

This implication ripples through the searcher: any candidate
`(n_persist, n_buffer, n_swap, n_checkpoint)` whose non-persistent
blocks aren't tagged CKPT is rejected as runtime-inadmissible. In
practice, on a 3B / 7B model the searcher converges on configs where
**every non-persistent block is CKPT** unless the entire layer fits in
the persistent set.

### 1.2 What this blocks experimentally

The MLSys paper's headline ZeRO-3 vs ProTrain comparison is "same
memory budget, fewer recomputed FLOPs". Our v1 implementation cannot
honor that comparison directly because:

* **DeepSpeed Stage-3** does not by default invoke gradient
  checkpointing. It offloads parameters and optimizer state to CPU,
  re-gathers them for backward via `all_gather_into_tensor`, and runs
  the *original* backward graph against the saved tensors — no
  recompute.
* **ProTrain Mode-C** (CPU-offload + ZeRO-3 sharding), as currently
  shipped, has CKPT forced on for every offloaded block. So our
  Mode-C-vs-Stage-3 throughput numbers compare an
  apples-to-recomputed-oranges system: we pay an entire extra forward
  pass per iteration that DeepSpeed does not.

Three slow tests document this gap by failing today with the
`n_checkpoint=0` overrides:

| Test | Location | Failure mode |
|---|---|---|
| `test_protrain_4gpu_zero3_sharding` | `tests/protrain/test_multi_gpu_7b.py:855-934` | `n_checkpoint_override=0` + `n_persist_override=2` configures a Mode-C run where blocks 2..N have non-persistent chunks but mode NONE → searcher path raises `block_map_runtime_admissible=False`, or the worker subprocess silently retags blocks as CKPT (defeating the test's "no recompute" premise). |
| `test_protrain_2gpu_mistral_modec_smoke` | `tests/protrain/test_multi_gpu_7b.py:1337+` | Same pattern: `n_persist_override=1`, `n_checkpoint_override=0` — Mistral has 4 blocks, only block 0 is persistent, blocks 1..3 hit the admissibility check and the searcher fails. |
| `test_modec_vs_deepspeed_stage3_4gpu` | (planned) | Apples-to-apples comparison test that does not yet exist; cannot be written until ProTrain can run a non-persistent block in NONE. |

The first two tests use explicit knob overrides, so the failure surfaces
inside `protrain_model_wrapper` *before* training starts (the searcher
either throws "no feasible config" or quietly bumps `n_checkpoint`).
The third is held back from the test suite until this design lands.

### 1.3 Goal of Option B

Lift the "non-persistent ⇒ CKPT" rule for blocks the user (or
searcher) explicitly opts into. In the new world, a block may be:

| Param chunks | Block mode | Status today | Status after Option B |
|---|---|---|---|
| persistent | NONE | legal | legal (unchanged) |
| persistent | CKPT | legal | legal (unchanged) |
| persistent | SWAP | legal | legal (unchanged) |
| non-persistent | CKPT | legal | legal (unchanged) |
| non-persistent | NONE | **runtime-rejected** | **legal under new path** |
| non-persistent | SWAP | runtime-rejected | (out of scope; see §6.6) |

The "non-persistent NONE" cell is the new feature. It enables the
apples-to-apples DeepSpeed Stage-3 comparison and re-opens a swathe
of the search space the v1 admissibility filter prunes.

---

## 2. Paper alignment

ProTrain (MLSys 2026, arXiv 2406.08334) is primarily a **memory
manager**. The paper's three-mode block taxonomy (§3.1.2) is:

* **NONE** — keep activations on GPU, no recompute, no swap.
* **CKPT** — drop forward activations, recompute in backward.
* **SWAP** — offload forward activations to pinned CPU, prefetch back
  for backward (no recompute).

Crucially, the paper does **not** couple these activation strategies
to the chunk's persistence state. §3.1.1 discusses chunk-level
persistence (`n_persist`, `n_buffer`); §3.1.2 discusses block-level
activation strategy. Eq. 8–10 (App A.2) compute peak memory under any
combination of `(n_persist, n_buffer, n_swap, n_checkpoint)`.

The paper's reference figure (Fig 2 / Fig 4 layouts) shows
configurations with non-persistent chunks AND NONE blocks coexisting:
the chunk is gathered on demand, the activations stay on GPU, and the
chunk is re-gathered on the backward pass. The paper assumes (without
naming the mechanism) that the chunk-management layer can re-materialize
the param storage when backward needs it — which is exactly what
Option B builds.

**Conclusion**: Option B is **paper-aligned**, not a paper extension.
What we have today (the `block_map_runtime_admissible` filter) is a
v1 implementation shortcut that the paper's design space allows but
our runtime didn't yet support. Adding it back-fills the design.

The shortcut was justifiable for v1 because:

* `torch.utils.checkpoint` already exists and ships in PyTorch — no
  custom autograd plumbing needed for CKPT.
* The chunk-state path for backward is independent of the
  saved-tensors path for backward, and we built the chunk-state path
  first (M2 / M4) before the activation-swap path (M5+).

So this design extends an already-paper-aligned axis rather than
introducing new paper-divergent surface.

---

## 3. Proposed design

### 3.1 BlockMode surface — extend NONE or add OFFLOAD?

Two options:

**Option A — extend NONE semantics.** Keep the existing 3-mode enum.
Make `BlockMode.NONE` work for both persistent and non-persistent
chunks; the runtime introspects the chunk persistence state at
attach-time and installs the offload hook only when needed.

* Pros: smaller API surface; no migration cost on `BlockStrategyMap`
  consumers; the cost model already enumerates NONE.
* Cons: the wrapper class' behavior depends on a property of a
  *different* dataclass (the chunk layout). `print(model)` no longer
  fully describes the activation strategy — you have to also know
  the chunk persistence map. Debug-ability drops.

**Option B — add `BlockMode.OFFLOAD`.** A 4th enum value. The wrapper
class for OFFLOAD blocks always installs the param-offload-aware hook,
regardless of chunk persistence state. The validator
(`block_map_runtime_admissible`) is updated to allow either
`{NONE, persistent}`, `{CKPT, anything}`, `{SWAP, persistent}`, or
`{OFFLOAD, non-persistent}`. NONE on non-persistent is still rejected
(degenerate case — no offload hook = unsafe).

* Pros: explicit; `print(model)` shows the strategy; cost model
  enumeration adds a new axis cleanly; failure modes are
  pre-validated at search time, not deferred to runtime.
* Cons: 4-mode enum touches every consumer; `assign_modes` returns a
  4-valued map; serialization (checkpoint manifests, etc.) needs a
  schema bump.

**Recommendation: Option B, the new enum value.** The
debug-ability win matters — every other strategy decision in this
runtime is explicit in the `BlockStrategyMap`, and breaking that
convention for one mode invites future bugs. The migration cost is
mechanical: `assign_modes` is the only producer of `BlockStrategyMap`
today, and the consumers (`dispatcher.wrap_block`, `cost/memory.py`,
`cost/runtime.py`, `runtime/scheduler.py`) all already pattern-match
on the enum.

> **Naming**: `OFFLOAD` reads cleanly against `SWAP` (which is
> activation-swap) and `CKPT` (which is recompute). If reviewers
> prefer `NONE_OFFLOAD` or `PARAM_OFFLOAD` we can rename — the
> semantics and dispatch are unchanged.

### 3.2 Saved-tensors-hooks for parameters

The mechanism is `torch.autograd.graph.saved_tensors_hooks`, the same
primitive `SwappedBlock` uses (`block/swap.py`). The difference:

| | `SwappedBlock` (SWAP) | `OffloadedBlock` (OFFLOAD, new) |
|---|---|---|
| Targets | activations (intermediates) | param tensors (model weights) |
| Pack does | D2H copy to pinned slot | record `(chunk_id, byte_offset, shape, dtype)` metadata; **no copy** |
| Unpack does | H2D copy from pinned slot to fresh GPU buffer | look up `chunk_id` in the manager → `gather(chunk_id)` if not resident → return view into pool buffer |
| Pool used | `ActivationSwapPool` (host pinned slots) | reuses `ChunkManager.buffer_pool` (GPU slots) |
| Cost | one D2H per saved tensor | zero copies in pack; gather amortized across chunk's params |

Pseudocode (deliberately incomplete — the implementation agent owns
exact bookkeeping):

```python
def pack_param_only(t: torch.Tensor):
    # Identify saved tensors that are views of a chunk-managed param.
    # Mechanism: each param.data carries an attribute set at gather
    # time (e.g. ``param._protrain_chunk_id``); when autograd saves a
    # tensor that aliases ``param.data``, we read that attribute via
    # ``t._base`` chain or ``t.untyped_storage().data_ptr()`` lookup
    # in a chunk-id table the manager maintains.
    chunk_id = _find_chunk_owning(t)
    if chunk_id is None:
        # Saved tensor is an activation, not a param — return as-is.
        # Pure-activation handling is the SWAP wrapper's job, NOT
        # ours; pass through to the next outer hook context (or to
        # default save).
        return t
    # Record metadata only. Do NOT keep a strong reference to ``t``
    # because that would pin the GPU storage we are trying to free.
    return _ParamHandle(
        chunk_id=chunk_id,
        byte_offset=_chunk_byte_offset(t),
        shape=t.shape,
        dtype=t.dtype,
        requires_grad=t.requires_grad,
    )

def unpack_param_only(handle):
    if not isinstance(handle, _ParamHandle):
        return handle  # passthrough for non-param saves
    # Re-gather the chunk if it isn't resident; idempotent on hit.
    chunk_manager.gather(handle.chunk_id)
    buf = chunk_manager.buffer_pool.lookup_resident(handle.chunk_id)
    # Reconstruct a view at the original offset/shape. The view
    # shares storage with the pool buffer; chunk_manager guarantees
    # the buffer outlives this backward pass via a refcount the
    # caller increments here and decrements on backward exit.
    view = _slice_chunk_buffer(buf, handle.byte_offset, handle.shape, handle.dtype)
    if handle.requires_grad:
        view.requires_grad_(True)
    return view
```

The crucial difference from SWAP: we never *copy* the bytes. The pack
hook drops its strong reference to the GPU tensor — autograd's
savedtensor table now holds only the metadata handle, so the
underlying chunk buffer is collectible the moment the scheduler
issues `offload(chunk_id)`. The unpack hook re-gathers the chunk
(which may trigger an H2D from the CPU shard) and hands back a view
into the freshly populated buffer.

### 3.3 Scheduler changes

The scheduler's job is to keep param chunks resident at the right
times. Today's lifecycle (per non-persistent chunk owned by a CKPT
block):

```text
forward enters block N:
    pre_block_forward(N)  → ensure_block_resident(N) gathers chunks
    block.forward()       → activations dropped (CKPT internally)
    post_block_forward(N) → offload(chunk) if not in N+1's set
backward enters block N:
    pre_block_backward(N) → gather(chunk)         # for recompute
    block.backward()      → torch.utils.checkpoint replays forward,
                            consumes the just-gathered chunks
    post_block_backward(N) → reduce_grads_and_offload(chunk)
```

Under OFFLOAD, the activation drops AREN'T happening, but the chunk
DOES get offloaded after forward — and the saved tensors point into
that chunk. The new lifecycle:

```text
forward enters block N:
    pre_block_forward(N)  → ensure_block_resident(N) gathers chunks
    block.forward()       → activations + saved-param-views captured;
                            saved_tensors_hooks rewrites param-aliasing
                            saves into _ParamHandle metadata-only
    post_block_forward(N) → offload(chunk)  # safe: saved tensors no
                            longer reference the GPU storage
backward enters block N:
    pre_block_backward(N) → gather(chunk)
                            (this is when the saved-param re-views
                             will resolve)
    block.backward()      → autograd unpack hook fires per saved param,
                            returns a view into the pool buffer; gradient
                            kernels consume both activations + re-viewed
                            params; activation tensors are freed by
                            the autograd engine as Nodes complete
    post_block_backward(N) → reduce_grads_and_offload(chunk)
```

Comparison table:

| Lifecycle event | persistent NONE | persistent SWAP | non-persistent CKPT (today) | non-persistent OFFLOAD (new) |
|---|---|---|---|---|
| Forward gather | once at startup | once at startup | per-block | per-block |
| Forward activations | retained on GPU | D2H to pinned slot | dropped | retained on GPU |
| Forward chunk offload | never | never | yes, after block | yes, after block |
| Backward gather | n/a | n/a | per-block (right before recompute) | per-block (right before backward kernels) |
| Backward activations | resident | H2D from pinned slot | recomputed in-place | resident from forward |
| Param saves point to | live GPU chunk | live GPU chunk | recomputed locals | gathered pool buffer (re-resolved via unpack hook) |

The scheduler change is small: `pre_block_backward` already calls
`gather(chunk)` for any block whose chunks aren't resident; OFFLOAD
piggybacks. The new requirement is **timing**: the gather must
complete *before* the autograd engine invokes the unpack hook for
this block's first saved-param. Today's scheduler runs
`pre_block_backward` from a forward-pre hook on the wrapper module —
that fires *before* autograd starts decoding the block's saved
tensors, so we're already correctly ordered. We will document this
ordering invariant explicitly in the `OffloadedBlock` docstring;
breaking it is the most subtle failure mode.

### 3.4 ChunkManager API changes

Today's `ChunkManager` exposes:

* `gather(chunk_id)` — make chunk GPU-resident; idempotent.
* `offload(chunk_id)` — release GPU buffer; chunk becomes resident on
  CPU only.
* `reduce_grads_and_offload(chunk_id)` — backward path: reduce grads
  cross-rank, drain to CPU shards, release GPU.
* `materialize_offload()` — one-time setup at construction.

The OFFLOAD path needs:

1. **A param → chunk_id resolver**. `_find_chunk_owning(tensor)` in
   the pseudocode. The manager already maintains
   `_params_by_id: dict[ParamId, nn.Parameter]` (used by
   `materialize_offload`) and `_param_to_chunk: dict[ParamId, ChunkId]`
   (the layout). Inversion is O(1) given a known param. The trick is
   identifying which param a saved tensor is a view of — proposed
   approach: tag `param.data` at gather time with a
   `_protrain_chunk_id` int attribute; saved tensors that are views
   of that data inherit nothing but share storage, so we look up via
   `tensor.untyped_storage().data_ptr()` against a
   `dict[storage_ptr, ChunkId]` the manager maintains alongside the
   pool. Cheap (pointer comparison), correctness-aligned (storage
   identity is what autograd actually saved).

2. **A backward-window pin counter on the buffer pool**. When the
   unpack hook re-gathers a chunk during backward, the chunk's pool
   slot must not be evicted by another `acquire(other_chunk)` call
   until *every* saved tensor in the autograd graph has been
   consumed. Mechanism: an `acquire_for_backward(chunk_id) -> handle`
   that bumps a refcount; the handle is returned by the unpack hook
   alongside the view, and the autograd engine's reference to the
   view (held until the consuming Node completes its `apply()`) keeps
   the refcount alive. The scheduler's
   `reduce_grads_and_offload(chunk_id)` only frees the slot once the
   refcount drops to zero. If the refcount is non-zero when reduce
   runs, the manager defers offload to a "post-backward drain" stage
   (queued and executed at the bottom of `Scheduler.drain`).

3. A small new helper:
   `gather_for_backward(chunk_id) -> BackwardHandle` which is the
   primitive the unpack hook calls. It is `gather()` + the refcount
   bump. The reverse (`release_after_backward`) is implicit: when the
   `BackwardHandle` is dropped, the refcount decrements; when the
   counter hits zero AND the scheduler has already queued an offload
   for the chunk, the offload runs.

Public API after Option B:

```text
class ChunkManager:
    # Existing.
    def gather(chunk_id: ChunkId) -> None: ...
    def offload(chunk_id: ChunkId) -> None: ...
    def reduce_grads_and_offload(chunk_id: ChunkId) -> None: ...
    def materialize_offload() -> int: ...

    # New for Option B.
    def gather_for_backward(chunk_id: ChunkId) -> BackwardHandle: ...
    def chunk_id_for_storage_ptr(ptr: int) -> ChunkId | None: ...
```

`BackwardHandle` is a tiny RAII helper holding `(chunk_id, manager)`;
`__del__` decrements the refcount.

### 3.5 `block_map_runtime_admissible` update

Replace the rule with:

> A block is admissible iff:
> * mode is `CKPT` (always safe; recompute re-binds storage), OR
> * mode is `OFFLOAD` (new path; safe because the saved-tensor hook
>   re-binds storage at backward), OR
> * every chunk owned by the block is in the persistent set
>   (NONE / SWAP both safe in this case).
>
> Modes `NONE` and `SWAP` on a block with any non-persistent chunk
> remain **inadmissible** — they would still capture saved tensors
> that don't survive the post-forward offload.

In code (no implementation, illustration only):

```python
mode = block_map[bid]
if mode in (CKPT, OFFLOAD):
    return True
return all(c in persistent for c in chunks_of(bid))
```

### 3.6 `assign_modes` update

`assign_modes(n_swap, n_checkpoint, N_block)` today returns
`{SWAP × n_swap, CKPT × n_checkpoint, NONE × rest}` under the
swap-early / interleave-CKPT / unopt-late rules. Under Option B we
add a new knob `n_offload`, and the function becomes:

```python
assign_modes(n_swap, n_checkpoint, n_offload, N_block) -> BlockStrategyMap
```

The placement rule for OFFLOAD blocks: they should sit in the
**non-persistent tail** of the chunk layout. Concretely: blocks whose
parameter chunks are all in `[n_persist, N_chunk)` are candidates for
OFFLOAD. Among those, OFFLOAD blocks should be placed in the same
"unopt-late" tail as NONE today — they free their PCIe budget on the
forward side (no extra gather for backward in the swap-early window)
and their backward gather competes with reduce-offload in the same
backward window CKPT recompute would have.

The cost-model implications of this placement decision are §4.

A subtle invariant: `assign_modes` does not (and cannot, locally)
know which chunks are persistent — it takes only `N_block`. So either
(a) the function takes a new `block_chunks_persistent: set[BlockId]`
parameter, or (b) the searcher post-validates the assignment via
`block_map_runtime_admissible` and skips infeasible candidates. (b)
is cheaper to implement and matches the existing pattern. We propose
(b).

### 3.7 Worked example

Llama-3B with N_block=26, N_chunk=29, capacity=20 GiB per rank, 4× 3090.
Searcher today on Mode-C (sharded, no DDP):

* Picks `n_persist=2, n_buffer=2, n_swap=0, n_checkpoint=24`.
* Runtime: 1.0 forward + 1.0 recompute + 1.0 backward ≈ 3× compute.
* 24 of 26 blocks recompute every iteration.

With Option B available:

* Searcher can pick `n_persist=2, n_buffer=2, n_swap=0,
  n_checkpoint=0, n_offload=24`.
* Runtime: 1.0 forward + 1.0 backward ≈ 2× compute. The offloaded
  chunks pay an extra H2D each on the backward path (gather at
  pre_block_backward), which the bandwidth model accounts for in §4.
* Comparison vs DeepSpeed Stage-3: now apples-to-apples — both
  systems run forward + backward without recompute; both gather
  chunks H2D for backward; only the chunk-management heuristics
  differ.

---

## 4. Cost model implications

### 4.1 `cost/memory.py`

Peak memory analysis adds OFFLOAD as a new live-bytes term in the
op-walk:

* OFFLOAD blocks contribute *forward-retained* activation bytes (same
  as NONE) during the forward window. They also contribute a
  **backward gather bump**: at each OFFLOAD block's first backward op,
  one chunk's worth of bytes (`S_chunk`) is materialized in the pool
  buffer concurrently with the activations. This bump is identical
  in shape to today's CKPT bump, but smaller: CKPT pays
  `S_chunk + activation_size` (gather + recompute); OFFLOAD pays
  `S_chunk` only.

* Order of contributions, op-walking the full forward + backward:
  forward live-NONE+OFFLOAD is the union (both retain activations).
  CKPT bumps land at first-op of each CKPT block. OFFLOAD bumps
  land at first **backward**-op of each OFFLOAD block.

`estimate_peak`'s op walk already supports a "bump at first op of
block X" pattern (used for CKPT). Adding the symmetric backward-side
bump for OFFLOAD is one more case.

### 4.2 `cost/runtime.py`

Three term updates:

1. **Forward** — unchanged for OFFLOAD blocks (forward is a normal
   compute pass with chunks gathered as today).
2. **Backward** — for OFFLOAD blocks, add a `T_bwd_gather` term per
   block: chunk bytes / effective_h2d_bps, less any overlap with
   the previous backward block's compute. Mirrors the existing
   forward-prefetch overlap accounting.
3. **Recompute** — drops out of the cost when CKPT count goes down.
   The new term is `T_offload_gather` (above); the searcher trades
   recompute time against gather time. Recompute scales with model
   compute per block; gather scales with chunk_bytes / pcie_bw. On
   PCIe Gen3 the trade tilts toward OFFLOAD when blocks are big
   (compute-heavy) but chunks are small (gather-cheap). On NVLink
   it tilts even further toward OFFLOAD; on slow PCIe with tiny
   blocks, CKPT may still win.

### 4.3 Searcher enumeration changes

`search/exhaustive.py` adds an outer loop over `n_offload`. The
combined enumeration:

```python
for n_ckpt in range(0, N_block + 1):
    for n_offload in range(0, N_block - n_ckpt + 1):
        max_swap = min(N_block - n_ckpt - n_offload, N_interval)
        for n_swap in range(0, max_swap + 1):
            for n_persist, n_buffer:
                block_map = assign_modes(n_swap, n_ckpt, n_offload, N_block)
                if not block_map_runtime_admissible(layout, block_map, n_persist):
                    continue
                # ...peak + runtime + capacity gates as today...
```

The search-space size grows by a factor of `~N_block`, from O(N^3) to
O(N^4). For Llama-3B (N_block=26) this takes us from ~17K candidates
to ~440K candidates — still finishes in seconds (the per-candidate
cost is closed-form arithmetic after the M5b shortcut landings). No
new pruning needed.

### 4.4 Calibration / hw_bench

The cost model needs no new measurements per se — H2D bandwidth and
NCCL gather throughput are already captured by `hw_bench`. The new
term `T_bwd_gather` is computed from existing fields.

What we may want to ADD as telemetry-only (no cost-model effect):

* A microbenchmark that times a "gather → compute → offload" cycle
  on a representative chunk size, to validate the cost-model
  prediction empirically. This is a calibration check, not a new
  knob.

---

## 5. Test matrix expansion

### 5.1 The three failing tests (must pass)

1. **`test_protrain_4gpu_zero3_sharding`** — keep the existing
   `n_checkpoint_override=0`, `n_persist_override=2`,
   `n_buffer_override=2`, `n_swap_override=0` config. Add
   `n_offload_override=N_block - n_persist_chunks_blocks` (the
   non-persistent block count) so the search/wrapper builds an
   OFFLOAD-tagged block_map. Asserts:
   * loss decreases across iterations (existing)
   * GPU peak memory matches replicated within 25% (existing)
   * NEW: total recompute time per iteration < 5% of total bwd time
     (proves no recompute is happening — the test's whole premise)

2. **`test_protrain_2gpu_mistral_modec_smoke`** — same change:
   `n_offload_override=3` (4 blocks total, block 0 persistent). The
   primary assertion ("no crash + finite loss") stays.

3. **`test_modec_vs_deepspeed_stage3_4gpu`** (NEW) — apples-to-apples
   throughput comparison. Both systems run Llama-3B + LoRA, bs=2,
   seq=256, fp16 on 4× 3090, world_size=4. ProTrain configured with
   Mode-C + OFFLOAD, DeepSpeed configured with Stage-3 (default — no
   activation checkpointing). Assert ProTrain's iter/s is within
   ±20% of DeepSpeed's, AND ProTrain's per-rank GPU peak is within
   ±15% of DeepSpeed's. The headline number for the paper-fidelity
   plan.

### 5.2 New unit / smoke tests

* **`test_offloaded_block_save_unsave_roundtrip`** — single-block
  unit test that wraps a synthetic linear layer in
  `OffloadedBlock`, runs forward + backward, asserts gradient
  matches a reference (same op without offload) within fp32
  numerical tolerance. Validates the saved-tensors-hooks plumbing
  in isolation.

* **`test_admissibility_under_offload_rule`** — pure function test
  for the updated `block_map_runtime_admissible`. Covers all 4×3
  cells (chunk-persistence × block-mode); verifies new OFFLOAD cell
  passes admissibility and SWAP-on-non-persistent still rejects.

* **`test_assign_modes_with_offload`** — extend
  `tests/protrain/block/test_layout_rules.py`. Verify the new
  `n_offload` axis honors the unopt-late placement rule and doesn't
  collide with SWAP / CKPT slots.

* **`test_search_picks_offload_when_advantageous`** — searcher unit
  test with a synthetic trace where compute-per-block is high and
  PCIe is fast; assert the searcher picks `n_offload > 0,
  n_checkpoint = 0`. Mirror with a slow-PCIe trace where the
  searcher should still pick `n_checkpoint > 0`.

### 5.3 Comparison test (the science)

* **`test_offload_vs_ckpt_memory_throughput`** — same model, two
  ProTrain configs:
  * `n_offload=N, n_checkpoint=0` (the new path)
  * `n_offload=0, n_checkpoint=N` (the existing path)
  Both with the same `n_persist`, `n_buffer`. Runs 4 iterations
  each, collects throughput + GPU peak. Asserts:
  * GPU peak under OFFLOAD is **higher** than CKPT (we keep
    activations resident) by an amount within ±20% of the
    cost model's prediction
  * throughput under OFFLOAD is **higher** than CKPT (we don't
    pay recompute) by an amount within ±20% of the cost model's
    prediction
  This documents the trade — and gives the searcher's calibration
  a regression target.

### 5.4 Existing tests to audit

`block_map_runtime_admissible` is called from at least:

* `search/exhaustive.py::search` (the validator)
* possibly `runtime/hooks.py::install` (defensive double-check)

Every caller must be updated to the new signature (no signature
change — the function is keyed on `(layout, block_map, n_persist)` —
but the *meaning* changes). Confirm with `grep -rn` during M1.

---

## 6. Risks and open questions

### 6.1 Storage-pointer aliasing

The pack hook identifies "is this saved tensor a view of a
chunk-managed param?" by `untyped_storage().data_ptr()` lookup.
Risk: PyTorch may collapse storage or change pointer identity in
edge cases (inplace ops, autocast, ZeRO-3 staging buffers we
introduce internally). Mitigation: at attach time, validate that
every parameter currently owned by the chunk shares the chunk
buffer's storage; assert in debug mode. Add a unit test that
asserts a wrapped block sees its first-iteration save/unpack cycle
return the SAME storage pointer the gather hook recorded. If this
ever fails in production, the fail-open path is to fall through to
standard `save_for_backward` semantics — correct but slow (chunk
won't get released after forward).

### 6.2 Autograd graph consistency

Saved-tensors-hooks operate on the saved-tensor table per Node, not
per param. A param tensor might be saved by *multiple* downstream
Nodes (e.g. linear weight saved by both matmul and a fused activation
gradient). The unpack hook is called once per Node per saved tensor.
Each call re-views the chunk buffer at the same offset/shape, so the
two views see the same bytes — but autograd considers them distinct
tensors. Risk: a Node that compares saved tensor identity via `is`
will see the views as different. Mitigation: PyTorch's autograd
internals do not rely on identity checks; verified once for
`SwappedBlock` (which has the same property). Add a regression test
that explicitly exercises a multi-save pattern.

### 6.3 Multi-rank gather timing under ZeRO-3

The unpack hook calls `chunk_manager.gather(chunk_id)`. In Mode-C
that triggers `all_gather_into_tensor` collectives — collective
operations require every rank to participate. Risk: if rank A's
unpack hook fires before rank B reaches the corresponding backward
block, rank A blocks waiting for the collective; deadlock if rank
B's autograd hits a different block first. Mitigation:
backward order is deterministic across ranks for the same model
and the same iteration (autograd processes the same DAG). If we
rely on `pre_block_backward` to issue the gather (which it
already does as the chunk manager's primary entry point), every
rank issues gather at the same wall-clock-ish point. The unpack
hook becomes a no-op if the chunk is already resident — i.e., it
hits the fast path. The risk reduces to "what happens if
pre_block_backward gets skipped on one rank but not another?" —
this is already a correctness invariant for the existing CKPT
path; OFFLOAD inherits the same safety.

### 6.4 Optimizer wrapper interaction

`chunk/optim.py` (DeepSpeedCPUAdam adapter) reads each
non-persistent param's `.data` via the pinned-CPU shard pointer set
during `materialize_offload`. The CPU step is kicked off in the
post-grad hook (per-param) and runs asynchronously. Risk: under
OFFLOAD the post-grad hook fires **after** the saved-param unpack
hook has already re-gathered the chunk for backward. If the unpack
hook re-binds `param.data` to the GPU pool buffer, and then the CPU
adam tries to read `param.data`, it sees a CUDA pointer and trips
the `"CPUAdam param is on cuda:N"` assertion (already documented in
`offload`'s docstring at chunk/manager.py:1666-1683).

Mitigation: the unpack hook does NOT need to rebind `param.data` —
it returns a view directly to the autograd engine, and `param.data`
stays bound to whatever the offload path left it (pinned CPU
during the CPU adam step, empty-GPU placeholder afterward). The
gradient kernels will read the unpack-returned view, NOT
`param.data`. We will add an assertion in the unpack hook that it
does not touch `param.data` — defensive, the failure mode is silent
otherwise. This is the highest-risk integration corner.

### 6.5 `param.data` rebinding cycles

Today's path:
- `gather` rebinds `param.data` to a GPU pool view.
- `offload` rebinds `param.data` to an empty-GPU placeholder (or
  leaves it on CPU if the grad-hook just touched it — see chunk/
  manager.py:1666-1683).

OFFLOAD adds a new path: the unpack hook re-gathers DURING backward.
After the unpack hook has done its work, what does `param.data`
point at? Decision: the unpack hook does NOT rebind `param.data`. It
only returns a view to autograd. After
`reduce_grads_and_offload` runs at end-of-block-backward,
`param.data` returns to the same null-placeholder state it was in
between forward-end and backward-start. The unpack-returned view
keeps the chunk buffer alive via the BackwardHandle refcount —
NOT via `param.data`. This decouples the "which tensor does
backward use" question from the "what does param.data look like
between phases" question.

### 6.6 SWAP-on-non-persistent

The combination "block uses SWAP wrapper AND its chunks are
non-persistent" is left **out of scope** for v1 of this design.
Reasons:
* The SWAP wrapper offloads activations to CPU; OFFLOAD-equivalent
  param handling on top would create two independent CPU-pinned
  paths in the same block, multiplying complexity.
* The use case is narrow (only really matters when both activations
  AND params are too big to keep resident, which on the 3090 target
  rig usually means "use a smaller model").

If a future workstream wants this combination, it will compose the
SWAP saved-tensors-hooks context with the OFFLOAD context (nested
contexts on torch.autograd.graph stack). The hooks compose
cleanly because each context only handles tensors it recognizes;
unrecognized tensors fall through to the outer context.

### 6.7 Effort estimate

* **Multi-day, not multi-week.** The riskiest piece is the
  storage-pointer aliasing layer (§6.1) — call it 2 days for a
  competent agent. The rest (enum + validator + scheduler hook +
  cost model + tests) is mechanical, ~1 day each.
* **Total best-case: ~5 days end-to-end** (M1–M5 below).
* **Worst case ~10 days** if §6.1 turns out to need a deeper
  PyTorch-internals workaround (e.g., autograd FunctionCtx
  introspection).

---

## 7. Implementation roadmap

### M1 — types + validator (small, ~1 day) — SHIPPED (`8264f773`)

Add `BlockMode.OFFLOAD = "offload"` to `types.py`. Update
`block/strategy.py` re-exports. Update
`search/exhaustive.py::block_map_runtime_admissible` to the new rule.
Update `block/layout_rules.py::assign_modes` to take `n_offload` and
honor it under unopt-late placement. Unit tests:
* `test_admissibility_under_offload_rule`
* `test_assign_modes_with_offload`

Exit criteria: tests pass; existing test suite green (no behavior
change yet because no producer sets `n_offload>0`).

### M2 — runtime hook (medium, ~3 days) — SHIPPED (`8264f773`)

Implement `block/offload.py::OffloadedBlock`:

* `__init__` mirrors `SwappedBlock`.
* `attach_runtime(chunk_manager, scheduler)`.
* `forward()` installs `saved_tensors_hooks(pack, unpack)` for the
  duration of the wrapped block's forward.
* `pack_param_only` resolves storage-ptr → chunk_id; replaces the
  saved tensor with a `_ParamHandle` metadata object.
* `unpack_param_only` calls `chunk_manager.gather_for_backward`,
  returns a view + holds the `BackwardHandle` on the view's
  lifetime.

Implement `chunk/manager.py` extensions:
* `chunk_id_for_storage_ptr(ptr)` — O(1) lookup against a dict
  populated at gather time.
* `gather_for_backward(chunk_id) -> BackwardHandle` — gather +
  refcount bump.
* Hook the refcount into `reduce_grads_and_offload` so it defers
  the actual offload until refcount=0.

Update `block/dispatcher.py::wrap_block` to emit `OffloadedBlock`
for `BlockMode.OFFLOAD`.

Unit tests:
* `test_offloaded_block_save_unsave_roundtrip`
* `test_chunk_manager_backward_handle_lifecycle`

Exit criteria: unit tests pass; manual smoke (a tiny 2-block model)
trains a few iterations and matches a reference forward+backward.

### M3 — scheduler integration (medium, ~3 days) — SHIPPED (`a1ab8aff`)

Wire `OffloadedBlock` into `runtime/hooks.py::install`. Update
`runtime/scheduler.py::pre_block_backward` to be aware of
OFFLOAD-mode blocks (gathers earlier than CKPT to give the unpack
hook a fast-path hit instead of forcing a synchronous gather inside
backward). Update `Scheduler.drain` to flush any deferred offloads.

Smoke test: `test_protrain_2gpu_mistral_modec_smoke` should now
pass with the OFFLOAD config.

### M4 — cost model + searcher (small, ~2 days) — SHIPPED (`ea20710a`)

Add the `T_bwd_gather` term to `cost/runtime.py`. Add the OFFLOAD
backward-bump term to `cost/memory.py::estimate_peak`. Extend
`search/exhaustive.py` to enumerate `n_offload`. Tests:
* `test_search_picks_offload_when_advantageous`
* `test_estimate_peak_offload_block_bump`
* `test_estimate_runtime_offload_gather_term`

Calibrate against measured throughput from the M3 smoke test;
adjust the hot-cap path in `cost/memory.py` if needed (per-block
peaks for OFFLOAD differ from CKPT).

### M5 — test enablement (small, ~1 day) — SHIPPED (`c7c155f7`)

Re-enabled the three previously-skipped slow tests:
* `test_protrain_4gpu_zero3_sharding` — asserts no recompute (new
  assertion).
* `test_protrain_2gpu_mistral_modec_smoke` — already passing from M3.
* `test_modec_vs_deepspeed_stage3_4gpu` — new comparison test.

Exit criteria met: all three pass on the 4× 3090 target rig (per
`MEMORY.md::hardware_protrain_targets`).

---

## 8. Deferral / kill criteria

This design should NOT be implemented if any of the following hold:

1. **Paper-clarification disagreement.** If during the §2 paper
   review (a re-read by the implementation agent before writing
   code) we find the paper *explicitly* requires CKPT for offloaded
   chunks, defer to a paper-clarification follow-up. Author-contact
   may be warranted before re-attempting Option B.

2. **PyTorch storage-pointer fragility.** If the §6.1 storage-ptr
   identification fails in unit testing (e.g. autograd internal
   collapses chunk-buffer storage with an unrelated tensor's
   storage), back out and consider Option A (extend NONE
   semantics), which can use weakrefs on `nn.Parameter` instances
   directly — those don't hit the storage-collapse path.

3. **DeepSpeed Stage-3 baseline shifts.** If the v1 acceptance
   criteria (per `MEMORY.md::feedback_paper_alignment`) tighten the
   "apples-to-apples DeepSpeed comparison" requirement to the point
   where Option B alone isn't sufficient (e.g. reviewer wants
   ZeRO-Infinity NVMe paths), defer until the broader scope is
   re-prioritized.

4. **Searcher-driven CKPT remains optimal in practice.** If the
   cost-model M4 work shows that on the 3090 / PCIe Gen3 target
   rig, OFFLOAD never wins against CKPT for realistic 7B-class
   models (because PCIe is too slow and per-block compute is too
   cheap), the throughput motivation evaporates. Option B would
   then be **scientific-completeness only** (the apples-to-apples
   comparison) and reviewers may legitimately defer. Plan the M4
   calibration check explicitly as a go/no-go gate before M5.

5. **Runtime correctness regressions in M2 / M3.** If the
   per-storage-ptr book-keeping introduces correctness bugs in
   non-OFFLOAD paths (e.g. the new dict in ChunkManager corrupts
   the persistent path), revert and re-architect with an explicit
   `OffloadedChunkManager` subclass instead of in-place mutation.

A go-decision requires (1) paper re-confirmation, (2) M4 calibration
showing at least 1.2× throughput win on the 3090 rig at 3B+, and
(3) reviewer sign-off on this doc.

---

## 9. Glossary

* **Persistent chunk** — chunk whose params live on GPU for the entire
  iteration; `chunk_id < n_persist` by index assignment.
* **Non-persistent chunk** — chunk whose params live on CPU between
  block visits and are gathered to GPU on demand.
* **Block mode** — per-block activation strategy
  (`NONE | CKPT | SWAP | OFFLOAD`).
* **OFFLOAD** (this doc) — new mode: param chunks may be non-persistent,
  but activations stay on GPU; backward re-gathers chunks via
  saved-tensors-hooks instead of via recompute.
* **Saved-tensors-hooks** — `torch.autograd.graph.saved_tensors_hooks`
  context manager; a (pack, unpack) pair that intercepts every
  saved tensor inside the context.
* **`block_map_runtime_admissible`** — current validator in
  `search/exhaustive.py` that enforces the v1 "non-persistent ⇒ CKPT"
  rule. Updated by Option B to allow OFFLOAD too.
* **Backward handle** — RAII helper introduced by Option B; bumps a
  refcount on a chunk buffer slot to keep it alive across the
  backward window.
* **Mode-C** — ProTrain ZeRO-3 sharded CPU-offload composition
  (`zero3_shard=True`). The composition mode that benefits most
  from Option B because it's the apples-to-apples target for
  DeepSpeed Stage-3.
