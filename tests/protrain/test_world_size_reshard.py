"""Live world-size reshard test (Mode-B replicated, 4 ranks → 2 ranks).

ProTrain's Mode-B replicated checkpoint format claims world-size-change
support — the on-disk state is rank-independent, so a save with
``world_size=4`` should load cleanly into a fresh ``world_size=2`` run.
``test_load_accepts_world_size_change_for_replicated`` only fakes the
metadata (mutates ``protrain_world_size`` in a 1-rank test) — it does
not exercise the live cross-process path. This test does:

1. Spawn 4 ranks via ``mp.spawn`` on GPUs 1, 2, 4, 5 (the free 24GB
   pool from MEMORY.md). Each rank builds an identical tiny model +
   ChunkManager + ``_ProTrainOptimizer``, runs one fwd+bwd+step so
   the inner Adam state is non-trivial, then saves the checkpoint.
   Rank-0 writes; rank-1..3 reach the post-callback barrier and exit.
2. Tear down the 4-rank world (every worker calls
   ``destroy_process_group``; ``mp.spawn`` joins).
3. Spawn 2 ranks on GPUs 1, 2 (subset of the same pool). Each rank
   builds the same tiny model fresh, calls
   ``_load_protrain_optim_dir`` against the saved directory, runs one
   step, and asserts the resulting loss is finite. The pre-step
   inner state must match what rank-0 wrote at save time (proving the
   load actually reads files, not silently no-ops).

Mode-B is the test target rather than Mode-C because Mode-C
explicitly hard-errors on ``saved_world != current_world``
(checkpoint.py:915). Cross-world-size reshard for Mode-C requires a
re-shard step that is documented as out-of-scope for Phase 2 (see
CHECKPOINT_DESIGN_PHASE2.md §4.1). The Mode-B path is the surface
that actually advertises world-size-change support today.

Slow-marked, single test, < 5 min wall on the rig per the handoff
budget.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from axolotl.integrations.protrain.api.checkpoint import (  # noqa: E402
    CPU_OPTIM_DIRNAME,
    DEFAULT_SAVE_MAX_BYTES,
    METADATA_FILENAME,
    PROTRAIN_OPTIM_DIRNAME,
    SAVE_MODE_REPLICATED,
    SAVE_MODE_SHARDED,
    _load_protrain_optim_dir,
    _save_protrain_optim_dir,
)

# Reuse the helper machinery from the main optimizer-checkpoint test —
# mp.spawn workers can re-import the test module fine because pytest's
# rootdir is on sys.path during test collection.
from tests.protrain.test_optimizer_checkpoint import (  # noqa: E402
    _build_chunk_manager,
    _build_optim_pair,
    _force_identical_inner_state,
    _teardown_mgr,
    _tiny_model,
)

# ---- worker bodies ---------------------------------------------------------


def _save_worker(rank: int, world_size: int, tmpdir: str) -> None:
    """One rank in the 4-rank save phase.

    Rank-0 writes; all ranks must reach the post-save barrier so the
    parent test can confirm liveness via ``rank{N}.done``. Inner state
    is zeroed before save so the load-phase post-load comparison has a
    deterministic target (eliminates DDP-vs-non-DDP / CPU-adam threading
    noise; this test is about the save+load mechanism, not about
    DDP determinism).
    """
    import torch
    import torch.distributed as dist

    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("worker: CUDA not available")

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-save",
            rank=rank,
            world_size=world_size,
        )

        torch.manual_seed(0)
        model = _tiny_model().to("cuda")
        mgr, host = _build_chunk_manager(model, n_persist=1, S_chunk=64 * 1024)
        mgr.materialize_offload()
        _, _, optim = _build_optim_pair(model, mgr)

        # One fwd+bwd+step so the inner state has real exp_avg / exp_avg_sq
        # entries (otherwise the gate would skip with a 0-byte estimate).
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(123)
        x = torch.randn(2, model.embed.in_features, generator=cpu_gen).to("cuda")
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        optim.zero_grad()
        out = model(x)
        out.sum().backward()
        optim.step()

        # Force byte-identical state across ranks. Mode-B's contract is
        # that DDP keeps the inner state replicated; we don't have DDP
        # in this test (it's a pure save/load mechanism check), so we
        # zero the state to skip past that question and focus the load
        # phase on file plumbing.
        _force_identical_inner_state(optim)

        save_dir = os.path.join(tmpdir, "save_root")
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        dist.barrier()

        # _save_protrain_optim_dir is collective (lockstep broadcast in its
        # finally — see api/checkpoint.py:_broadcast_status_or_raise); every
        # rank must call it. Only rank-0 actually writes (gated internally),
        # but every rank must reach the broadcast so a rank-0 write failure
        # raises in lockstep instead of deadlocking the trailing barrier.
        wrote = _save_protrain_optim_dir(
            optim,
            save_dir,
            step=1,
            save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
            rank=rank,
            world_size=world_size,
        )
        if not wrote:
            raise RuntimeError(f"rank {rank}: save returned False")
        dist.barrier()

        with open(os.path.join(tmpdir, f"save_rank{rank}.done"), "w") as f:
            f.write("ok")

        _teardown_mgr(mgr, optim)
        host.close()
        del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(os.path.join(tmpdir, f"save_rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _load_worker(rank: int, world_size: int, tmpdir: str) -> None:
    """One rank in the 2-rank load phase.

    Builds a fresh model + manager + optim (same arch, same seed), then
    loads from the directory rank-0 wrote during the 4-rank save phase.

    Acceptance:
      * ``_load_protrain_optim_dir`` returns True (loaded the dir).
      * Loaded inner state == zero (matches what was forced+saved
        during the save phase). This proves the load actually read the
        on-disk bytes — without a load, the post-step state would be
        the result of one freshly-randomised step (non-zero with high
        probability).
      * One additional optimizer step lands without exception and
        produces a finite loss — proves the resharded state is
        consistent with the rebuilt chunk geometry.
    """
    import torch
    import torch.distributed as dist

    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("worker: CUDA not available")

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-load",
            rank=rank,
            world_size=world_size,
        )

        torch.manual_seed(0)  # identical init across ranks → same arch hash
        model = _tiny_model().to("cuda")
        mgr, host = _build_chunk_manager(model, n_persist=1, S_chunk=64 * 1024)
        mgr.materialize_offload()
        _, _, optim = _build_optim_pair(model, mgr)

        # Take a non-zero step BEFORE the load so that "post-load state ==
        # zero" is a strong signal that the load happened. Without this,
        # a no-op load would leave the freshly-built (zero) inner state
        # and the assertion would falsely pass.
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(rank + 7)  # different per rank for noise
        x = torch.randn(2, model.embed.in_features, generator=cpu_gen).to("cuda")
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        optim.zero_grad()
        out = model(x)
        out.sum().backward()
        optim.step()

        # Snapshot inner state pre-load — every state tensor should be
        # non-zero now (one Adam step on a random batch).
        non_zero_pre_load = False
        if optim._gpu_optim is not None:
            for s in optim._gpu_optim._optim.state.values():
                for v in s.values():
                    if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                        non_zero_pre_load = True
        if optim._cpu_optim is not None:
            for inner in optim._cpu_optim._optims.values():
                for s in inner.state.values():
                    for v in s.values():
                        if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                            non_zero_pre_load = True
        if not non_zero_pre_load:
            raise RuntimeError(
                "load worker: pre-load inner state was already zero — "
                "the post-load==zero check below would be ambiguous"
            )

        save_dir = os.path.join(tmpdir, "save_root")
        loaded = _load_protrain_optim_dir(optim, save_dir)
        if not loaded:
            raise RuntimeError(
                f"rank {rank}: _load_protrain_optim_dir returned False — "
                f"checkpoint dir {save_dir} not found?"
            )

        # Acceptance: post-load state must match the saved (zero) state.
        post_load_all_zero = True
        if optim._gpu_optim is not None:
            for s in optim._gpu_optim._optim.state.values():
                for v in s.values():
                    if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                        post_load_all_zero = False
        if optim._cpu_optim is not None:
            for inner in optim._cpu_optim._optims.values():
                for s in inner.state.values():
                    for v in s.values():
                        if isinstance(v, torch.Tensor) and v.abs().sum() > 0:
                            post_load_all_zero = False
        if not post_load_all_zero:
            raise RuntimeError(
                f"rank {rank}: post-load inner state has non-zero entries — "
                "load did not overwrite the pre-load step's state, so "
                "the resharded state is not actually being applied"
            )

        # Acceptance: one more step on the resharded state must produce
        # a finite loss without exception. Re-gather every offloaded
        # chunk first — after the pre-load step, ``param.data`` for
        # non-persistent chunks is back to its empty placeholder, so a
        # forward without gather would crash on a (numel=0) weight.
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        cpu_gen2 = torch.Generator(device="cpu")
        cpu_gen2.manual_seed(rank + 17)
        x2 = torch.randn(2, model.embed.in_features, generator=cpu_gen2).to("cuda")
        optim.zero_grad()
        out2 = model(x2)
        loss2 = out2.sum()
        if not bool(torch.isfinite(loss2).item()):
            raise RuntimeError(
                f"rank {rank}: post-load step produced non-finite loss "
                f"{float(loss2.detach())}"
            )
        loss2.backward()
        optim.step()

        with open(os.path.join(tmpdir, f"load_rank{rank}.done"), "w") as f:
            f.write(f"loss2={float(loss2.detach())}\n")

        _teardown_mgr(mgr, optim)
        host.close()
        del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(os.path.join(tmpdir, f"load_rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


# ---- driver test -----------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
def test_replicated_world_size_reshard_4_to_2(tmp_path):
    """Live save N=4 / load N=2 replicated reshard end-to-end.

    Save phase uses 4 mp.spawn workers (one per visible GPU); load
    phase uses 2 (subset of the same physical pool). Both phases
    rendezvous via gloo on a file:// store rooted in tmp_path so the
    test does not need MASTER_PORT plumbing.

    The test is the live counterpart to
    ``test_load_accepts_world_size_change_for_replicated`` (which only
    mutates metadata in a single-process test). If Mode-B replicated
    state ever stops being world-size-independent, this test catches it.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    n_visible = torch.cuda.device_count()
    if n_visible < 4:
        pytest.skip(
            f"world-size-reshard test needs >= 4 visible GPUs (got {n_visible})"
        )

    import torch.multiprocessing as mp

    # ---- Phase 1: save with world_size=4 ----------------------------
    save_world = 4
    mp.spawn(
        _save_worker,
        args=(save_world, str(tmp_path)),
        nprocs=save_world,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"save-phase worker errors:\n{bodies}")
    for r in range(save_world):
        assert (tmp_path / f"save_rank{r}.done").is_file(), (
            f"save rank {r} did not reach post-save sentinel"
        )

    # Verify the saved metadata records world_size=4 (Mode-B) so the
    # load phase has something meaningful to reshard from.
    proot = tmp_path / "save_root" / PROTRAIN_OPTIM_DIRNAME
    assert proot.is_dir(), f"save root {proot} missing post-spawn"
    meta = json.loads((proot / METADATA_FILENAME).read_text())
    assert meta["protrain_save_mode"] == SAVE_MODE_REPLICATED, (
        f"expected replicated save_mode (Mode-B), got {meta['protrain_save_mode']!r}"
    )
    assert meta["protrain_world_size"] == save_world, (
        f"expected protrain_world_size={save_world}, got {meta['protrain_world_size']}"
    )

    # ---- Phase 2: load with world_size=2 (different from save) ------
    load_world = 2
    mp.spawn(
        _load_worker,
        args=(load_world, str(tmp_path)),
        nprocs=load_world,
        join=True,
    )
    err_files = sorted(tmp_path.glob("load_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"load-phase worker errors:\n{bodies}")
    for r in range(load_world):
        assert (tmp_path / f"load_rank{r}.done").is_file(), (
            f"load rank {r} did not reach post-load sentinel"
        )


# ===========================================================================
# Mode-C (ZeRO-3 sharded) — offline reshard tool round-trip
# ===========================================================================
#
# Mode-C explicitly hard-errors on ``saved_world != current_world``
# (CHECKPOINT_DESIGN_PHASE2.md §4.1, api/checkpoint.py:_load_protrain_optim_dir
# Mode-C branch). The offline tool ``scripts/protrain/reshard_optim.py``
# converts the saved per-rank shards from N1 to N2 so the loader sees
# what looks like a natively-saved-at-N2 directory. This test cell
# exercises the round-trip end-to-end:
#
# Phase 1 (N=4 ranks, GPUs 1,2,4,5): build the sharded chunk_manager
#     mixed-dtype model, take one fwd+bwd+step, force a deterministic
#     pattern in the inner state (so the equivalence check below has a
#     stable target), save Mode-C to ``save_n4/``.
#
# Phase 1b (N=2 ranks, GPUs 1,2): same pattern but at world_size=2 so
#     we have a "natively-N=2" reference save in ``save_n2/`` against
#     which to verify semantic equivalence.
#
# Phase 2 (offline, no GPUs): invoke ``scripts/protrain/reshard_optim.py``
#     to reshard ``save_n4/`` → ``save_n4_resharded/``.
#
# Phase 3 (N=2 ranks, GPUs 1,2): two parallel paths run in the SAME
#     mp.spawn worker, one after the other:
#       (a) load ``save_n4_resharded/``, take one optimizer step on a
#           fixed deterministic batch, snapshot the post-step weights.
#       (b) load ``save_n2/`` (the natively-N=2 reference), take the
#           same step, snapshot the post-step weights.
#     Acceptance: (a) and (b) match within float-tolerance — the
#     resharded state is semantically equivalent to natively-N=2 state.
#
# The "build the sharded chunk_manager" helpers (mixed-dtype model
# + materialize_offload + optim pair) are reused from
# tests.protrain.test_optimizer_checkpoint.

from tests.protrain.test_optimizer_checkpoint import (  # noqa: E402
    _build_sharded_chunk_manager_mixed_dtype,
)


def _force_pattern_inner_state(optim) -> None:
    """Fill every inner-state tensor with a deterministic pattern.

    The pattern depends only on the (region_idx, state_key) and the
    flat element index within the rank's shard slice. This lets the
    test set up the SAME logical full-padded-region content at both
    world_size=4 and world_size=2: each rank's slice of the global
    pattern is determined by the (rank, world_size, region_idx)
    identity, derived from the offset in the global flat array.

    Specifically: for region ``i``, state key ``k``, the global
    flat tensor is ``[ (i+1) * (k_idx+1) * (g_idx + 1) ]`` for
    ``g_idx in [0, region_bytes_padded / elem_size)``. The trailing
    pad positions are zeroed. Each rank's shard is its
    ``[rank * shard_numel, (rank+1) * shard_numel)`` slice.

    Inputs use float-dtype tensors so the cast doesn't truncate.
    """
    import torch as _torch

    if optim._cpu_optim is None:
        return

    chunk_manager = optim._chunk_manager
    rank = int(getattr(chunk_manager, "rank", 0))

    state_key_idx = {"exp_avg": 0, "exp_avg_sq": 1}

    for cid, inner in optim._cpu_optim._optims.items():
        shard_state = chunk_manager._chunk_shards.get(cid)
        if shard_state is None:
            continue
        regions = shard_state.regions
        for region_idx, region in enumerate(regions):
            inner_state = inner.state.get(region.shard_param)
            if inner_state is None:
                continue
            elem_size = region.element_size
            region_bytes = region.region_bytes
            region_bytes_padded = region.region_bytes_padded
            shard_bytes = region.shard_bytes

            valid_numel = region_bytes // elem_size
            padded_numel = region_bytes_padded // elem_size
            shard_numel = shard_bytes // elem_size

            # Build the global flat pattern (length padded_numel),
            # zero-pad the trailing [valid_numel:padded_numel) slice.
            for k, k_idx in state_key_idx.items():
                v = inner_state.get(k)
                if not isinstance(v, _torch.Tensor):
                    continue
                base = float((region_idx + 1) * (k_idx + 1))
                global_flat = _torch.zeros(padded_numel, dtype=v.dtype)
                if valid_numel > 0:
                    indices = _torch.arange(valid_numel, dtype=_torch.float64)
                    global_flat[:valid_numel] = (base * (indices + 1.0)).to(v.dtype)
                # This rank's slice.
                slice_ = global_flat[rank * shard_numel : (rank + 1) * shard_numel]
                # In-place copy preserves the inner optimizer's pointer
                # identity (DeepSpeedCPUAdam tracks tensors by id).
                v.copy_(slice_)


def _hash_inner_state(optim) -> str:
    """Stable cross-process hash over the rank's inner CPU optim state."""
    import hashlib

    import torch as _torch

    h = hashlib.sha256()
    if optim._cpu_optim is None:
        return h.hexdigest()
    for cid in sorted(optim._cpu_optim._optims):
        inner = optim._cpu_optim._optims[cid]
        h.update(f"chunk:{int(cid)}:".encode("utf-8"))
        for region_idx, (_param, st) in enumerate(inner.state.items()):
            h.update(f"region:{region_idx}:".encode("utf-8"))
            for k in sorted(st.keys()):
                v = st[k]
                if isinstance(v, _torch.Tensor):
                    h.update(f"{k}:".encode("utf-8"))
                    h.update(str(v.dtype).encode("utf-8"))
                    h.update(b":")
                    if v.numel() > 0:
                        h.update(
                            v.detach()
                            .contiguous()
                            .cpu()
                            .flatten()
                            .view(_torch.uint8)
                            .numpy()
                            .tobytes()
                        )
    return h.hexdigest()


def _save_worker_modec(rank: int, world_size: int, tmpdir: str, tag: str) -> None:
    """One rank in the Mode-C save phase (used for both N=4 and N=2 saves).

    Builds the mixed-dtype sharded chunk_manager + optim, takes one
    fwd+bwd+step, FORCES a deterministic pattern via
    :func:`_force_pattern_inner_state`, then writes its per-rank
    shard files via the Mode-C save path.
    """
    import os

    import torch
    import torch.distributed as dist

    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    from axolotl.integrations.protrain.api.checkpoint import (
        DEFAULT_SAVE_MAX_BYTES as _DEFAULT_SAVE_MAX_BYTES,
        _save_protrain_optim_dir as _save_dir,
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("worker: CUDA not available")

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-{tag}",
            rank=rank,
            world_size=world_size,
        )

        model, mgr, host = _build_sharded_chunk_manager_mixed_dtype(rank, world_size)
        mgr.materialize_offload()
        _, _, optim = _build_optim_pair(model, mgr)

        # One fwd+bwd+step so the inner state has real exp_avg /
        # exp_avg_sq tensors.
        #
        # The Mode-C sharded path defers the CPU Adam step to
        # ``ChunkManager.reduce_grads_and_offload`` (chunk-level
        # reduce-scatter, then ``cpu_optim.step_async``). In real
        # training the block-level model wrapper triggers
        # reduce_grads_and_offload for each block — without that
        # wrapper, our hand-built test has to trigger it manually
        # after backward so the per-chunk CPU adam actually runs and
        # populates ``inner.state``.
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(123)
        x = torch.randn(2, 32, generator=cpu_gen).to("cuda").to(torch.float16)
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        # set_to_none=False keeps the per-region shard_param.grad
        # tensor alive — the reduce_scatter path copies into it
        # in-place, so a None grad would crash with AttributeError
        # in ChunkManager._reduce_scatter_and_offload_shard.
        optim.zero_grad(set_to_none=False)
        out = model.h[0].proj(x)
        out = model.h[0].norm(out.to(torch.float32))
        out.sum().backward()
        # Manually drive each non-persistent chunk's reduce-then-CPU-step
        # since the wrapper-level scheduler isn't installed in this
        # hand-built setup.
        for cid in list(mgr._non_persistent_ids):
            mgr.reduce_grads_and_offload(cid)
        optim.step()
        # Drain pending async adam futures so .state is populated before
        # the pattern-forcing step below indexes by region.
        mgr.wait_cpu_optim_all()

        # Force the deterministic cross-world pattern. After this every
        # rank's inner state is its slice of an identical "global" full-
        # padded-region tensor — so saving at N=4 and at N=2 produces
        # the same logical state, just sliced differently.
        _force_pattern_inner_state(optim)

        save_dir = os.path.join(tmpdir, f"save_{tag}")
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        dist.barrier()

        wrote = _save_dir(
            optim,
            save_dir,
            step=1,
            save_max_bytes=_DEFAULT_SAVE_MAX_BYTES,
            rank=rank,
            world_size=world_size,
        )
        if not wrote:
            raise RuntimeError(f"rank {rank}: save returned False")
        dist.barrier()

        with open(os.path.join(tmpdir, f"save_modec_{tag}_rank{rank}.done"), "w") as f:
            f.write("ok")

        # Snapshot the rank's inner-state hash for forensic comparison.
        with open(os.path.join(tmpdir, f"save_modec_{tag}_rank{rank}.hash"), "w") as f:
            f.write(_hash_inner_state(optim))

        try:
            mgr.restore_to_gpu()
        except Exception:  # noqa: BLE001
            pass
        if optim._cpu_optim is not None:
            try:
                optim._cpu_optim.shutdown()
            except Exception:  # noqa: BLE001
                pass
        host.close()
        del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(os.path.join(tmpdir, f"save_modec_{tag}_rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _load_worker_modec(
    rank: int,
    world_size: int,
    tmpdir: str,
    save_subdir: str,
    sentinel_tag: str,
    allow_online_reshard: bool = False,
) -> None:
    """One rank in a Mode-C load phase. Builds fresh model + manager,
    loads from ``tmpdir/save_subdir/protrain_optim``, takes one
    optimizer step on a deterministic fixed batch, writes a hash of
    the post-step inner-state and post-step model parameters to a
    sentinel file.

    ``allow_online_reshard`` is forwarded into
    :func:`_load_protrain_optim_dir`. When True the loader handles
    cross-world-size resume internally (rank-0 reshards into a temp
    dir; all ranks load from there). When False (the default) the
    legacy behaviour applies: world-size mismatch is a hard error.
    """
    import os

    import torch
    import torch.distributed as dist

    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    from axolotl.integrations.protrain.api.checkpoint import (
        _load_protrain_optim_dir as _load_dir,
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("worker: CUDA not available")

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-load-{sentinel_tag}",
            rank=rank,
            world_size=world_size,
        )

        model, mgr, host = _build_sharded_chunk_manager_mixed_dtype(rank, world_size)
        mgr.materialize_offload()
        _, _, optim = _build_optim_pair(model, mgr)

        # The inner state is empty pre-load. ``_load_protrain_optim_dir``
        # must overwrite it with the saved (or resharded) bytes.
        save_dir = os.path.join(tmpdir, save_subdir)
        # _load_protrain_optim_dir expects a "checkpoint_dir" that
        # contains a ``protrain_optim/`` child. Our save_dir is
        # exactly such a parent (see _save_protrain_optim_dir's
        # ``target = os.path.join(output_dir, PROTRAIN_OPTIM_DIRNAME)``).
        loaded = _load_dir(optim, save_dir, allow_online_reshard=allow_online_reshard)
        if not loaded:
            raise RuntimeError(
                f"rank {rank}: _load_protrain_optim_dir({save_dir!r}) returned False"
            )

        post_load_hash = _hash_inner_state(optim)

        # Fixed deterministic batch — identical across the two phase-3
        # paths (resharded vs natively-N=2) so the post-step state is
        # comparable.
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(999)
        x = torch.randn(2, 32, generator=cpu_gen).to("cuda").to(torch.float16)
        for cid in list(mgr._non_persistent_ids):
            mgr.gather(cid)
        # set_to_none=False keeps the per-region shard_param.grad
        # tensor alive — the reduce_scatter path copies into it
        # in-place, so a None grad would crash with AttributeError
        # in ChunkManager._reduce_scatter_and_offload_shard.
        optim.zero_grad(set_to_none=False)
        out = model.h[0].proj(x)
        out = model.h[0].norm(out.to(torch.float32))
        loss = out.sum()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"rank {rank}: post-load loss is non-finite")
        loss.backward()
        # Manually fire reduce_grads_and_offload (see save worker note —
        # without the wrapper-level scheduler, the CPU adam step needs
        # to be triggered explicitly so .state actually updates).
        for cid in list(mgr._non_persistent_ids):
            mgr.reduce_grads_and_offload(cid)
        optim.step()

        # Drain the async CPU adam queue so we hash a consistent state.
        mgr.wait_cpu_optim_all()

        post_step_hash = _hash_inner_state(optim)

        # Hash post-step model parameters (after restore to GPU). The
        # restore copies sharded bytes back into rank-0 view via
        # all_gather; every rank then sees the same full param values,
        # so we hash once on rank-0.
        # NOTE: doing restore_to_gpu would interfere with subsequent
        # mp.spawn invocations in this process; instead, hash the
        # params' .data view directly (post-step Adam already wrote
        # the new values into the CPU shard buffers, and the
        # ``materialize_offload`` indirection doesn't affect what's on
        # disk in cpu_shard_bytes).
        # Hash the rank's CPU shard bytes for every region.
        import hashlib

        h = hashlib.sha256()
        for cid in sorted(mgr._chunk_shards):
            shard_state = mgr._chunk_shards[cid]
            for region_idx, region in enumerate(shard_state.regions):
                h.update(f"chunk:{int(cid)}:region:{region_idx}:".encode("utf-8"))
                h.update(region.cpu_shard_bytes.detach().cpu().numpy().tobytes())
        param_hash = h.hexdigest()

        with open(
            os.path.join(tmpdir, f"load_modec_{sentinel_tag}_rank{rank}.done"), "w"
        ) as f:
            f.write(f"loss={float(loss.detach())}\n")
        with open(
            os.path.join(tmpdir, f"load_modec_{sentinel_tag}_rank{rank}.hash"), "w"
        ) as f:
            # post_load_hash:post_step_hash:param_hash
            f.write(f"{post_load_hash}:{post_step_hash}:{param_hash}\n")

        try:
            mgr.restore_to_gpu()
        except Exception:  # noqa: BLE001
            pass
        if optim._cpu_optim is not None:
            try:
                optim._cpu_optim.shutdown()
            except Exception:  # noqa: BLE001
                pass
        host.close()
        del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(
            os.path.join(tmpdir, f"load_modec_{sentinel_tag}_rank{rank}.err"), "w"
        ) as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_world_size_reshard_4_to_2_offline(tmp_path):
    """Live Mode-C 4→2 reshard via the offline tool.

    Phase 1: spawn 4 ranks → save Mode-C with deterministic state pattern.
    Phase 1b: spawn 2 ranks → save Mode-C with the SAME pattern (the
        per-rank slicing differs, but the underlying logical full-
        padded-region content is identical). This is the "natively-N=2"
        reference.
    Phase 2: invoke scripts/protrain/reshard_optim.py to reshard 4→2,
        producing a directory whose layout matches the natively-N=2 one.
    Phase 3a: spawn 2 ranks → load the resharded dir → step → hash.
    Phase 3b: spawn 2 ranks → load the natively-N=2 dir → step → hash.
        Phase 3a and 3b's hashes must match — the resharded state is
        semantically equivalent to natively-N=2 state.
    """
    pytest.importorskip("torch")
    import subprocess

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    n_visible = torch.cuda.device_count()
    if n_visible < 4:
        pytest.skip(f"reshard test needs >= 4 visible GPUs (got {n_visible})")

    import torch.multiprocessing as mp

    # ---- Phase 1: save N=4 ------------------------------------------
    save_world_4 = 4
    mp.spawn(
        _save_worker_modec,
        args=(save_world_4, str(tmp_path), "n4"),
        nprocs=save_world_4,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_modec_n4_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 1 (N=4 save) errors:\n{bodies}")
    for r in range(save_world_4):
        assert (tmp_path / f"save_modec_n4_rank{r}.done").is_file(), (
            f"N=4 save rank {r} did not reach sentinel"
        )

    save_n4_root = tmp_path / "save_n4" / PROTRAIN_OPTIM_DIRNAME
    assert save_n4_root.is_dir(), f"save_n4 root {save_n4_root} missing post-spawn"
    n4_meta = json.loads((save_n4_root / METADATA_FILENAME).read_text())
    assert n4_meta["protrain_save_mode"] == SAVE_MODE_SHARDED
    assert n4_meta["protrain_world_size"] == save_world_4
    assert "layout_fingerprint" in n4_meta, (
        "save metadata must record layout_fingerprint for offline reshard"
    )

    # ---- Phase 1b: save N=2 (reference) -----------------------------
    save_world_2 = 2
    mp.spawn(
        _save_worker_modec,
        args=(save_world_2, str(tmp_path), "n2"),
        nprocs=save_world_2,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_modec_n2_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 1b (N=2 save) errors:\n{bodies}")
    save_n2_root = tmp_path / "save_n2" / PROTRAIN_OPTIM_DIRNAME
    assert save_n2_root.is_dir()

    # ---- Phase 2: offline reshard 4→2 -------------------------------
    save_n4_resharded_root = tmp_path / "save_n4_resharded" / PROTRAIN_OPTIM_DIRNAME
    save_n4_resharded_root.parent.mkdir(parents=True, exist_ok=True)

    # Run the reshard tool as a subprocess so it exercises the CLI path.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    reshard_script = os.path.join(repo_root, "scripts", "protrain", "reshard_optim.py")
    assert os.path.isfile(reshard_script), f"reshard tool not found at {reshard_script}"

    cmd = [
        sys.executable,
        reshard_script,
        "--src",
        str(save_n4_root),
        "--dst",
        str(save_n4_resharded_root),
        "--target-world",
        str(save_world_2),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.fail(
            f"reshard tool failed: rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    # Sanity: resharded metadata records new world_size and matching
    # per-rank shard files exist.
    resharded_meta = json.loads(
        (save_n4_resharded_root / METADATA_FILENAME).read_text()
    )
    assert resharded_meta["protrain_world_size"] == save_world_2, (
        f"resharded metadata still records world_size="
        f"{resharded_meta['protrain_world_size']}"
    )
    assert resharded_meta["protrain_save_mode"] == SAVE_MODE_SHARDED
    assert resharded_meta["resharded_from_world_size"] == save_world_4
    cpu_dir = save_n4_resharded_root / CPU_OPTIM_DIRNAME
    for cid in resharded_meta["regions_per_chunk"]:
        for r in range(save_world_2):
            shard_path = cpu_dir / f"chunk_{int(cid)}_rank_{r}.pt"
            assert shard_path.is_file(), (
                f"resharded dir missing per-rank shard {shard_path.name}"
            )
        # No leftover N=4 ranks.
        for r in range(save_world_2, save_world_4):
            stale = cpu_dir / f"chunk_{int(cid)}_rank_{r}.pt"
            assert not stale.exists(), (
                f"resharded dir contains leftover N=4 shard {stale.name}"
            )

    # ---- Phase 3a: load resharded dir, step --------------------------
    mp.spawn(
        _load_worker_modec,
        args=(
            save_world_2,
            str(tmp_path),
            os.path.join("save_n4_resharded"),
            "resharded",
        ),
        nprocs=save_world_2,
        join=True,
    )
    err_files = sorted(tmp_path.glob("load_modec_resharded_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 3a (resharded load) errors:\n{bodies}")

    # ---- Phase 3b: load natively-N=2 dir, step -----------------------
    mp.spawn(
        _load_worker_modec,
        args=(
            save_world_2,
            str(tmp_path),
            os.path.join("save_n2"),
            "native",
        ),
        nprocs=save_world_2,
        join=True,
    )
    err_files = sorted(tmp_path.glob("load_modec_native_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 3b (native N=2 load) errors:\n{bodies}")

    # ---- Equivalence check: per-rank, all three hashes must match ----
    # post_load_hash, post_step_hash, param_hash all should match
    # between the resharded and the native paths (the deterministic
    # state pattern, the deterministic gradient batch, and the
    # deterministic Adam step combine to give bit-identical results
    # IFF the reshard preserved the underlying logical state).
    for r in range(save_world_2):
        resharded_hash = (
            (tmp_path / f"load_modec_resharded_rank{r}.hash").read_text().strip()
        )
        native_hash = (tmp_path / f"load_modec_native_rank{r}.hash").read_text().strip()
        rh_post_load, rh_post_step, rh_param = resharded_hash.split(":")
        nh_post_load, nh_post_step, nh_param = native_hash.split(":")
        assert rh_post_load == nh_post_load, (
            f"rank {r}: post-load inner-state hash differs between "
            f"resharded and native paths.\n"
            f"  resharded={rh_post_load}\n"
            f"  native   ={nh_post_load}\n"
            "The reshard tool produced semantically different state."
        )
        assert rh_post_step == nh_post_step, (
            f"rank {r}: post-step inner-state hash differs between "
            f"resharded and native paths.\n"
            f"  resharded={rh_post_step}\n"
            f"  native   ={nh_post_step}\n"
            "One Adam step on the resharded state diverged from one "
            "step on natively-saved-N=2 state — semantic equivalence "
            "broken."
        )
        assert rh_param == nh_param, (
            f"rank {r}: post-step parameter hash differs between "
            f"resharded and native paths."
        )


# ===========================================================================
# Mode-C (ZeRO-3 sharded) — online reshard on load (opt-in)
# ===========================================================================
#
# Mirror of the offline test above. The save phases (N=4 and N=2 reference)
# are reused verbatim. Phase 2 — instead of running the offline CLI — the
# load workers pass ``allow_online_reshard=True`` against the original N=4
# save dir. The loader does the reshard internally:
#
#   * rank-0 invokes ``reshard_mode_c_shards`` against a sibling temp dir
#     (``<save_dir>/protrain_optim/.reshard_to_N2/``)
#   * all ranks barrier
#   * load proceeds against the temp dir as if it were a natively-N=2 save
#   * rank-0 cleans up the temp dir post-load.
#
# Acceptance is identical to the offline test: per-rank post-load hash,
# post-step hash, and post-step parameter hash must match the natively-
# N=2 reference path.


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_world_size_reshard_4_to_2_online(tmp_path):
    """Live Mode-C 4→2 reshard via the online opt-in path.

    Phase 1: spawn 4 ranks → save Mode-C with deterministic state pattern.
    Phase 1b: spawn 2 ranks → save Mode-C natively-N=2 (reference).
    Phase 2: spawn 2 ranks → load the original N=4 dir with
        ``allow_online_reshard=True``. The loader reshards internally.
    Phase 3: spawn 2 ranks → load the natively-N=2 dir as a control.
        Phase 2 and Phase 3 hashes must match — proves the online
        reshard produced semantically identical state, with no CLI
        invocation in the loop.

    Sanity: after the online load completes, the temp dir
    ``protrain_optim/.reshard_to_N2/`` must be cleaned up by rank-0.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    n_visible = torch.cuda.device_count()
    if n_visible < 4:
        pytest.skip(f"online reshard test needs >= 4 visible GPUs (got {n_visible})")

    import torch.multiprocessing as mp

    # ---- Phase 1: save N=4 ------------------------------------------
    save_world_4 = 4
    mp.spawn(
        _save_worker_modec,
        args=(save_world_4, str(tmp_path), "n4"),
        nprocs=save_world_4,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_modec_n4_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 1 (N=4 save) errors:\n{bodies}")
    for r in range(save_world_4):
        assert (tmp_path / f"save_modec_n4_rank{r}.done").is_file(), (
            f"N=4 save rank {r} did not reach sentinel"
        )

    save_n4_root = tmp_path / "save_n4" / PROTRAIN_OPTIM_DIRNAME
    assert save_n4_root.is_dir()
    n4_meta = json.loads((save_n4_root / METADATA_FILENAME).read_text())
    assert n4_meta["protrain_save_mode"] == SAVE_MODE_SHARDED
    assert n4_meta["protrain_world_size"] == save_world_4

    # ---- Phase 1b: save N=2 (reference) -----------------------------
    save_world_2 = 2
    mp.spawn(
        _save_worker_modec,
        args=(save_world_2, str(tmp_path), "n2"),
        nprocs=save_world_2,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_modec_n2_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 1b (N=2 save) errors:\n{bodies}")

    # ---- Phase 2: online load N=4 → N=2 with opt-in flag ------------
    # Pointed at the ORIGINAL N=4 save dir; the loader handles the
    # reshard internally. Sentinel tag "online" namespaces the .done /
    # .hash artifacts so they don't collide with the N=2 native load
    # below.
    mp.spawn(
        _load_worker_modec,
        args=(
            save_world_2,
            str(tmp_path),
            "save_n4",  # original N=4 dir
            "online",
            True,  # allow_online_reshard
        ),
        nprocs=save_world_2,
        join=True,
    )
    err_files = sorted(tmp_path.glob("load_modec_online_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 2 (online reshard load) errors:\n{bodies}")

    # ---- Phase 3: load natively-N=2 dir as control ------------------
    mp.spawn(
        _load_worker_modec,
        args=(
            save_world_2,
            str(tmp_path),
            "save_n2",
            "native_for_online",
            False,  # native — no reshard needed
        ),
        nprocs=save_world_2,
        join=True,
    )
    err_files = sorted(tmp_path.glob("load_modec_native_for_online_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 3 (native N=2 control load) errors:\n{bodies}")

    # ---- Equivalence check ------------------------------------------
    for r in range(save_world_2):
        online_hash = (tmp_path / f"load_modec_online_rank{r}.hash").read_text().strip()
        native_hash = (
            (tmp_path / f"load_modec_native_for_online_rank{r}.hash")
            .read_text()
            .strip()
        )
        oh_post_load, oh_post_step, oh_param = online_hash.split(":")
        nh_post_load, nh_post_step, nh_param = native_hash.split(":")
        assert oh_post_load == nh_post_load, (
            f"rank {r}: post-load inner-state hash differs between "
            f"online-resharded and native paths.\n"
            f"  online ={oh_post_load}\n"
            f"  native ={nh_post_load}\n"
            "The online reshard produced semantically different state."
        )
        assert oh_post_step == nh_post_step, (
            f"rank {r}: post-step inner-state hash differs between "
            f"online-resharded and native paths.\n"
            f"  online ={oh_post_step}\n"
            f"  native ={nh_post_step}"
        )
        assert oh_param == nh_param, (
            f"rank {r}: post-step parameter hash differs between "
            f"online-resharded and native paths."
        )

    # ---- Cleanup sanity: temp dir must be removed -------------------
    # The online load worker exits cleanly, so rank-0's cleanup should
    # have run. We verify the temp dir under save_n4/protrain_optim/
    # is gone — leftover means a regression in the cleanup branch.
    temp_dir = save_n4_root / f".reshard_to_N{save_world_2}"
    assert not temp_dir.exists(), (
        f"online reshard temp dir {temp_dir} still present after "
        "successful load; rank-0 cleanup must have failed silently"
    )


# ===========================================================================
# Mode-C (ZeRO-3 sharded) — opt-out default still hard-errors
# ===========================================================================
#
# When ``protrain_allow_online_reshard=False`` (the default) and
# saved_world != current_world, the load path must hard-error with a
# message that points the user at BOTH the offline CLI and the opt-in
# flag. Mirror of the existing single-process metadata-fake test, but
# this time covers the live cross-world-size error surface from the
# loader-as-of-2026-04-30.


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_world_size_reshard_4_to_2_default_hard_errors(tmp_path):
    """Default (no opt-in) Mode-C cross-world-size load is a hard error.

    Phase 1: save N=4 (reuse _save_worker_modec).
    Phase 2: spawn 2 ranks, attempt to load the N=4 save without
        ``allow_online_reshard=True``. Each rank must raise; the error
        message must reference both the offline CLI and the opt-in
        flag.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    n_visible = torch.cuda.device_count()
    if n_visible < 4:
        pytest.skip(
            f"hard-error opt-out test needs >= 4 visible GPUs (got {n_visible})"
        )

    import torch.multiprocessing as mp

    # ---- Phase 1: save N=4 ------------------------------------------
    save_world_4 = 4
    mp.spawn(
        _save_worker_modec,
        args=(save_world_4, str(tmp_path), "n4"),
        nprocs=save_world_4,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_modec_n4_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 1 (N=4 save) errors:\n{bodies}")

    # ---- Phase 2: load N=2 default (no opt-in) — must hard-error ----
    save_world_2 = 2
    # The load worker raises on the worker side; ``mp.spawn`` propagates
    # via a ProcessRaisedException on the parent. We catch it and check
    # the .err sentinel for the message.
    with pytest.raises(Exception):  # noqa: PT011, B017
        mp.spawn(
            _load_worker_modec,
            args=(
                save_world_2,
                str(tmp_path),
                "save_n4",
                "default_hard_err",
                False,  # allow_online_reshard=False (the default)
            ),
            nprocs=save_world_2,
            join=True,
        )

    err_files = sorted(tmp_path.glob("load_modec_default_hard_err_rank*.err"))
    assert err_files, (
        "expected per-rank .err sentinels from the failing load workers; "
        "either the workers didn't raise or the spawn didn't propagate"
    )
    # The lockstep broadcast surfaces a synthesised message on non-source
    # ranks; the source rank carries the full human message. The recovery
    # routes (offline CLI + opt-in flag) must be visible somewhere across
    # ranks — check the union.
    union = "\n".join(ef.read_text() for ef in err_files)
    assert "scripts.protrain.reshard_optim" in union, (
        "default-error must point at the offline CLI tool"
    )
    assert "protrain_allow_online_reshard" in union, (
        "default-error must point at the opt-in flag"
    )


# ===========================================================================
# Mode-C (ZeRO-3 sharded) — lockstep failure surface for online reshard
# ===========================================================================
#
# When ``allow_online_reshard=True`` but rank-0's reshard fails (e.g.
# the source dir has been corrupted between save and load), every rank
# must surface the error consistently — no rank-0-only stuck state.
# We simulate the failure by deleting one of the N=4 per-rank shard
# files between the save and the load; rank-0's reshard tries to read
# it, raises, and broadcasts a non-zero status to the other ranks via
# ``_broadcast_status_or_raise``.


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_world_size_online_reshard_lockstep_failure(tmp_path):
    """Rank-0 reshard failure surfaces on every rank in lockstep.

    Phase 1: save N=4 normally.
    Phase 1b: corrupt the save by deleting one of the per-rank shards
        (rank 3's shard for an arbitrary chunk).
    Phase 2: spawn 2 ranks with ``allow_online_reshard=True``. Rank-0
        starts the reshard, hits the missing file, broadcasts status=1.
        Every rank's worker writes a .err sentinel; the spawn surfaces
        a non-zero exit on the parent.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    n_visible = torch.cuda.device_count()
    if n_visible < 4:
        pytest.skip(f"lockstep-failure test needs >= 4 visible GPUs (got {n_visible})")

    import torch.multiprocessing as mp

    # ---- Phase 1: save N=4 ------------------------------------------
    save_world_4 = 4
    mp.spawn(
        _save_worker_modec,
        args=(save_world_4, str(tmp_path), "n4"),
        nprocs=save_world_4,
        join=True,
    )
    err_files = sorted(tmp_path.glob("save_modec_n4_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"phase 1 (N=4 save) errors:\n{bodies}")

    # ---- Phase 1b: corrupt one shard --------------------------------
    save_n4_root = tmp_path / "save_n4" / PROTRAIN_OPTIM_DIRNAME
    cpu_dir = save_n4_root / CPU_OPTIM_DIRNAME
    # Pick the first chunk + rank 3 (will fail when the reshard tries
    # to read all 4 ranks for that chunk).
    n4_meta = json.loads((save_n4_root / METADATA_FILENAME).read_text())
    chunk_ids = sorted(int(c) for c in n4_meta["regions_per_chunk"].keys())
    if not chunk_ids:
        pytest.skip("Mode-C save produced no chunk shards (no non-persistent chunks)")
    cid = chunk_ids[0]
    victim = cpu_dir / f"chunk_{cid}_rank_3.pt"
    assert victim.is_file(), f"setup error: expected {victim} to exist"
    victim.unlink()

    # ---- Phase 2: online load with corrupted source -----------------
    save_world_2 = 2
    with pytest.raises(Exception):  # noqa: PT011, B017
        mp.spawn(
            _load_worker_modec,
            args=(
                save_world_2,
                str(tmp_path),
                "save_n4",
                "lockstep_fail",
                True,  # allow_online_reshard=True
            ),
            nprocs=save_world_2,
            join=True,
        )

    err_files = sorted(tmp_path.glob("load_modec_lockstep_fail_rank*.err"))
    assert err_files, (
        "expected per-rank .err sentinels from the lockstep failure; "
        "if only rank-0 raised the cluster would have wedged at the "
        "trailing barrier"
    )
    # Acceptance: BOTH ranks must have an .err sentinel (not just rank-0).
    rank_to_err = {int(p.name.split("rank")[1].split(".")[0]): p for p in err_files}
    assert set(rank_to_err.keys()) == set(range(save_world_2)), (
        f"only ranks {sorted(rank_to_err.keys())} surfaced an error — "
        "lockstep failure protocol broken; expected every rank to raise"
    )


# ===========================================================================
# v3 persistent partition: world_size IDENTITY required on resume
# ===========================================================================


def test_persistent_partition_refuses_world_size_change_on_resume(tmp_path):
    """A v3 save with ``protrain_persistent_partition_version: 1`` and
    ``protrain_persistent_owner_world_size: 4`` must hard-error when
    loaded into a w=2 build (online reshard does NOT support
    repartitioning the persistent fp32 master).

    Single-process test — exercises the metadata-driven refuse path,
    not the multi-rank live save. The live save+load cycle is covered
    by the partitioned tests in ``test_modec_persistent_partition.py``
    and ``test_optimizer_checkpoint.py``.
    """
    from unittest import mock

    from axolotl.integrations.protrain.api.checkpoint import (
        SCHEMA_FORMAT_VERSION,
        _layout_signature,
        _load_protrain_optim_dir,
    )

    proot = tmp_path / PROTRAIN_OPTIM_DIRNAME
    proot.mkdir()

    fake_layout = mock.MagicMock(S_chunk=1024, N_chunk=1, chunks=(("a",),))
    fake_mgr = mock.MagicMock(
        layout=fake_layout,
        _persistent_ids={0},
        zero3_shard=False,
        _chunk_shards={},
    )
    sig = _layout_signature(fake_mgr, world_size=4, zero3_shard=False)
    meta = {
        "format_version": SCHEMA_FORMAT_VERSION,
        "protrain_layout_signature": sig,
        "protrain_persistent_ids": [0],
        "protrain_n_buffer": 1,
        "protrain_world_size": 4,
        "protrain_zero3_shard": False,
        "protrain_save_mode": "replicated",
        "saving_rank": 0,
        "param_groups_meta": [],
        "saved_at_step": 0,
        "torch_version": "x",
        "estimated_optim_state_bytes": 0,
        "protrain_persistent_partition_version": 1,
        "protrain_persistent_owner_world_size": 4,
    }
    (proot / METADATA_FILENAME).write_text(json.dumps(meta))

    fake_optim = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    fake_optim._chunk_manager = fake_mgr
    fake_optim._gpu_optim = None
    fake_optim._cpu_optim = None

    # Saved w=4, but pretend current_world=2. Loader must error with the
    # documented identity-required message.
    with mock.patch(
        "axolotl.integrations.protrain.api.checkpoint._current_world_size",
        return_value=2,
    ):
        with pytest.raises(RuntimeError, match="world_size mismatch on resume"):
            _load_protrain_optim_dir(fake_optim, str(tmp_path))
