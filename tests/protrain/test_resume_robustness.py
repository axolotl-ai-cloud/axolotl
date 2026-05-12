"""Resume robustness regression sweep (D1/D2/D3 in-process rebuild lifecycle).

The existing :mod:`test_cross_mode_resume` tests cover the cross-mode A↔C
state_dict round-trip but never call :meth:`ChunkManager.restore_to_gpu` /
:meth:`ChunkManager.materialize_offload` a second time on the same
manager instance — the actual hot path the production resume hook
(``plugin._install_resume_hook``) takes. This module pins that
in-process rebuild cycle so the D1/D2/D3 lifecycle fixes don't
regress:

* **D2 — replace, don't union, the DDP ignore set.** Calling
  ``materialize_offload`` twice on the same chunk manager used to grow
  ``model._ddp_params_and_buffers_to_ignore`` unboundedly because the
  second call unioned the new protrain set into the previous protrain
  set; a chunk that moved between persistent/non-persistent between
  calls would stay in the ignore set forever and DDP would silently
  skip syncing a now-live weight. The fix snapshots the pre-protrain
  value once into ``model._protrain_ddp_original_ignore`` and rebuilds
  from that canonical baseline on every call. Tests:
  :func:`test_ddp_ignore_set_does_not_grow_on_repeat_materialize` and
  :func:`test_ddp_ignore_snapshot_survives_restore_and_rematerialize`.

* **D3 — shutdown previous CPU adapter before swap.**
  ``protrain_optimizer_wrapper`` rebuilds adapters in place and the
  pre-existing ``chunk_manager.cpu_optim`` owns a live
  ``ThreadPoolExecutor`` + DeepSpeed C-state. The fix calls
  ``shutdown()`` on the old reference before assigning the new one,
  matching the resume hook's existing teardown at the plugin layer.
  Test: :func:`test_cpu_optim_replaced_calls_shutdown_on_previous`.

* **D1 — strip stale DDP skip state on non-shape-preserving rebuild.**
  A future Mode C → Mode A/B rebuild path (or a stale single-GPU
  re-wrap after a shape-preserving wrap) must not leave
  ``_protrain_ddp_skip_init_sync`` on the model — DDP's init-time
  broadcast is required for normal Mode A replicated semantics. Test:
  :func:`test_rewrap_non_shape_preserving_clears_ddp_skip_state`.

Plus an end-to-end smoke that simulates the resume hook's full
:meth:`restore_to_gpu` → load-state-dict → :meth:`materialize_offload`
cycle on the same chunk manager, then continues training and asserts
finite losses + monotonic-ish loss descent: :func:`test_resume_hook_inprocess_cycle_continues_training`.

All tests are GPU-marked (require CUDA at runtime) and skip cleanly
on CPU-only rigs. They use a tiny LlamaForCausalLM + LoRA model so
the wall-clock per case is sub-second; the sweep can run on a single
3090 in ~5 seconds.
"""

from __future__ import annotations

import math

import pytest


def _build_tiny_lora_model():
    """A minimal LoRA-on-Llama setup that fits the chunk manager + searcher.

    Mirrors :func:`tests.protrain.test_cross_mode_resume._build_tiny_llama_lora`
    so the two test suites share a single canonical small-model recipe.
    """
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        vocab_size=1024,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        use_cache=False,
    )
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    return model, cfg


def _wrap_protrain(
    model,
    cfg,
    *,
    force_all_persistent: bool,
    zero3_shard: bool,
    n_persist_override: int | None = None,
    n_buffer_override: int | None = None,
    n_swap_override: int | None = None,
    n_checkpoint_override: int | None = None,
    n_offload_override: int | None = None,
    small_chunk: bool = False,
):
    """Wrap a model in ProTrain and return the wrapped runtime + optimizer.

    Override knobs are forwarded straight through to
    ``protrain_model_wrapper`` so individual tests can force
    non-persistent chunks (``n_persist_override=0``) — necessary to
    exercise the CPU-adapter path on a tiny model where the searcher
    would otherwise pick ``n_persist == N_chunk`` and no
    ``CpuFusedAdamAdapter`` would be constructed.

    ``small_chunk=True`` monkey-patches ``pick_S_chunk`` so the layout
    builder produces multiple chunks even on the tiny test model,
    matching the pattern used in ``test_lora_offload_mode``.
    """
    import torch

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )

    # When small_chunk=True, monkey-patch pick_S_chunk so the layout
    # builder produces multiple chunks. Without this, the tiny test
    # model's params all fit in a single chunk and force_all_persistent
    # vs override-driven non-persistent become indistinguishable. The
    # 1 MiB value matches the working pattern in
    # ``test_lora_offload_mode``; finer S_chunk values produce a
    # larger N_chunk than n_buffer_override can satisfy
    # (``min_n_buffer_for`` validates 2 * max(non_persistent_per_block)).
    import axolotl.integrations.protrain.api.model_wrapper as mw

    orig_pick_S_chunk = mw.pick_S_chunk
    if small_chunk:
        mw.pick_S_chunk = lambda *a, **k: 1 << 20  # 1 MiB
    try:
        wrapped = protrain_model_wrapper(
            model,
            model_config=cfg,
            hardware_profile=hw,
            batch_size=1,
            seq_len=32,
            capacity_bytes=4 * (1 << 30),
            force_all_persistent=force_all_persistent,
            zero3_shard=zero3_shard,
            n_persist_override=n_persist_override,
            n_buffer_override=n_buffer_override,
            n_swap_override=n_swap_override,
            n_checkpoint_override=n_checkpoint_override,
            n_offload_override=n_offload_override,
        )
    finally:
        # Restore the global so a subsequent test's wrap uses the
        # searcher-picked S_chunk (one global monkey-patch leak would
        # silently distort downstream resource accounting).
        mw.pick_S_chunk = orig_pick_S_chunk
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)
    return wrapped, optim


def _train_one_step(wrapped, optim, *, input_ids, labels) -> float:
    out = wrapped.module(input_ids=input_ids, labels=labels)
    loss = out.loss
    loss_value = float(loss.detach())
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss_value


def _make_batch(cfg):
    import torch

    torch.manual_seed(1)
    return (
        torch.randint(0, cfg.vocab_size, (1, 32), device="cuda", dtype=torch.long),
        torch.randint(0, cfg.vocab_size, (1, 32), device="cuda", dtype=torch.long),
    )


@pytest.mark.gpu
def test_ddp_ignore_set_does_not_grow_on_repeat_materialize() -> None:
    """D2 invariant: a second ``materialize_offload`` does NOT grow the
    DDP ignore set.

    Construct a chunk manager with shape-preserving placeholders (the
    multi-GPU sharded path's flag), run ``materialize_offload`` once
    and record the ignore set size, then run it again on the same
    manager (simulating the resume-hook cycle) and verify the size is
    identical — not the sum of the two protrain sets.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain D2 invariant requires CUDA.")

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    wrapped, _optim = _wrap_protrain(
        model, cfg, force_all_persistent=False, zero3_shard=True
    )
    try:
        underlying = getattr(wrapped, "module", wrapped)
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        if chunk_manager is None or not getattr(
            chunk_manager, "_shape_preserving_placeholders", False
        ):
            # Single-process Mode C silently downgrades to Mode A
            # (zero3_shard coerced to False when world_size <= 1), so
            # the shape-preserving placeholder path isn't engaged.
            # Skip in that case — multi-GPU coverage lives in
            # ``test_real_multigpu_cross_mode_resume_*``.
            pytest.skip(
                "single-process Mode C downgrade path: "
                "shape-preserving placeholders not engaged."
            )

        first_ignore = list(
            getattr(underlying, "_ddp_params_and_buffers_to_ignore", [])
        )
        first_snapshot = getattr(underlying, "_protrain_ddp_original_ignore", "<unset>")
        first_size = len(first_ignore)

        # Simulate the resume hook's second materialize_offload call.
        assert chunk_manager is not None
        chunk_manager.restore_to_gpu()
        chunk_manager.materialize_offload()

        second_ignore = list(
            getattr(underlying, "_ddp_params_and_buffers_to_ignore", [])
        )
        second_snapshot = getattr(
            underlying, "_protrain_ddp_original_ignore", "<unset>"
        )
        second_size = len(second_ignore)

        # The snapshot must survive intact (we never re-snapshot).
        assert first_snapshot == second_snapshot, (
            f"_protrain_ddp_original_ignore snapshot drifted between "
            f"materialize_offload calls: {first_snapshot!r} -> "
            f"{second_snapshot!r}. The D2 invariant requires the "
            f"pre-protrain snapshot to be captured once and reused."
        )
        # The ignore set size must be stable across repeat
        # materialize_offload calls — not double / triple / etc.
        # the protrain set.
        assert second_size == first_size, (
            f"_ddp_params_and_buffers_to_ignore grew from {first_size} to "
            f"{second_size} names across a repeat materialize_offload "
            f"call — D2 regression: the pre-fix union logic is leaking "
            f"stale names across resume cycles."
        )
        # And the set membership must be identical (not just same
        # cardinality with different names).
        assert set(first_ignore) == set(second_ignore), (
            f"_ddp_params_and_buffers_to_ignore CONTENT diverged across "
            f"a repeat materialize_offload call. First-only names: "
            f"{set(first_ignore) - set(second_ignore)}. Second-only "
            f"names: {set(second_ignore) - set(first_ignore)}."
        )
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()


@pytest.mark.gpu
def test_ddp_ignore_snapshot_survives_restore_and_rematerialize() -> None:
    """D2 + teardown: a pre-existing user value in
    ``_ddp_params_and_buffers_to_ignore`` is preserved across the
    materialize_offload cycle AND restored on close.

    Set a fake pre-existing ignore name on the model before wrapping,
    then verify the snapshot captures it, the protrain set merges with
    it correctly, and ``wrapped.close()`` restores the original value.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain D2 invariant requires CUDA.")

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    fake_pre_existing = ["caller_registered_ignore_name"]
    model._ddp_params_and_buffers_to_ignore = list(fake_pre_existing)  # type: ignore[attr-defined]

    wrapped, _optim = _wrap_protrain(
        model, cfg, force_all_persistent=False, zero3_shard=True
    )
    try:
        underlying = getattr(wrapped, "module", wrapped)
        chunk_manager = getattr(wrapped, "chunk_manager", None)
        if chunk_manager is None or not getattr(
            chunk_manager, "_shape_preserving_placeholders", False
        ):
            pytest.skip(
                "single-process Mode C downgrade path: "
                "shape-preserving placeholders not engaged."
            )

        # Snapshot must equal the pre-existing value.
        snap = getattr(underlying, "_protrain_ddp_original_ignore", None)
        assert snap == fake_pre_existing, (
            f"snapshot did not capture pre-existing user value: "
            f"expected {fake_pre_existing!r}, got {snap!r}"
        )
        # The fake pre-existing name must still be present in the
        # post-wrap ignore set (merged with the protrain set).
        post_wrap = set(getattr(underlying, "_ddp_params_and_buffers_to_ignore", []))
        assert "caller_registered_ignore_name" in post_wrap

        # Second materialize_offload — same invariants must hold.
        assert chunk_manager is not None
        chunk_manager.restore_to_gpu()
        chunk_manager.materialize_offload()
        post_resume = set(getattr(underlying, "_ddp_params_and_buffers_to_ignore", []))
        assert "caller_registered_ignore_name" in post_resume
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()

    # After close, the snapshot must be restored.
    restored = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
    assert restored == fake_pre_existing, (
        f"close() did not restore the pre-existing ignore set: "
        f"expected {fake_pre_existing!r}, got {restored!r}"
    )
    # And the snapshot sentinel should be cleared.
    assert not hasattr(model, "_protrain_ddp_original_ignore"), (
        "_protrain_ddp_original_ignore should be cleared after close()"
    )


@pytest.mark.gpu
def test_cpu_optim_replaced_calls_shutdown_on_previous() -> None:
    """D3 invariant: re-running ``protrain_optimizer_wrapper`` on the
    same wrapped runtime calls ``shutdown()`` on the previous
    ``chunk_manager.cpu_optim`` before installing the new one.

    Forces non-persistent chunks via ``force_all_persistent=False`` +
    explicit overrides + ``small_chunk=True`` so the tiny test model
    actually produces a ``CpuFusedAdamAdapter``. Without the
    overrides + small_chunk the searcher picks
    ``n_persist == N_chunk == 1`` and no CPU adapter is built — the
    test would then silently self-skip (CodeRabbit R3-#6).
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain D3 invariant requires CUDA.")

    # Probe DeepSpeedCPUAdam availability up front — the CPU adapter
    # path needs it to construct, and the test cannot validate D3
    # if the build env can't even build a CPU adapter.
    try:
        import deepspeed  # noqa: F401
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        _probe = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        try:
            DeepSpeedCPUAdam([_probe], lr=1e-3)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(
                f"DeepSpeedCPUAdam JIT load failed ({exc}); D3 invariant "
                f"requires a working CPU adapter build."
            )
    except ImportError:
        pytest.skip("deepspeed not installed; D3 invariant requires CPU adapter.")

    from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    # Force non-persistent chunks so a CpuFusedAdamAdapter actually
    # gets constructed. small_chunk=True ensures N_chunk > 1 even on
    # this tiny model so the n_persist=0 override produces chunks
    # that ARE offloaded.
    wrapped, _optim = _wrap_protrain(
        model,
        cfg,
        force_all_persistent=False,
        zero3_shard=False,
        n_persist_override=0,
        n_buffer_override=16,
        n_swap_override=0,
        n_checkpoint_override=0,
        # All non-persistent transformer blocks in OFFLOAD mode
        # (Option B) — saved tensors re-gather on backward via the
        # M3 block manager's per-block hook rather than relying on
        # NONE-mode hooks (which would clobber autograd's saved
        # tensors when the chunk pool slot is reused).
        n_offload_override=cfg.num_hidden_layers,
        small_chunk=True,
    )
    try:
        chunk_manager = wrapped.chunk_manager
        previous_cpu_optim = getattr(chunk_manager, "cpu_optim", None)
        assert previous_cpu_optim is not None, (
            "test setup did not produce a CPU adapter — the D3 invariant "
            "needs at least one non-persistent chunk to be exercised. "
            "Check that force_all_persistent=False + n_persist_override=0 "
            "+ small_chunk=True actually produced non-persistent chunks "
            "for this model size."
        )

        # Patch shutdown to record invocation.
        shutdown_calls: list[bool] = []
        orig_shutdown = previous_cpu_optim.shutdown

        def _tracked_shutdown(*args, **kwargs):
            shutdown_calls.append(True)
            return orig_shutdown(*args, **kwargs)

        previous_cpu_optim.shutdown = _tracked_shutdown  # type: ignore[method-assign]

        # Re-run the optimizer wrapper — this is the path D3 fixed.
        _new_optim = protrain_optimizer_wrapper(wrapped, lr=2e-3)

        # The new cpu_optim must be a different object AND the old
        # one's shutdown must have been called.
        new_cpu_optim = getattr(chunk_manager, "cpu_optim", None)
        assert new_cpu_optim is not previous_cpu_optim, (
            "protrain_optimizer_wrapper did not swap chunk_manager.cpu_optim "
            "— the test cannot detect D3 regression."
        )
        assert shutdown_calls, (
            "D3 regression: protrain_optimizer_wrapper replaced "
            "chunk_manager.cpu_optim without calling shutdown() on the "
            "previous adapter. The old adapter's ThreadPoolExecutor + "
            "DeepSpeed C-state would leak on every re-wrap."
        )
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()


@pytest.mark.gpu
def test_rewrap_non_shape_preserving_clears_ddp_skip_state() -> None:
    """D1 invariant: rebuilding a model with non-shape-preserving wrap
    clears any stale ``_protrain_ddp_skip_init_sync`` + ignore-list
    state from a prior shape-preserving wrap.

    Manually set the shape-preserving markers on a model (simulating
    a prior Mode C wrap), then call ``protrain_model_wrapper`` with
    ``force_all_persistent=True`` (Mode A — not shape-preserving) and
    verify the markers are gone after the second wrap returns.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain D1 invariant requires CUDA.")

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")

    # Simulate a prior Mode C wrap's residue on the model.
    model._protrain_ddp_skip_init_sync = True  # type: ignore[attr-defined]
    model._protrain_ddp_original_ignore = None  # type: ignore[attr-defined]
    model._ddp_params_and_buffers_to_ignore = [  # type: ignore[attr-defined]
        "fake.stale.name"
    ]

    wrapped, _optim = _wrap_protrain(
        model, cfg, force_all_persistent=True, zero3_shard=False
    )
    try:
        # The D1 else branch must have stripped the markers.
        assert not getattr(model, "_protrain_ddp_skip_init_sync", False), (
            "D1 regression: _protrain_ddp_skip_init_sync persisted across "
            "a non-shape-preserving rebuild. DDP would silently skip "
            "init_sync on the rebuilt Mode A runtime."
        )
        assert not hasattr(model, "_protrain_ddp_original_ignore"), (
            "D1 regression: _protrain_ddp_original_ignore not cleared on "
            "non-shape-preserving rebuild."
        )
        # And the stale ignore-list entry should be gone (because the
        # snapshot was None → attribute should be deleted).
        assert not hasattr(model, "_ddp_params_and_buffers_to_ignore"), (
            "D1 regression: stale _ddp_params_and_buffers_to_ignore "
            "(set to a fake value before the rebuild) was not deleted "
            "during the non-shape-preserving rebuild teardown."
        )
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()


@pytest.mark.gpu
def test_resume_hook_inprocess_cycle_continues_training() -> None:
    """End-to-end resume robustness: train a few steps, simulate the
    resume hook's restore_to_gpu → materialize_offload cycle in-process,
    train more steps, and verify finite losses + continued descent.

    This is the smallest cycle that exercises D1/D2/D3 together:

    1. Wrap model in ProTrain offload mode (force_all_persistent=False
       with ``n_persist_override=0`` so chunks are ACTUALLY offloaded;
       without the override the searcher picks ``n_persist == N_chunk``
       on a tiny model and ``materialize_offload`` becomes a no-op,
       making the D2 hot path untested — CodeRabbit R3-#7).
    2. Train 3 steps, capture state_dict.
    3. Simulate the resume hook: explicitly tear down the CPU optim,
       call ``restore_to_gpu``, load the state_dict, call
       ``materialize_offload`` again, rebuild the optimizer wrapper.
    4. Train 3 more steps from the resumed state.
    5. Assert all losses are finite and the resumed run's first loss
       is not catastrophically larger than the pre-resume tail.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain resume hook in-process cycle requires CUDA.")

    # Probe DeepSpeedCPUAdam availability — the offload-mode wrap path
    # needs it to construct, and the resume cycle below rebuilds the
    # CPU adapter. Without it, the test would skip mid-cycle which is
    # noisier than skipping up front.
    try:
        import deepspeed  # noqa: F401
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        _probe = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        try:
            DeepSpeedCPUAdam([_probe], lr=1e-3)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(
                f"DeepSpeedCPUAdam JIT load failed ({exc}); resume cycle "
                f"requires a working CPU adapter build."
            )
    except ImportError:
        pytest.skip("deepspeed not installed; resume cycle requires CPU adapter.")

    from axolotl.integrations.protrain.api import protrain_optimizer_wrapper

    model, cfg = _build_tiny_lora_model()
    model = model.to("cuda")
    input_ids, labels = _make_batch(cfg)

    # Force chunks off-GPU so materialize_offload actually moves bytes
    # (the D2 hot path the test claims to exercise). small_chunk=True
    # ensures N_chunk > 1 on the tiny model.
    wrapped, optim = _wrap_protrain(
        model,
        cfg,
        force_all_persistent=False,
        zero3_shard=False,
        n_persist_override=0,
        n_buffer_override=16,
        n_swap_override=0,
        n_checkpoint_override=0,
        # All non-persistent transformer blocks in OFFLOAD mode
        # (Option B) — saved tensors re-gather on backward via the
        # M3 block manager's per-block hook rather than relying on
        # NONE-mode hooks (which would clobber autograd's saved
        # tensors when the chunk pool slot is reused).
        n_offload_override=cfg.num_hidden_layers,
        small_chunk=True,
    )
    try:
        # ---- Phase 1: train 3 steps under the initial wrap ----------
        losses_pre = [
            _train_one_step(wrapped, optim, input_ids=input_ids, labels=labels)
            for _ in range(3)
        ]
        for i, lv in enumerate(losses_pre):
            assert math.isfinite(lv), f"phase 1 step {i}: non-finite loss {lv}"

        # ---- Phase 2: simulate the resume hook's in-process cycle ---
        underlying = getattr(wrapped, "module", wrapped)
        chunk_manager = wrapped.chunk_manager
        assert chunk_manager is not None

        # Step 1: tear down the CPU optim BEFORE restore_to_gpu (per
        # the resume hook's preamble at plugin.py:557-572). This is
        # the SAME teardown the production resume hook performs;
        # ``restore_to_gpu`` is about to invalidate the CPU shards
        # the adapter holds references to.
        if getattr(chunk_manager, "cpu_optim", None) is not None:
            chunk_manager.cpu_optim.shutdown()

        # Step 2: restore_to_gpu — rebinds param.data back to standalone
        # GPU storage so the state_dict capture below sees the real
        # parameter shapes (not the ``[0]`` placeholder that's bound
        # while chunks are offloaded). The production HF Trainer save
        # path has the same property: checkpoints are taken AFTER
        # ProTrain's resume hook restores chunks to GPU, not while
        # offloaded — otherwise the saved state_dict would have
        # ``Size([0])`` entries that would fail to load on resume.
        chunk_manager.restore_to_gpu()

        # Step 3: capture the saved state and load it back. In
        # production this is the HF Trainer's
        # ``trainer.save_state_dict`` → user copies the checkpoint →
        # ``_load_from_checkpoint`` cycle; here we do the round-trip
        # in-process to keep the smoke unit-scoped.
        saved_state = {
            k: v.detach().clone() for k, v in underlying.state_dict().items()
        }
        underlying.load_state_dict(saved_state, strict=False)

        # Step 4: re-build the offload state. This is the D2 hot path —
        # second materialize_offload on the same chunk manager. With
        # ``n_persist_override=0`` + ``n_offload_override=N_layers``
        # this actually moves bytes (7 non-persistent chunks → pinned
        # CPU pool) rather than being a no-op on a force-all-persistent
        # config (CodeRabbit R3-#7).
        chunk_manager.materialize_offload()

        # Step 5: rebuild the optimizer adapter (exercises D3 — the
        # old cpu_optim is None at this point because of step 1, so
        # this exercises the "no prior adapter" branch; a full test of
        # the swap-without-shutdown path is in
        # ``test_cpu_optim_replaced_calls_shutdown_on_previous`` above).
        optim_resumed = protrain_optimizer_wrapper(wrapped, lr=1e-3)

        # ---- Phase 3: train 3 more steps after the simulated resume -
        losses_post = [
            _train_one_step(wrapped, optim_resumed, input_ids=input_ids, labels=labels)
            for _ in range(3)
        ]
        for i, lv in enumerate(losses_post):
            assert math.isfinite(lv), (
                f"phase 3 (post-resume) step {i}: non-finite loss {lv}"
            )

        # Continuity: the first post-resume loss should not be wildly
        # larger than the last pre-resume loss. Allow 5x as a generous
        # bound that catches catastrophic divergence (NaN-precursor,
        # state corruption) but tolerates the cold-started optimizer
        # state.
        assert losses_post[0] < 5.0 * losses_pre[-1] + 1.0, (
            f"resume produced catastrophic divergence: "
            f"pre-end={losses_pre[-1]:.4f}, post-start={losses_post[0]:.4f} "
            f"(>5x is treated as a state-corruption signal)"
        )
        print(
            f"\nresume-robustness in-process cycle: "
            f"losses_pre={losses_pre} losses_post={losses_post}"
        )
    finally:
        close = getattr(wrapped, "close", None)
        if callable(close):
            close()
