"""In-process rebuild lifecycle invariants: DDP ignore rebuilds from snapshot, CPU adapter shuts down before swap, stale skip-state clears on non-shape-preserving rewrap."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest


def _build_tiny_lora_model():
    """Minimal LoRA-on-Llama setup small enough for the chunk manager + searcher to fit on any test rig."""
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
    """Wrap a model in ProTrain; small_chunk + overrides let tests force the CPU-adapter / non-persistent paths the searcher would otherwise skip."""
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


def test_restore_to_gpu_removes_only_chunk_owned_ddp_ignore_names() -> None:
    """restore_to_gpu clears chunk-owned DDP ignores while preserving caller entries."""
    pytest.importorskip("torch")
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
    from axolotl.integrations.protrain.types import ChunkId, ChunkLayout, ParamId

    model = nn.Linear(2, 2, bias=False)
    pid = ParamId("weight")
    layout = ChunkLayout(
        S_chunk=64,
        N_chunk=1,
        chunks=((pid,),),
        param_to_chunk={pid: ChunkId(0)},
        block_to_chunks={},
    )
    host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
    pool = BufferPool(
        n_buffer=1,
        S_chunk=layout.S_chunk,
        pinned_host=host,
        device=torch.device("cpu"),
    )
    model._ddp_params_and_buffers_to_ignore = ["caller"]  # type: ignore[attr-defined]
    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=0,
        buffer_pool=pool,
        device=torch.device("cpu"),
        shape_preserving_placeholders=True,
    )

    try:
        mgr.materialize_offload()
        assert "weight" in set(model._ddp_params_and_buffers_to_ignore)  # type: ignore[attr-defined]
        model._ddp_params_and_buffers_to_ignore.append("external_after")  # type: ignore[attr-defined]

        mgr.restore_to_gpu()

        live = set(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
        assert live == {"caller", "external_after"}
        assert not hasattr(model, "_protrain_ddp_original_ignore")
        assert not hasattr(model, "_protrain_ddp_ignore_owners")
    finally:
        mgr.close()
        host.close()


def test_resume_hook_rolls_back_offload_when_original_load_fails(monkeypatch) -> None:
    """A failing original load leaves the ProTrain runtime re-materialized."""
    import axolotl.integrations.protrain.api as protrain_api
    from axolotl.integrations.protrain.plugin import _install_resume_hook

    calls: list[str] = []

    class _CpuOptim:
        def shutdown(self) -> None:
            calls.append("cpu_shutdown")

    class _ChunkManager:
        _cpu_slots = {0: [object()]}
        _chunk_shards = {}

        def __init__(self) -> None:
            self.cpu_optim = _CpuOptim()
            self.gpu_optim = object()

        def restore_to_gpu(self) -> None:
            calls.append("restore")

        def materialize_offload(self) -> None:
            calls.append("materialize")

    class _Trainer:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                learning_rate=1e-3,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_epsilon=1e-8,
                weight_decay=0.01,
                optim="adamw_torch",
            )
            self.model = SimpleNamespace()
            self.optimizer = "old"

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
            calls.append("load")
            raise RuntimeError("load boom")

    def _fake_optimizer_wrapper(*args, **kwargs):
        calls.append("optim_rebuild")
        return "rebuilt"

    monkeypatch.setattr(
        protrain_api,
        "protrain_optimizer_wrapper",
        _fake_optimizer_wrapper,
    )
    trainer = _Trainer()
    wrapped = SimpleNamespace(chunk_manager=_ChunkManager())
    cfg = SimpleNamespace(
        protrain_own_lora_grad_sync=False,
        protrain_persistent_huge_param_threshold_bytes=None,
    )

    _install_resume_hook(trainer, cfg, wrapped)

    with pytest.raises(RuntimeError, match="load boom"):
        trainer._load_from_checkpoint("checkpoint")

    assert calls == ["cpu_shutdown", "restore", "load", "materialize", "optim_rebuild"]
    assert trainer.optimizer == "rebuilt"


@pytest.mark.gpu
def test_ddp_ignore_set_does_not_grow_on_repeat_materialize() -> None:
    """A second materialize_offload must not grow the DDP ignore set; rebuild from the original snapshot, do not union."""
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
    """Pre-existing _ddp_params_and_buffers_to_ignore is preserved across materialize_offload and restored on close()."""
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
    """Re-wrapping the optimizer must call shutdown() on the previous cpu_optim before installing the new one."""
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
        # OFFLOAD mode re-gathers saved tensors on backward via the per-block hook, avoiding the NONE-mode chunk-slot-reuse hazard.
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
    """Non-shape-preserving rewrap must clear stale _protrain_ddp_skip_init_sync and ignore-list state from a prior shape-preserving wrap."""
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
    """In-process resume hook cycle (restore_to_gpu, reload state_dict, re-materialize) must produce finite losses without catastrophic divergence."""
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
        # OFFLOAD mode re-gathers saved tensors on backward via the per-block hook, avoiding the NONE-mode chunk-slot-reuse hazard.
        n_offload_override=cfg.num_hidden_layers,
        small_chunk=True,
    )
    try:
        # Train 3 steps under the initial wrap.
        losses_pre = [
            _train_one_step(wrapped, optim, input_ids=input_ids, labels=labels)
            for _ in range(3)
        ]
        for i, lv in enumerate(losses_pre):
            assert math.isfinite(lv), f"phase 1 step {i}: non-finite loss {lv}"

        # Simulate the resume hook's in-process cycle.
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

        # Second materialize_offload on the same manager actually moves bytes thanks to the non-persistent overrides.
        chunk_manager.materialize_offload()

        # Rebuild the optimizer adapter; cpu_optim is None here so this exercises the "no prior adapter" branch.
        optim_resumed = protrain_optimizer_wrapper(wrapped, lr=1e-3)

        # Train 3 more steps after the simulated resume.
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
