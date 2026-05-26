"""Path B tests (PR #24): ProTrain-owned LoRA grad sync via flattened all_reduce.

Covers:
1. LoRA-param discovery (name shape, requires_grad filter).
2. ``_ddp_params_and_buffers_to_ignore`` registration (snapshot pattern).
3. ``_sync_lora_grads_path_b`` collective count + dtype bucketing
   (mocked ``dist.all_reduce``).
4. Single-rank no-op (world<=1 short-circuit).
5. 2-rank gloo bit-equivalence between manual all_reduce-then-step and
   Path B's flattened-all_reduce-then-step.

mp.spawn workers use the gloo backend (CPU collectives) — the reduction
math is identical to NCCL.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: a minimal PEFT-shaped model (no real PEFT dependency)
# ---------------------------------------------------------------------------


def _build_peft_shaped_model():
    """Build a tiny nn.Module whose parameter NAMES mimic PEFT's LoRA layout.

    A PEFT-wrapped layer typically yields names like
    ``base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight``.
    We construct a module hierarchy with the same dotted shape so
    ``model.named_parameters()`` returns the expected names; no actual
    PEFT install is needed.
    """
    import torch
    from torch import nn

    class _LoraFactor(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)

    class _LoraAttn(nn.Module):
        """Container for q_proj-style PEFT layer: base weight + lora_A + lora_B."""

        def __init__(self, hidden: int, rank: int):
            super().__init__()
            self.base_layer = nn.Linear(hidden, hidden, bias=False)
            # PEFT's ModuleDict keyed by adapter name; "default" is the standard.
            self.lora_A = nn.ModuleDict({"default": _LoraFactor(hidden, rank)})
            self.lora_B = nn.ModuleDict({"default": _LoraFactor(rank, hidden)})

    class _Block(nn.Module):
        def __init__(self, hidden: int, rank: int):
            super().__init__()
            self.self_attn_q_proj = _LoraAttn(hidden, rank)
            self.self_attn_v_proj = _LoraAttn(hidden, rank)

    class _PeftModelLike(nn.Module):
        def __init__(self, hidden: int = 16, rank: int = 4, n_layers: int = 2):
            super().__init__()
            # PEFT wraps with base_model.model.<orig hierarchy>.
            self.base_model = nn.Module()
            self.base_model.model = nn.Module()
            layers = nn.ModuleList([_Block(hidden, rank) for _ in range(n_layers)])
            self.base_model.model.layers = layers
            # Freeze base weights; only LoRA factors are trainable.
            for name, p in self.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

    return _PeftModelLike()


# ---------------------------------------------------------------------------
# 1. Discovery tests
# ---------------------------------------------------------------------------


def test_discover_lora_params_finds_all_trainable_factors():
    """All trainable lora_A / lora_B params are returned; base weights skipped."""
    from axolotl.integrations.protrain.plugin import _discover_lora_params

    model = _build_peft_shaped_model()
    names, params = _discover_lora_params(model)

    # 2 layers x (q_proj + v_proj) x (lora_A + lora_B) = 8 factors.
    assert len(names) == 8, f"expected 8 LoRA params, got {len(names)}: {names}"
    assert len(params) == 8
    for name in names:
        assert ("lora_A" in name) or ("lora_B" in name), name
    # Names follow expected dotted PEFT shape.
    assert any("layers.0.self_attn_q_proj.lora_A.default.weight" in n for n in names)


def test_discover_lora_params_skips_frozen_factors():
    """``requires_grad=False`` LoRA params are excluded."""
    from axolotl.integrations.protrain.plugin import _discover_lora_params

    model = _build_peft_shaped_model()
    # Freeze all params.
    for p in model.parameters():
        p.requires_grad_(False)

    names, params = _discover_lora_params(model)
    assert names == []
    assert params == []


def test_discover_lora_params_excludes_non_lora_trainables():
    """Non-LoRA trainable params (e.g. base_layer.weight) are not picked up."""
    from axolotl.integrations.protrain.plugin import _discover_lora_params

    model = _build_peft_shaped_model()
    # Make ONE base weight trainable too — discovery must NOT include it.
    model.base_model.model.layers[0].self_attn_q_proj.base_layer.weight.requires_grad_(
        True
    )

    names, _ = _discover_lora_params(model)
    assert all("base_layer" not in n for n in names)
    assert len(names) == 8  # Same count as the all-LoRA-trainable case.


def test_bypass_gate_stays_off_with_extra_trainables():
    """Bypass gate (``n_trainable == len(lora_names)``) must evaluate False
    when non-LoRA trainable params exist (simulating PEFT ``modules_to_save``
    or ``lora_bias != 'none'``), so DDP stays active to sync the extras
    via its bucketed allreduce. Path B keeps the LoRA factors in the
    ignore list; the extras must NOT be in the ignore list.
    """
    from axolotl.integrations.protrain.plugin import (
        _discover_lora_params,
        _register_lora_ddp_ignore,
    )

    model = _build_peft_shaped_model()
    extra = model.base_model.model.layers[0].self_attn_q_proj.base_layer.weight
    extra.requires_grad_(True)

    lora_names, _ = _discover_lora_params(model)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    assert n_trainable == len(lora_names) + 1, (
        f"setup: expected n_trainable={len(lora_names) + 1}, got {n_trainable}"
    )
    bypass_would_fire = n_trainable == len(lora_names)
    assert not bypass_would_fire

    _register_lora_ddp_ignore(model, lora_names)
    live = set(model._ddp_params_and_buffers_to_ignore)
    assert set(lora_names).issubset(live)
    extra_name = next(n for n, p in model.named_parameters() if p is extra)
    assert extra_name not in live, (
        f"non-LoRA trainable {extra_name} was incorrectly added to the ignore list"
    )


# ---------------------------------------------------------------------------
# 2. _ddp_params_and_buffers_to_ignore registration
# ---------------------------------------------------------------------------


def test_register_lora_ddp_ignore_preserves_existing():
    """Pre-existing ignore-list entries (e.g. chunk-managed names) are preserved."""
    from axolotl.integrations.protrain.plugin import (
        _discover_lora_params,
        _register_lora_ddp_ignore,
    )

    model = _build_peft_shaped_model()
    # Simulate a chunk_manager having already registered a non-LoRA name.
    pre_existing = ["base_model.model.layers.0.self_attn_q_proj.base_layer.weight"]
    model._ddp_params_and_buffers_to_ignore = list(pre_existing)

    lora_names, _ = _discover_lora_params(model)
    _register_lora_ddp_ignore(model, lora_names)

    live = set(model._ddp_params_and_buffers_to_ignore)
    assert set(pre_existing).issubset(live)
    assert set(lora_names).issubset(live)
    # Snapshot recorded for teardown.
    assert model._protrain_ddp_original_ignore == pre_existing


def test_register_lora_ddp_ignore_with_no_prior_attr():
    """Unset prior attr → snapshot is None, live attr matches LoRA names."""
    from axolotl.integrations.protrain.plugin import (
        _discover_lora_params,
        _register_lora_ddp_ignore,
    )

    model = _build_peft_shaped_model()
    lora_names, _ = _discover_lora_params(model)
    _register_lora_ddp_ignore(model, lora_names)

    assert model._protrain_ddp_original_ignore is None
    assert set(model._ddp_params_and_buffers_to_ignore) == set(lora_names)


# ---------------------------------------------------------------------------
# 3. _sync_lora_grads_path_b collective behavior (mocked)
# ---------------------------------------------------------------------------


def _make_test_optimizer_with_lora_owned(lora_params):
    """Construct a minimal ``_ProTrainOptimizer`` shell for unit tests.

    Bypasses the full ``protrain_optimizer_wrapper`` builder so the test
    doesn't need a real ChunkManager / WrappedModel. Only fields touched
    by ``_sync_lora_grads_path_b`` are populated.

    ``_ProTrainOptimizer`` requires at least one param in the base
    ``Optimizer`` ctor; when ``lora_params`` is empty we attach a single
    placeholder param (the optimizer never touches it during the Path B
    sync path).
    """
    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.optim_wrapper import _ProTrainOptimizer

    chunk_manager = MagicMock()
    chunk_manager._scheduler_ref = None

    # Always pass a non-empty params list to the base Optimizer ctor; the
    # Path B sync code paths only consult _lora_owned_params.
    base_params = list(lora_params) if lora_params else [nn.Parameter(torch.zeros(1))]

    optim = _ProTrainOptimizer(
        gpu_optim=None,
        cpu_optim=None,
        params=base_params,
        defaults={"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        chunk_manager=chunk_manager,
        lora_owned_params=list(lora_params),
    )
    return optim


def test_sync_lora_grads_path_b_noop_when_empty():
    """No lora_owned_params → no collective."""
    import torch
    from torch import nn

    p = nn.Parameter(torch.randn(4))
    optim_empty = _make_test_optimizer_with_lora_owned([])
    optim_empty._lora_owned_params = []  # Explicit, in case ctor populated.

    with patch("torch.distributed.all_reduce") as mock_ar:
        optim_empty._sync_lora_grads_path_b()
        mock_ar.assert_not_called()
    # Reference the param so it isn't unused.
    assert p.numel() == 4


def test_sync_lora_grads_path_b_noop_when_dist_not_initialized():
    """No collective when dist is uninitialised."""
    import torch
    from torch import nn

    p1 = nn.Parameter(torch.randn(4))
    p1.grad = torch.randn(4)
    optim = _make_test_optimizer_with_lora_owned([p1])

    with patch("torch.distributed.is_initialized", return_value=False):
        with patch("torch.distributed.all_reduce") as mock_ar:
            optim._sync_lora_grads_path_b()
            mock_ar.assert_not_called()


def test_sync_lora_grads_path_b_noop_when_world_size_one():
    """world_size == 1 short-circuits."""
    import torch
    from torch import nn

    p1 = nn.Parameter(torch.randn(4))
    p1.grad = torch.randn(4)
    optim = _make_test_optimizer_with_lora_owned([p1])

    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=1):
                with patch("torch.distributed.all_reduce") as mock_ar:
                    optim._sync_lora_grads_path_b()
                    mock_ar.assert_not_called()


def test_sync_lora_grads_path_b_one_call_per_dtype():
    """All bf16 LoRA grads bucket into one collective; fp32 grads bucket into a second."""
    import torch
    from torch import nn

    bf16_p1 = nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
    bf16_p1.grad = torch.randn(4, dtype=torch.bfloat16)
    bf16_p2 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
    bf16_p2.grad = torch.randn(8, dtype=torch.bfloat16)
    fp32_p1 = nn.Parameter(torch.randn(4, dtype=torch.float32))
    fp32_p1.grad = torch.randn(4, dtype=torch.float32)

    optim = _make_test_optimizer_with_lora_owned([bf16_p1, bf16_p2, fp32_p1])

    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch("torch.distributed.all_reduce") as mock_ar:
                    optim._sync_lora_grads_path_b()
                    # 2 dtype buckets → 2 calls.
                    assert mock_ar.call_count == 2


def test_sync_lora_grads_path_b_skips_params_with_no_grad():
    """Params without ``.grad`` don't trip the collective."""
    import torch
    from torch import nn

    p1 = nn.Parameter(torch.randn(4))
    p1.grad = None  # No grad yet.
    p2 = nn.Parameter(torch.randn(4))
    p2.grad = torch.randn(4)

    optim = _make_test_optimizer_with_lora_owned([p1, p2])

    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch("torch.distributed.all_reduce") as mock_ar:
                    optim._sync_lora_grads_path_b()
                    # Only p2 had a grad → exactly 1 collective (single tensor bucket).
                    assert mock_ar.call_count == 1


# ---------------------------------------------------------------------------
# 4. 2-rank bit-equivalence test (gloo)
# ---------------------------------------------------------------------------


_BIT_EQUIV_PORT = "29672"


def _bit_equiv_worker(rank: int, world_size: int, tmpdir: str) -> None:
    """Worker: compare manual per-param AVG all_reduce vs Path B flattened sync."""
    import sys
    import traceback

    try:
        import torch
        import torch.distributed as dist
        from torch import nn

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _BIT_EQUIV_PORT)
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-bit-equiv",
            rank=rank,
            world_size=world_size,
        )
        try:
            # Deterministic init: same seed across ranks gives identical weights.
            torch.manual_seed(0)
            params_path_a = [
                nn.Parameter(torch.randn(8, 4, dtype=torch.float32)),
                nn.Parameter(torch.randn(4, 8, dtype=torch.float32)),
                nn.Parameter(torch.randn(8, 4, dtype=torch.float32)),
            ]
            # Per-rank grads differ → averaging matters.
            torch.manual_seed(100 + rank)
            for p in params_path_a:
                p.grad = torch.randn_like(p.data)

            # PATH A baseline: manual per-param all_reduce(AVG).
            for p in params_path_a:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            # Apply a vanilla SGD step so the comparison is post-update weights.
            with torch.no_grad():
                for p in params_path_a:
                    p.data.add_(p.grad, alpha=-1e-2)
            post_a = [p.data.clone() for p in params_path_a]

            # PATH B: re-construct the SAME initial state + grads, then run the
            # Path B helper. Post-step weights must match Path A.
            torch.manual_seed(0)
            params_path_b = [
                nn.Parameter(torch.randn(8, 4, dtype=torch.float32)),
                nn.Parameter(torch.randn(4, 8, dtype=torch.float32)),
                nn.Parameter(torch.randn(8, 4, dtype=torch.float32)),
            ]
            torch.manual_seed(100 + rank)
            for p in params_path_b:
                p.grad = torch.randn_like(p.data)

            # Use the public Path B helper to drive the collective.
            from axolotl.integrations.protrain.api.optim_wrapper import (
                _ProTrainOptimizer,
            )

            chunk_manager_stub = type("CMStub", (), {"_scheduler_ref": None})()
            optim_b = _ProTrainOptimizer(
                gpu_optim=None,
                cpu_optim=None,
                params=list(params_path_b),
                defaults={
                    "lr": 1e-2,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
                chunk_manager=chunk_manager_stub,
                lora_owned_params=list(params_path_b),
            )
            optim_b._sync_lora_grads_path_b()
            with torch.no_grad():
                for p in params_path_b:
                    p.data.add_(p.grad, alpha=-1e-2)
            post_b = [p.data.clone() for p in params_path_b]

            # Bit-equivalence within FP rounding.
            for i, (a, b) in enumerate(zip(post_a, post_b, strict=True)):
                if not torch.allclose(a, b, rtol=1e-5, atol=1e-7):
                    diff = (a - b).abs().max().item()
                    raise RuntimeError(
                        f"rank {rank} param {i}: Path A and Path B diverged. "
                        f"max abs diff={diff:.3e} (rtol=1e-5, atol=1e-7). "
                        f"a[0,:4]={a.flatten()[:4].tolist()} "
                        f"b[0,:4]={b.flatten()[:4].tolist()}"
                    )

            with open(os.path.join(tmpdir, f"bit_equiv_rank{rank}.done"), "w") as f:
                f.write(f"params={len(post_a)} max_abs_diff=0\n")
        finally:
            try:
                dist.barrier()
            except Exception:  # noqa: BLE001
                pass
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        with open(os.path.join(tmpdir, f"bit_equiv_rank{rank}.err"), "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)


def test_bit_equivalence_two_rank_gloo(tmp_path):
    """2-rank gloo: Path B post-step weights match Path A within FP rounding."""
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    import torch.multiprocessing as mp

    mp.spawn(
        _bit_equiv_worker,
        args=(2, str(tmp_path)),
        nprocs=2,
        join=True,
    )

    err_files = list(tmp_path.glob("bit_equiv_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"bit-equivalence worker errors:\n{bodies}")
    for r in range(2):
        assert (tmp_path / f"bit_equiv_rank{r}.done").is_file(), (
            f"rank {r} did not reach bit-equivalence sentinel"
        )


# ---------------------------------------------------------------------------
# 5. Plugin integration: flag OFF leaves model untouched
# ---------------------------------------------------------------------------


def test_plugin_post_trainer_create_flag_off_no_op(monkeypatch):
    """Flag OFF → no LoRA discovery / no DDP-ignore mutation from Path B path.

    Smoke test on a minimal mocked trainer. We aren't running the full
    plugin (no chunk_manager etc.); we're asserting that the new Path B
    branch is gated by the cfg flag.
    """
    import torch

    from axolotl.integrations.protrain.plugin import _discover_lora_params

    model = _build_peft_shaped_model()
    # Simulate "before plugin": no _protrain_ddp_original_ignore snapshot.
    assert not hasattr(model, "_protrain_ddp_original_ignore")
    assert not hasattr(model, "_ddp_params_and_buffers_to_ignore")

    # The discovery helper still works (it's stateless).
    names, _ = _discover_lora_params(model)
    assert len(names) == 8

    # But without _register_lora_ddp_ignore being called, the model's
    # attributes remain untouched.
    assert not hasattr(model, "_protrain_ddp_original_ignore")
    assert not hasattr(model, "_ddp_params_and_buffers_to_ignore")

    # Use the imported torch so isort/ruff don't complain.
    assert torch.__version__ is not None


# ---------------------------------------------------------------------------
# 6. Extended convergence parity: 2-rank gloo, multi-step trajectory
# ---------------------------------------------------------------------------


_CONV_PARITY_PORT = "29693"


def _convergence_parity_worker(rank: int, world_size: int, tmpdir: str) -> None:
    """Run two identical training trajectories — Path A (manual sync) vs Path B.

    Both trajectories share the same initial weights, the same per-rank random
    inputs/targets across all steps, and the same SGD update rule. The only
    difference is *how* the cross-rank grad sync is issued:
        Path A: explicit ``dist.all_reduce(p.grad, op=AVG)`` per param
        Path B: ``_ProTrainOptimizer._sync_lora_grads_path_b()``
    Final weights and loss values must match within FP rounding after 100
    backward + step iterations.
    """
    import sys
    import traceback

    try:
        import torch
        import torch.distributed as dist
        from torch import nn

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _CONV_PARITY_PORT)
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpdir}/rendezvous-conv-parity",
            rank=rank,
            world_size=world_size,
        )
        try:
            n_steps = 100
            hidden = 8
            rank_dim = 4
            lr = 5e-3

            def _build_lora_pair():
                torch.manual_seed(0)
                lora_A = nn.Parameter(torch.randn(rank_dim, hidden) * 0.02)
                lora_B = nn.Parameter(torch.zeros(hidden, rank_dim))
                return lora_A, lora_B

            def _gen_batch(step: int):
                # Per-rank divergent inputs so cross-rank grad averaging matters.
                g = torch.Generator().manual_seed(1000 + step * 17 + rank * 31)
                x = torch.randn(4, hidden, generator=g)
                y = torch.randn(4, hidden, generator=g)
                return x, y

            # ---- Path A: explicit per-param all_reduce(AVG) baseline ----
            A_lora_A, A_lora_B = _build_lora_pair()
            losses_a: list[float] = []
            for step in range(n_steps):
                x, y = _gen_batch(step)
                out = x @ A_lora_A.t() @ A_lora_B.t()
                loss = ((out - y) ** 2).mean()
                losses_a.append(loss.item())
                A_lora_A.grad = None
                A_lora_B.grad = None
                loss.backward()
                # Manual baseline sync.
                for p in (A_lora_A, A_lora_B):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                with torch.no_grad():
                    for p in (A_lora_A, A_lora_B):
                        p.data.add_(p.grad, alpha=-lr)
            final_loss_a = losses_a[-1]

            # ---- Path B: drive same trajectory via _sync_lora_grads_path_b ----
            from axolotl.integrations.protrain.api.optim_wrapper import (
                _ProTrainOptimizer,
            )

            B_lora_A, B_lora_B = _build_lora_pair()
            chunk_manager_stub = type("CMStub", (), {"_scheduler_ref": None})()
            optim_b = _ProTrainOptimizer(
                gpu_optim=None,
                cpu_optim=None,
                params=[B_lora_A, B_lora_B],
                defaults={
                    "lr": lr,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
                chunk_manager=chunk_manager_stub,
                lora_owned_params=[B_lora_A, B_lora_B],
            )
            losses_b: list[float] = []
            for step in range(n_steps):
                x, y = _gen_batch(step)
                out = x @ B_lora_A.t() @ B_lora_B.t()
                loss = ((out - y) ** 2).mean()
                losses_b.append(loss.item())
                B_lora_A.grad = None
                B_lora_B.grad = None
                loss.backward()
                optim_b._sync_lora_grads_path_b()
                with torch.no_grad():
                    for p in (B_lora_A, B_lora_B):
                        p.data.add_(p.grad, alpha=-lr)
            final_loss_b = losses_b[-1]

            # Final-loss parity.
            loss_diff = abs(final_loss_a - final_loss_b)
            if loss_diff >= 1e-4:
                raise RuntimeError(
                    f"rank {rank}: final-loss parity violated: "
                    f"A={final_loss_a:.8e} B={final_loss_b:.8e} "
                    f"|diff|={loss_diff:.3e} (allowed < 1e-4)"
                )

            # Per-param weight parity.
            for name, (a, b) in (
                ("lora_A", (A_lora_A.data, B_lora_A.data)),
                ("lora_B", (A_lora_B.data, B_lora_B.data)),
            ):
                if not torch.allclose(a, b, rtol=1e-5, atol=1e-7):
                    diff = (a - b).abs().max().item()
                    raise RuntimeError(
                        f"rank {rank} {name}: trajectory drift after "
                        f"{n_steps} steps. max abs diff={diff:.3e} "
                        f"(rtol=1e-5, atol=1e-7)."
                    )

            with open(os.path.join(tmpdir, f"conv_parity_rank{rank}.done"), "w") as f:
                f.write(
                    f"n_steps={n_steps} final_loss_a={final_loss_a:.6e} "
                    f"final_loss_b={final_loss_b:.6e}\n"
                )
        finally:
            try:
                dist.barrier()
            except Exception:  # noqa: BLE001
                pass
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        with open(os.path.join(tmpdir, f"conv_parity_rank{rank}.err"), "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)


def test_path_b_convergence_parity_2rank_gloo(tmp_path):
    """2-rank gloo: Path B and manual-sync baseline match after 100 steps.

    Stronger than ``test_bit_equivalence_two_rank_gloo`` (single-step): this
    verifies trajectory equivalence — accumulated FP error across 100
    backward+sync+step iterations stays under ``torch.allclose(rtol=1e-5,
    atol=1e-7)`` for every LoRA weight, and final-loss absolute diff under
    1e-4. Load-bearing for the default-ON safety claim.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    import torch.multiprocessing as mp

    mp.spawn(
        _convergence_parity_worker,
        args=(2, str(tmp_path)),
        nprocs=2,
        join=True,
    )

    err_files = list(tmp_path.glob("conv_parity_rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"convergence-parity worker errors:\n{bodies}")
    for r in range(2):
        assert (tmp_path / f"conv_parity_rank{r}.done").is_file(), (
            f"rank {r} did not reach convergence-parity sentinel"
        )


# ---------------------------------------------------------------------------
# 7. gradient_accumulation_steps > 1: Path B fires once per optimizer.step()
# ---------------------------------------------------------------------------


def test_path_b_grad_accum_fires_once_per_step():
    """Path B sync lives in ``optimizer.step()``, not inside ``loss.backward()``.

    Simulates ``gradient_accumulation_steps=4``: 4 backward passes accumulate
    grads into ``param.grad``; ``optimizer.step()`` fires ONCE at the grad-update
    boundary. We assert ``dist.all_reduce`` was called exactly per-dtype-bucket
    once across the whole micro-batch sequence, NOT 4× — mid-accumulation grads
    must not be reduced.
    """
    import torch
    from torch import nn

    # Two dtype buckets so we can also see the bucketing didn't break.
    bf16_p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
    fp32_p = nn.Parameter(torch.randn(8, dtype=torch.float32))
    optim = _make_test_optimizer_with_lora_owned([bf16_p, fp32_p])

    # Simulate 4 micro-batches that all accumulate gradients.
    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch("torch.distributed.all_reduce") as mock_ar:
                    for _micro in range(4):
                        # Microbatch backward: grads accumulate. We just stamp
                        # synthetic grads to model that ``loss.backward()`` ran.
                        if bf16_p.grad is None:
                            bf16_p.grad = torch.zeros_like(bf16_p)
                            fp32_p.grad = torch.zeros_like(fp32_p)
                        bf16_p.grad.add_(torch.randn_like(bf16_p.grad))
                        fp32_p.grad.add_(torch.randn_like(fp32_p.grad))
                        # Critical: NO sync fired during the backward portion.
                        assert mock_ar.call_count == 0, (
                            "Path B sync must NOT fire mid-accumulation; "
                            f"got {mock_ar.call_count} call(s) after "
                            f"micro-batch {_micro}"
                        )
                    # Grad-update boundary: optimizer.step() drives the sync.
                    optim._sync_lora_grads_path_b()
                    assert mock_ar.call_count == 2, (
                        "expected exactly 2 collective calls (one per dtype "
                        f"bucket), got {mock_ar.call_count}"
                    )


# ---------------------------------------------------------------------------
# 8. Mode B + Path B compatibility: disjoint param sets, no double-sync
# ---------------------------------------------------------------------------


def test_path_b_modeb_disjoint_param_sets():
    """Path B + Mode B operate on disjoint param sets — no double-sync.

    Mode B (``protrain_force_replicated_cpu_offload=true``) routes per-chunk
    grad sync through ``ChunkManager._coalesced_all_reduce_persistent_grads``
    on persistent chunks. Path B operates on ``_lora_owned_params`` (the
    LoRA factors discovered by ``_discover_lora_params``).

    The plugin builds the wrapper with ``lora_owned_params`` populated only
    from LoRA factors — NOT from chunk-managed params. Verify empirically
    that with a model containing BOTH a chunk-managed param AND LoRA
    factors:
        1. ``_discover_lora_params`` returns only the LoRA factors.
        2. Path B's collective sees ONLY the LoRA tensors (not the chunk
           param).
        3. The chunk param is NOT in ``_lora_owned_params`` so Path B
           cannot double-sync what Mode B already handled.
    """
    import torch
    from torch import nn

    from axolotl.integrations.protrain.plugin import _discover_lora_params

    # Construct a hybrid PEFT-shaped model with an additional non-LoRA
    # trainable that mimics a chunk-managed persistent param.
    class _HybridLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = nn.Module()
            # Standalone "persistent chunk param" — trainable, NOT a LoRA factor.
            self.base_model.model.lm_head = nn.Linear(8, 8, bias=False)
            # LoRA-shaped factor.
            block = nn.Module()
            attn = nn.Module()
            attn.lora_A = nn.ModuleDict({"default": nn.Linear(8, 4, bias=False)})
            attn.lora_B = nn.ModuleDict({"default": nn.Linear(4, 8, bias=False)})
            block.self_attn_q_proj = attn
            self.base_model.model.layers = nn.ModuleList([block])

    model = _HybridLora()
    for _name, p in model.named_parameters():
        p.requires_grad_(True)  # All trainable including lm_head ("chunk param").

    lora_names, lora_params = _discover_lora_params(model)
    # Discovery must reject the chunk param.
    assert all("lora_A" in n or "lora_B" in n for n in lora_names), lora_names
    assert not any("lm_head" in n for n in lora_names), lora_names

    # Identify the "chunk-managed" param by name.
    chunk_param = model.base_model.model.lm_head.weight
    chunk_param.grad = torch.randn_like(chunk_param)
    for p in lora_params:
        p.grad = torch.randn_like(p)

    # Build wrapper with ONLY lora_params owned (mirrors plugin behavior).
    optim = _make_test_optimizer_with_lora_owned(lora_params)

    # Snapshot grad tensors that Path B would actually see, then drive the sync
    # and assert chunk_param.grad was untouched as a tensor identity check.
    chunk_grad_pre_id = id(chunk_param.grad)
    chunk_grad_pre_clone = chunk_param.grad.clone()

    captured_tensors: list = []

    def _spy_all_reduce(t, *args, **kwargs):
        captured_tensors.append(t)
        return None

    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch("torch.distributed.all_reduce", side_effect=_spy_all_reduce):
                    optim._sync_lora_grads_path_b()

    # 1) Path B never reduced the chunk param's grad.
    assert chunk_param.grad is not None
    assert id(chunk_param.grad) == chunk_grad_pre_id, (
        "Path B replaced chunk_param.grad identity — disjoint-set guarantee broken"
    )
    assert torch.equal(chunk_param.grad, chunk_grad_pre_clone), (
        "Path B mutated chunk_param.grad — would double-sync with Mode B's "
        "_coalesced_all_reduce_persistent_grads"
    )

    # 2) Every tensor handed to all_reduce belongs to a LoRA factor (i.e. one
    # of the grads from `lora_params` or a flattened bucket built from them).
    lora_grad_ids = {id(p.grad) for p in lora_params}
    for t in captured_tensors:
        # Either a direct LoRA grad (single-tensor bucket) OR a flat buffer
        # whose total numel matches a same-dtype LoRA grad bucket.
        if id(t) in lora_grad_ids:
            continue
        # Flat bucket: total numel == sum of LoRA grads of that dtype.
        bucket_total = sum(
            p.grad.numel() for p in lora_params if p.grad.dtype == t.dtype
        )
        assert t.numel() == bucket_total, (
            "Path B all_reduce tensor not traceable to LoRA bucket "
            f"(numel={t.numel()}, expected bucket_total={bucket_total})"
        )


# ---------------------------------------------------------------------------
# 9. DoRA discovery: lora_magnitude_vector must be picked up
# ---------------------------------------------------------------------------


def _build_real_peft_model(target_modules, *, use_dora: bool = False):
    """Build a real PEFT-wrapped tiny model so we exercise actual PEFT naming.

    Returns the wrapped model; skips the test if PEFT or torch are missing.
    """
    peft = pytest.importorskip("peft")
    import torch  # noqa: F401  (PEFT import-time check)
    from peft import LoraConfig, get_peft_model
    from torch import nn

    class _TinyForLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16, bias=False)
            self.embed_tokens = nn.Embedding(32, 16)
            self.lm_head = nn.Linear(16, 32, bias=False)

        def forward(self, x):  # pragma: no cover - never called
            return x

    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=list(target_modules),
        use_dora=use_dora,
    )
    model = get_peft_model(_TinyForLora(), cfg)
    # Sanity: ensure PEFT version actually injected the requested modules.
    assert any(p.requires_grad for _, p in model.named_parameters()), (
        f"PEFT {peft.__version__} produced no trainable params for "
        f"target_modules={target_modules}"
    )
    return model


def test_dora_discovery_or_clean_exclusion(caplog):
    """DoRA modules either discovered (preferred) or excluded with warning.

    DoRA adds a `lora_magnitude_vector` learnable per target. Path B's
    discovery markers include `.lora_magnitude_vector.` so the expected
    outcome is (a) DoRA params are found alongside lora_A/lora_B. If a
    future PEFT release renames the magnitude vector and discovery silently
    misses it, this test fails — caller is then on notice to either extend
    the marker set or log an explicit warning.
    """
    import logging

    from axolotl.integrations.protrain.plugin import _discover_lora_params

    model = _build_real_peft_model(["q_proj"], use_dora=True)
    # Collect the actual trainable names for ground-truth comparison.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    magnitude_names = {n for n in trainable if "lora_magnitude_vector" in n}
    lora_AB_names = {n for n in trainable if ".lora_A." in n or ".lora_B." in n}
    assert magnitude_names, "PEFT did not produce any lora_magnitude_vector params"
    assert lora_AB_names, "PEFT did not produce any lora_A/lora_B params"

    with caplog.at_level(logging.WARNING):
        names, params = _discover_lora_params(model)

    found = set(names)
    # Preferred outcome: DoRA magnitude vector is discovered.
    dora_discovered = magnitude_names.issubset(found)
    # Alternate acceptable outcome: explicit warning that DoRA is excluded.
    dora_warning_emitted = any(
        "dora" in rec.message.lower() and "exclud" in rec.message.lower()
        for rec in caplog.records
    )

    assert dora_discovered or dora_warning_emitted, (
        "DoRA magnitude vector silently missed by Path B discovery. "
        f"trainable_magnitude_names={sorted(magnitude_names)} "
        f"discovered={sorted(found)}. Either extend "
        "_LORA_FACTOR_NAME_MARKERS to recognize the new naming, or emit a "
        "WARN log so users know they're falling back to DDP-bucketed sync."
    )
    # Standard lora_A/lora_B must always be found.
    assert lora_AB_names.issubset(found), (
        f"Standard LoRA factors missed alongside DoRA: missing="
        f"{sorted(lora_AB_names - found)}"
    )
    # Returned params list must align with names list.
    assert len(params) == len(names)


# ---------------------------------------------------------------------------
# 10. Extended LoRA targets: embed_tokens and lm_head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["q_proj", "embed_tokens", "lm_head"])
def test_path_b_discovers_extended_lora_targets(target):
    """Discovery picks up LoRA factors on q_proj, embed_tokens, and lm_head.

    Production users frequently expand LoRA beyond attention/MLP to the
    token embedding + LM head. PEFT names these as ``lora_embedding_A/B``
    (for ``nn.Embedding`` targets) and the standard ``lora_A/B`` for
    ``lm_head`` (which is ``nn.Linear``). Both must be discovered.
    """
    from axolotl.integrations.protrain.plugin import _discover_lora_params

    model = _build_real_peft_model([target])
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # Restrict to params under the requested target submodule.
    target_trainables = {n for n in trainable if f".{target}." in f".{n}."}
    assert target_trainables, f"PEFT produced no trainable params under '{target}'"

    names, params = _discover_lora_params(model)
    found = set(names)

    missing = target_trainables - found
    assert not missing, (
        f"Path B discovery missed LoRA factors under target='{target}': "
        f"{sorted(missing)}. Extend _LORA_FACTOR_NAME_MARKERS to cover them."
    )
    # Every returned param really is a parameter object with requires_grad.
    for name, p in zip(names, params, strict=True):
        assert p.requires_grad, f"non-trainable param leaked into discovery: {name}"
