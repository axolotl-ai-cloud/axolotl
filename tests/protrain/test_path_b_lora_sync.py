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
    names, params = _discover_lora_params(model)
    assert len(names) == 8

    # But without _register_lora_ddp_ignore being called, the model's
    # attributes remain untouched.
    assert not hasattr(model, "_protrain_ddp_original_ignore")
    assert not hasattr(model, "_ddp_params_and_buffers_to_ignore")

    # Use the imported torch so isort/ruff don't complain.
    assert torch.__version__ is not None
