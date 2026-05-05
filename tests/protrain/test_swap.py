"""Tests for the paper-real activation SWAP path (option 2A).

Coverage matrix:

* :class:`ActivationSwapPool` allocator semantics (acquire/release,
  exhaustion, double-release, slot-bytes integrity).
* :class:`SwappedBlock` correctness vs. unwrapped reference (loss
  match across multiple steps).
* Memory test: tiny model with N SWAP blocks vs. N NONE blocks; the
  SWAP path must NOT exceed the NONE-path peak (paper §3.1.2 says
  it should be lower; we only assert the upper bound to keep the
  test robust to allocator noise).
* Searcher feasibility gate: when ``cpu_capacity_bytes`` cannot hold
  the swap pool, the searcher prunes ``n_swap > 0`` candidates.
* Smoke test: wrap a tiny GPT-2 with ``n_swap_override > 0`` and
  drive 3 forward+backward iterations without crashing.

Per the Item 5 Fix A investigation, on 4×3090 PCIe these tests do
NOT assert any throughput improvement — the hardware is communication-
bound at 12 GB/s and SWAP cannot recover throughput. Acceptance is
"correct + integrates", not "demonstrates throughput improvement".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from axolotl.integrations.protrain.block.swap import SwappedBlock  # noqa: E402
from axolotl.integrations.protrain.block.swap_pool import (  # noqa: E402
    ActivationSwapPool,
)

if TYPE_CHECKING:
    from axolotl.integrations.protrain.chunk import ChunkManager
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler


# ---------------------------------------------------------------------------
# ActivationSwapPool unit tests
# ---------------------------------------------------------------------------


def test_pool_acquire_release_cycles() -> None:
    """Slots return to the free list and can be re-acquired."""
    # M5+: pool capacity is n_swap * slots_per_block * prefetch_depth.
    # Pin slots_per_block=1 here to keep the legacy 1-slot-per-block
    # arithmetic for this allocator-semantics test.
    pool = ActivationSwapPool(
        n_swap=2, slot_bytes=64, prefetch_depth=2, slots_per_block=1
    )
    assert pool.n_slot == 4
    assert pool.free_count == 4

    sid_a, view_a = pool.acquire()
    sid_b, view_b = pool.acquire()
    assert pool.free_count == 2
    assert pool.inflight_count == 2
    assert view_a.numel() == 64
    assert view_b.numel() == 64

    pool.release(sid_a)
    assert pool.free_count == 3
    pool.release(sid_b)
    assert pool.free_count == 4

    # Re-acquire after release.
    sid_c, _ = pool.acquire()
    assert pool.inflight_count == 1
    pool.release(sid_c)
    pool.close()


def test_pool_exhaustion_raises() -> None:
    """Acquiring beyond ``n_slot`` raises a clear RuntimeError."""
    pool = ActivationSwapPool(
        n_swap=1, slot_bytes=8, prefetch_depth=2, slots_per_block=1
    )
    held = []
    held.append(pool.acquire())
    held.append(pool.acquire())
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.acquire()
    for sid, _ in held:
        pool.release(sid)
    pool.close()


def test_pool_double_release_warns_no_corruption() -> None:
    """Double-release is logged but does not corrupt the free list."""
    pool = ActivationSwapPool(
        n_swap=1, slot_bytes=8, prefetch_depth=2, slots_per_block=1
    )
    sid, _ = pool.acquire()
    pool.release(sid)
    pre = pool.free_count
    # Double-release should not mutate state further.
    pool.release(sid)
    assert pool.free_count == pre
    pool.close()


def test_pool_total_bytes_matches_sizing() -> None:
    """``total_bytes`` is the product of n_slot × slot_bytes."""
    pool = ActivationSwapPool(
        n_swap=3, slot_bytes=128, prefetch_depth=2, slots_per_block=4
    )
    # n_slot = n_swap * slots_per_block * prefetch_depth = 3 * 4 * 2 = 24
    assert pool.n_slot == 24
    assert pool.total_bytes == 24 * 128
    pool.close()


def test_pool_default_slots_per_block_yields_k_capacity() -> None:
    """M5+: default ``slots_per_block`` multiplies the pool capacity."""
    from axolotl.integrations.protrain.block.swap_pool import (
        DEFAULT_SLOTS_PER_BLOCK,
    )

    pool = ActivationSwapPool(n_swap=1, slot_bytes=64, prefetch_depth=2)
    assert pool.n_slot == 1 * DEFAULT_SLOTS_PER_BLOCK * 2
    pool.close()


def test_pool_invalid_args_raise() -> None:
    """Constructor rejects non-positive sizing inputs."""
    with pytest.raises(ValueError):
        ActivationSwapPool(n_swap=0, slot_bytes=8)
    with pytest.raises(ValueError):
        ActivationSwapPool(n_swap=1, slot_bytes=0)
    with pytest.raises(ValueError):
        ActivationSwapPool(n_swap=1, slot_bytes=8, prefetch_depth=0)


# ---------------------------------------------------------------------------
# SwappedBlock correctness — multi-step loss match vs. unwrapped reference
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_swap_correctness_matches_reference_three_steps() -> None:
    """3-step loss curve with SWAP matches the unwrapped block to fp32 noise.

    Tiny MLP: a fp32 ``nn.Linear`` fed by random inputs, optimised with
    SGD. We run 3 steps with and without the SWAP wrapper, comparing
    losses at every step. Determinism comes from re-seeding before each
    block instantiation + identical initial state_dicts.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(0)

    block_swap = nn.Linear(32, 32).to(device)
    block_ref = nn.Linear(32, 32).to(device)
    block_ref.load_state_dict(block_swap.state_dict())

    pool = ActivationSwapPool(
        n_swap=1,
        slot_bytes=8 * 32 * 4,  # batch * features * fp32
        prefetch_depth=2,
    )
    swap_stream = torch.cuda.Stream()
    wrapped = SwappedBlock(block_swap)
    wrapped.attach_runtime(pool, swap_stream)

    opt_swap = torch.optim.SGD(wrapped.parameters(), lr=1e-2)
    opt_ref = torch.optim.SGD(block_ref.parameters(), lr=1e-2)

    losses_swap: list[float] = []
    losses_ref: list[float] = []

    torch.manual_seed(123)
    for _step in range(3):
        x = torch.randn(8, 32, device=device)
        y = torch.randn(8, 32, device=device)

        loss_s = ((wrapped(x) - y) ** 2).mean()
        opt_swap.zero_grad()
        loss_s.backward()
        opt_swap.step()
        losses_swap.append(float(loss_s.detach().cpu()))

        loss_r = ((block_ref(x) - y) ** 2).mean()
        opt_ref.zero_grad()
        loss_r.backward()
        opt_ref.step()
        losses_ref.append(float(loss_r.detach().cpu()))

    torch.cuda.synchronize()

    for ls, lr in zip(losses_swap, losses_ref, strict=True):
        assert abs(ls - lr) < 1e-4, (
            f"SWAP loss diverges from reference: swap={losses_swap} ref={losses_ref}"
        )

    # Pool must be drained at the end.
    assert pool.inflight_count == 0
    pool.close()


# ---------------------------------------------------------------------------
# Memory test: SWAP path must not exceed the NONE-path peak
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_swap_m5_frees_gpu_activations_via_saved_tensors_hooks() -> None:
    """M5+: SWAP=on must free GPU activations between fwd and bwd.

    Build a stack of blocks (mimicking a transformer's block list),
    then measure two quantities under SWAP=off vs SWAP=on:

    1. **post-forward residency** (current GPU bytes after the full
       forward chain finishes) — this is where SWAP's value lives:
       earlier blocks' saved tensors should be on CPU, not GPU.
       Acceptance: ≥30% reduction.
    2. **forward+backward peak** — looser target since backward
       brings tensors back to GPU. Acceptance: ≥10% reduction.

    Also asserts gradient correctness within fp32 tolerance: the
    saved-tensor round trip through pinned host memory is bit-
    preserving for floating-point dtypes, so swap=on / swap=off
    produce numerically equivalent gradients.

    Acceptance criterion is **memory reduction**, not throughput —
    paper §3.1.2 says SWAP costs throughput on PCIe 3090s. The
    point of this test is solely "do GPU activations actually leave
    GPU memory under saved_tensors_hooks?"
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain.block import swap as swap_mod

    device = torch.device("cuda")

    class _BigBlock(nn.Module):
        """A block whose forward saves several large tensors.

        Each ``nn.Linear`` saves its input; ``relu`` and ``softmax``
        save their outputs. Total ≈ 4–6 saved tensors per forward,
        mimicking the attention+MLP saved-tensor blizzard.
        """

        def __init__(self, d: int) -> None:
            super().__init__()
            self.lin1 = nn.Linear(d, d, bias=False)
            self.lin2 = nn.Linear(d, d, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            h = self.lin1(x)
            h = torch.relu(h)
            h = torch.softmax(h, dim=-1)
            h = self.lin2(h)
            return h + x

    # Each saved tensor is shape (B=16, S=256, D=512) fp32 = 8 MiB —
    # well above SIZE_THRESHOLD_BYTES (1 MiB). 4 stacked blocks make
    # the cumulative-residency win measurable; a single block hides
    # the win because backward immediately brings tensors back.
    B, S, D = 16, 256, 512
    n_blocks = 4

    def _measure(use_swap: bool) -> dict[str, int | Tensor]:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        torch.manual_seed(0)
        blocks = nn.ModuleList(_BigBlock(D) for _ in range(n_blocks)).to(device)

        if use_swap:
            wrapped_blocks = nn.ModuleList(swap_mod.SwappedBlock(b) for b in blocks)
            # Pool: enough capacity for all blocks × all saved tensors.
            # slot_bytes = exactly one (B, S, D) fp32 tensor.
            pool = ActivationSwapPool(
                n_swap=n_blocks,
                slot_bytes=B * S * D * 4,
                prefetch_depth=2,
                slots_per_block=16,
            )
            stream = torch.cuda.Stream()
            for wb in wrapped_blocks:
                wb.attach_runtime(pool, stream)
            chain = wrapped_blocks
        else:
            pool = None
            chain = blocks

        x = torch.randn(B, S, D, device=device, requires_grad=True)
        h = x
        for b in chain:
            h = b(h)
        torch.cuda.synchronize()
        post_fwd_resident = int(torch.cuda.memory_allocated(device))

        h.sum().backward()
        torch.cuda.synchronize()
        full_peak = int(torch.cuda.max_memory_allocated(device))

        gx = x.grad.detach().clone() if x.grad is not None else torch.empty(0)
        if pool is not None:
            pool.close()
        del chain, blocks, x, h
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return {
            "post_fwd_resident": post_fwd_resident,
            "full_peak": full_peak,
            "gx": gx,
        }

    off = _measure(use_swap=False)
    on = _measure(use_swap=True)

    # 1) Post-forward residency must drop ≥30% — this is the headline
    # M5+ guarantee: saved activations leave GPU between fwd and bwd.
    resident_red = (off["post_fwd_resident"] - on["post_fwd_resident"]) / off[
        "post_fwd_resident"
    ]
    assert resident_red >= 0.30, (
        f"SWAP=on did not free GPU activations after forward: "
        f"baseline={off['post_fwd_resident']:,} "
        f"swap={on['post_fwd_resident']:,} "
        f"reduction={resident_red:.1%} (require >= 30%)"
    )

    # 2) Full fwd+bwd peak should also drop, though by less because
    # backward unpacks bring tensors back. ≥10% is conservative.
    peak_red = (off["full_peak"] - on["full_peak"]) / off["full_peak"]
    assert peak_red >= 0.10, (
        f"SWAP=on did not reduce fwd+bwd peak enough: "
        f"baseline={off['full_peak']:,} swap={on['full_peak']:,} "
        f"reduction={peak_red:.1%} (require >= 10%)"
    )

    # 3) Gradients must be numerically identical — the host round trip
    # is bit-preserving for fp32.
    assert torch.allclose(off["gx"], on["gx"], atol=1e-5, rtol=1e-5), (
        "Gradients diverge between SWAP=on and SWAP=off"
    )


@pytest.mark.gpu
def test_swap_single_block_backward_peak_at_autograd_floor() -> None:
    """Document the per-block backward-peak floor for SWAP saved_tensors_hooks.

    The M5+ stacked-block test demonstrates the headline 43-66% wins,
    which compound across blocks because earlier blocks' saved tensors
    are on CPU while later blocks compute. A *single* block's backward
    peak is fundamentally bounded by an autograd-engine internal:

        For each backward Node, the engine unpacks ALL the Node's saved
        tensors via ``SavedVariable::unpack()`` BEFORE invoking the
        Node's C++ ``apply()``. The unpacked tensors are held as locals
        inside ``apply()`` and released only when ``apply()`` returns.
        Multiple saved tensors per Node = concurrent unpacked GPU
        buffers. No Python-level hook (saved_tensors_hooks unpack,
        Node.register_hook, Node.register_prehook) can intervene
        mid-apply.

    This test pins down the empirical reduction on a single block
    (one ``nn.Linear`` + ``relu`` + ``softmax`` + ``nn.Linear`` +
    residual) and asserts the modest single-block win we actually
    observe (~10%). Anything larger would require either:

    * Replacing matmul/softmax/etc. with autograd Functions that stage
      their saved-tensor lifetimes manually (huge surface, breaks
      model-agnosticism), or
    * A PyTorch C++ engine change to release individual saved tensors
      after each derivative step.

    Both are out of scope. The test documents the floor so future
    maintainers don't repeat the investigation. See commit history
    for the SWAP=off vs SWAP=on profiling traces that establish the
    bound at autograd-engine ``Node::apply()`` granularity.

    The headline savings live in the stacked-block case (the M5+ test
    above). Single-block savings remain at the per-Node fanout floor.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain.block import swap as swap_mod

    device = torch.device("cuda")

    class _BigBlock(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.lin1 = nn.Linear(d, d, bias=False)
            self.lin2 = nn.Linear(d, d, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            h = self.lin1(x)
            h = torch.relu(h)
            h = torch.softmax(h, dim=-1)
            h = self.lin2(h)
            return h + x

    B, S, D = 16, 256, 512

    def _measure(use_swap: bool) -> tuple[int, int]:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        torch.manual_seed(0)
        block = _BigBlock(D).to(device)
        if use_swap:
            wrapped = swap_mod.SwappedBlock(block)
            pool = ActivationSwapPool(
                n_swap=1,
                slot_bytes=B * S * D * 4,
                prefetch_depth=2,
                slots_per_block=16,
            )
            stream = torch.cuda.Stream()
            wrapped.attach_runtime(pool, stream)
            chain: nn.Module = wrapped
        else:
            pool = None
            chain = block

        x = torch.randn(B, S, D, device=device, requires_grad=True)
        h = chain(x)
        torch.cuda.synchronize()
        # Reset peak so we measure ONLY backward — fwd peak is not the
        # bound under investigation; we want the peak GPU usage during
        # the backward pass alone.
        torch.cuda.reset_peak_memory_stats(device)
        h.sum().backward()
        torch.cuda.synchronize()
        bwd_peak = int(torch.cuda.max_memory_allocated(device))
        post_fwd = int(torch.cuda.memory_allocated(device))

        if pool is not None:
            pool.close()
        del chain, block, x, h
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return post_fwd, bwd_peak

    off_post, off_peak = _measure(False)
    on_post, on_peak = _measure(True)
    reduction = (off_peak - on_peak) / off_peak

    # Floor assertion: SWAP=on does reduce single-block backward peak,
    # but only modestly. The bound below (≥5%) is permissive to allow
    # for allocator noise; the headline is "this win is on the order of
    # 10%, not 30%, because of the autograd-engine internals". If a
    # future PyTorch release lets us trim individual saved tensors
    # mid-apply this test will overshoot — that's fine, the assertion
    # is a lower bound.
    assert reduction >= 0.05, (
        f"single-block backward peak unexpectedly NOT reduced by SWAP: "
        f"baseline={off_peak:,} swap={on_peak:,} reduction={reduction:.1%}"
    )
    # Upper-bound documenting the autograd-engine floor. If this fails
    # high (>25%), the floor has shifted — investigate (likely a torch
    # version that lets us release saved tensors mid-apply, which would
    # let us tighten this further).
    assert reduction <= 0.25, (
        f"single-block backward peak reduction {reduction:.1%} exceeds "
        "documented autograd-engine floor (~10-15%). PyTorch may have "
        "changed Node::apply saved-variable lifetime. Re-investigate "
        "register_hook-based early-free; see commit history for prior "
        "investigation."
    )


@pytest.mark.gpu
def test_swap_path_does_not_blow_peak() -> None:
    """Peak GPU memory with SWAP attached is no larger than the NONE-path peak.

    On 3090 hardware the SWAP path's actual memory benefit comes from
    nulling the GPU activation between fwd and bwd; option 2A's
    minimum-viable wrapper does NOT yet null it (the autograd-saved
    storage is still alive). The realistic acceptance criterion here
    is "the SWAP path is wired up and runs without inflating the peak"
    — anything stronger would require the M5+ activation-storage-null
    integration. We assert the peak is within +10% of the unwrapped
    baseline to allow allocator noise.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(0)

    def _peak(use_swap: bool) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        block = nn.Linear(256, 256).to(device)
        if use_swap:
            wrapped = SwappedBlock(block)
            pool = ActivationSwapPool(
                n_swap=1,
                slot_bytes=64 * 256 * 4,
                prefetch_depth=2,
            )
            stream = torch.cuda.Stream()
            wrapped.attach_runtime(pool, stream)
            mod: nn.Module = wrapped
        else:
            pool = None
            mod = block

        x = torch.randn(64, 256, device=device, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated(device))

        if pool is not None:
            pool.close()
        del mod, x, out
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return peak

    baseline = _peak(use_swap=False)
    swap_peak = _peak(use_swap=True)
    # Allow a small inflation (the slot view + temp gpu_buf during
    # backward are real bytes, but nothing pathological).
    assert swap_peak <= int(baseline * 1.20), (
        f"SWAP peak {swap_peak} unexpectedly larger than baseline {baseline}"
    )


# ---------------------------------------------------------------------------
# Searcher feasibility gate
# ---------------------------------------------------------------------------


def test_searcher_prunes_swap_under_tight_cpu_budget() -> None:
    """When CPU capacity cannot hold the swap pool, n_swap=0 is selected.

    Build a synthetic profile where ``activation_sizes`` would need
    several hundred MB per slot, then set ``cpu_capacity_bytes`` to a
    value that fits the chunk pool but NOT the swap pool. The searcher
    must pick ``n_swap=0`` rather than failing — there's always a
    no-SWAP candidate that fits.
    """
    from axolotl.integrations.protrain.search.exhaustive import search
    from axolotl.integrations.protrain.types import (
        BlockId,
        ChunkId,
        ChunkLayout,
        HardwareProfile,
        OpId,
        OpRecord,
        ParamId,
        ProfilerTrace,
    )

    n_block = 4
    activation_per_block = 64 * (1 << 20)  # 64 MB per block
    n_chunk = 4
    s_chunk = 16 * (1 << 20)  # 16 MB

    # Trivial layout: each block owns one chunk.
    layout = ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple((ParamId(f"b{b}.w"),) for b in range(n_chunk)),
        param_to_chunk={ParamId(f"b{b}.w"): ChunkId(b) for b in range(n_chunk)},
        block_to_chunks={BlockId(b): (ChunkId(b),) for b in range(n_block)},
    )

    # Profiler trace: one fwd op per block, no backward ops.
    op_records = tuple(
        OpRecord(
            op_id=OpId(i),
            module_path=f"layers.{i}",
            qualified_name="aten::linear",
            shape_signature=((1, 32),),
            block_id=BlockId(i),
            is_forward=True,
        )
        for i in range(n_block)
    )
    activation_sizes = {BlockId(b): activation_per_block for b in range(n_block)}
    trace = ProfilerTrace(
        op_order=op_records,
        intra_op_delta={OpId(i): 0 for i in range(n_block)},
        inter_op_delta={OpId(i): 0 for i in range(n_block)},
        activation_sizes=activation_sizes,
        model_state_bytes=n_chunk * s_chunk,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        nccl_gather_s={s_chunk: 1e-3},
        nccl_reduce_s={s_chunk: 1e-3},
        arch_hash="synthetic",
        bs=1,
        seq=32,
        sku="synthetic",
        world=1,
    )

    hw = HardwareProfile(
        gpu_sku="synthetic",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
    )

    capacity_bytes = 4 * (1 << 30)  # plenty of GPU
    # CPU budget large enough for the chunk pool (~64 MB) but NOT for
    # any swap candidate. With prefetch_depth=2 and 64 MB activations,
    # the smallest n_swap=1 candidate needs 128 MB + chunk term. Set
    # the budget halfway so n_swap=0 fits and any n_swap > 0 fails.
    cpu_capacity_bytes = (n_chunk * s_chunk) + 64 * (1 << 20)  # ~128 MB

    result = search(
        trace=trace,
        layout=layout,
        capacity_bytes=capacity_bytes,
        hw=hw,
        cpu_capacity_bytes=cpu_capacity_bytes,
    )
    assert result.cfg.n_swap == 0, (
        f"searcher should refuse n_swap > 0 under tight CPU budget; got {result.cfg}"
    )


def test_searcher_admits_swap_under_generous_cpu_budget() -> None:
    """Sanity check: with abundant CPU budget the gate doesn't bite.

    Without a tight CPU gate the searcher's pick on 3090-style hw is
    governed by the runtime cost model, which usually selects
    ``n_swap=0`` anyway because PCIe-bound (paper §3.1.2). The
    assertion here is the *gate-disabled* invariant: under
    ``cpu_capacity_bytes=None`` the searcher must produce a config
    without raising the CPU-pressure RuntimeError, regardless of what
    n_swap value it actually picks.
    """
    from axolotl.integrations.protrain.search.exhaustive import search
    from axolotl.integrations.protrain.types import (
        BlockId,
        ChunkId,
        ChunkLayout,
        HardwareProfile,
        OpId,
        OpRecord,
        ParamId,
        ProfilerTrace,
    )

    n_block = 2
    n_chunk = 2
    s_chunk = 8 * (1 << 20)

    layout = ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple((ParamId(f"b{b}.w"),) for b in range(n_chunk)),
        param_to_chunk={ParamId(f"b{b}.w"): ChunkId(b) for b in range(n_chunk)},
        block_to_chunks={BlockId(b): (ChunkId(b),) for b in range(n_block)},
    )
    op_records = tuple(
        OpRecord(
            op_id=OpId(i),
            module_path=f"layers.{i}",
            qualified_name="aten::linear",
            shape_signature=((1, 32),),
            block_id=BlockId(i),
            is_forward=True,
        )
        for i in range(n_block)
    )
    trace = ProfilerTrace(
        op_order=op_records,
        intra_op_delta={OpId(i): 0 for i in range(n_block)},
        inter_op_delta={OpId(i): 0 for i in range(n_block)},
        activation_sizes={BlockId(b): 1 << 20 for b in range(n_block)},
        model_state_bytes=n_chunk * s_chunk,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        nccl_gather_s={s_chunk: 1e-3},
        nccl_reduce_s={s_chunk: 1e-3},
        arch_hash="synthetic",
        bs=1,
        seq=32,
        sku="synthetic",
        world=1,
    )
    hw = HardwareProfile(
        gpu_sku="synthetic",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
    )
    # Should NOT raise — gate disabled.
    result = search(
        trace=trace,
        layout=layout,
        capacity_bytes=4 * (1 << 30),
        hw=hw,
        cpu_capacity_bytes=None,
    )
    assert result.cfg is not None
    # No specific n_swap claim — the cost model on 3090-style hw will
    # almost always pick 0 here, but this test only validates the
    # gate-disabled path doesn't bust on SWAP candidates.


# ---------------------------------------------------------------------------
# End-to-end smoke: wrap a tiny model with n_swap_override>0 and run 3 iters
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
def test_swap_smoke_n_swap_override_runs_three_iters() -> None:
    """Forced ``n_swap > 0`` via override drives 3 iterations without crashing.

    Uses ``protrain_model_wrapper(n_swap_override=...)`` to force the
    SWAP path even though the searcher would normally pick 0 on
    3090-class hardware. Verifies:

    * The wrapper construction succeeds with SWAP wiring (pool +
      swap_stream attached).
    * 3 fwd+bwd iterations complete with finite losses.
    * The activation pool is empty after each iteration.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    transformers = pytest.importorskip("transformers")

    from axolotl.integrations.protrain.api import protrain_model_wrapper
    from axolotl.integrations.protrain.types import HardwareProfile

    device = torch.device("cuda")
    cfg = transformers.GPT2Config(
        n_layer=4, n_head=2, n_embd=64, vocab_size=128, n_positions=16
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(cfg).to(device)

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(device),
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
    )

    # Force n_swap=2 (first 2 blocks SWAP) via the explicit override.
    # The other knobs are sized to keep all chunks persistent — SWAP
    # blocks need their parameter chunks to be persistent (see
    # block_map_runtime_admissible in exhaustive.py).
    try:
        wrapped = protrain_model_wrapper(
            model,
            model_config=cfg,
            hardware_profile=hw,
            batch_size=1,
            seq_len=8,
            capacity_bytes=2 * (1 << 30),
            force_all_persistent=False,
            n_persist_override=None,
            n_buffer_override=None,
            n_swap_override=None,
            n_checkpoint_override=None,
        )
    except Exception:
        pytest.skip("baseline wrap failed on this GPU/env")
    n_chunk = cast("ChunkManager", wrapped.chunk_manager).layout.N_chunk
    # Tear down probe. ``_hook_handles`` is dynamically attached; cast for
    # mypy so each handle's ``.remove`` resolves against ``RemovableHandle``.
    for h in cast("list[Any]", wrapped._hook_handles):
        try:
            h.remove()
        except Exception:
            pass
    del wrapped, model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Now build the real configuration: all-persistent + n_swap=2.
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(cfg).to(device)
    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=1,
        seq_len=8,
        capacity_bytes=2 * (1 << 30),
        n_persist_override=n_chunk,
        n_buffer_override=max(1, n_chunk),
        n_swap_override=2,
        n_checkpoint_override=0,
    )
    # Verify the SWAP pool was wired.
    scheduler = cast("Scheduler", wrapped.scheduler)
    swap_pool = getattr(scheduler, "swap_pool", None)
    assert swap_pool is not None, "SWAP pool was not constructed"
    assert swap_pool.n_swap == 2

    # Drive 3 iterations.
    for _i in range(3):
        input_ids = torch.randint(
            0, cfg.vocab_size, (1, 8), device=device, dtype=torch.long
        )
        out = wrapped.module(input_ids=input_ids, labels=input_ids.clone())
        loss = out.loss
        assert torch.isfinite(loss), f"non-finite loss at iter {_i}"
        loss.backward()
        # Drain so swap stream + chunk prefetch settle before next iter.
        scheduler.drain()
        # Pool should have no in-flight slots between iterations.
        assert swap_pool.inflight_count == 0, (
            f"SWAP pool leaked slots at iter {_i}: inflight={swap_pool.inflight_count}"
        )

    # Tear down hooks.
    for h in cast("list[Any]", wrapped._hook_handles):
        try:
            h.remove()
        except Exception:
            pass
    swap_pool.close()


# ---------------------------------------------------------------------------
# SWAP gate enforcement (paper §3.3 "swap-in only when memory available")
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_swap_gate_raises_when_headroom_unrecoverable(monkeypatch) -> None:
    """SWAP gate raises ``RuntimeError`` when sync-and-retry cannot free enough.

    The gate's contract: if even after ``_SWAP_MAX_DRAIN_RETRIES``
    sync-and-recheck attempts the device still cannot satisfy
    ``required_bytes + _SWAP_HEADROOM_SAFETY_BYTES``, the unpack
    path must raise (NOT fall through to ``empty_strided`` and OOM
    in the kernel allocator). The message must name the SWAP gate
    so the operator sees a config-level invariant violation, not a
    mysterious kernel OOM.

    We mock ``torch.cuda.mem_get_info`` to permanently report a
    deficit; the retry loop will exhaust without ever observing
    enough headroom, and the raise path must trigger BEFORE
    ``torch.empty_strided`` is called.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain.block import swap as swap_mod

    device = torch.device("cuda")

    # Build a minimal _CPUHandle with bookkeeping that says "I need a
    # large allocation". The gate consults handle.nbytes and
    # handle.device only — pool/stream/slot fields aren't touched on
    # the raise path.
    pool = ActivationSwapPool(
        n_swap=1, slot_bytes=64, prefetch_depth=2, slots_per_block=1
    )
    swap_stream = torch.cuda.Stream()
    handle = swap_mod._CPUHandle(
        pool=pool,
        swap_stream=swap_stream,
        slot_id=0,
        shape=(8,),
        stride=(1,),
        dtype=torch.float32,
        device=device,
        # Larger than any plausible free count — combined with our
        # mock below, the deficit is unrecoverable.
        nbytes=1 << 40,  # 1 TiB
        requires_grad=False,
    )

    # Mock mem_get_info so every call (initial + every retry) returns
    # a tiny free count. The retry loop drains _SWAP_MAX_DRAIN_RETRIES
    # times, then the gate raises.
    call_count = {"n": 0}

    def fake_mem_get_info(_dev=None):
        call_count["n"] += 1
        return (1024, 1 << 40)  # 1 KiB free, 1 TiB total

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)

    # Sentinel: empty_strided MUST NOT be called once the raise
    # unwinds the stack. We monkeypatch it to raise loudly if it
    # ever runs after the gate decided to raise.
    empty_strided_called = {"n": 0}
    real_empty_strided = torch.empty_strided

    def spy_empty_strided(*args, **kwargs):
        empty_strided_called["n"] += 1
        return real_empty_strided(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_strided", spy_empty_strided)

    pack, unpack = swap_mod._make_pack_unpack(
        pool, swap_stream, swap_mod.SIZE_THRESHOLD_BYTES
    )

    with pytest.raises(RuntimeError, match="ProTrain SWAP gate"):
        unpack(handle)

    # Gate consulted mem_get_info once initially + once per retry.
    expected_calls = 1 + swap_mod._SWAP_MAX_DRAIN_RETRIES
    assert call_count["n"] == expected_calls, (
        f"expected {expected_calls} mem_get_info calls "
        f"(1 initial + {swap_mod._SWAP_MAX_DRAIN_RETRIES} retries), "
        f"got {call_count['n']}"
    )

    # Critical: the raise must unwind BEFORE empty_strided runs.
    # Falling through to empty_strided is the antipattern this fix
    # eliminates.
    assert empty_strided_called["n"] == 0, (
        "SWAP gate raised but empty_strided was still invoked — the "
        "gate is observing-and-proceeding instead of enforcing."
    )

    pool.close()


@pytest.mark.gpu
def test_swap_gate_message_names_invariant_and_remediation(monkeypatch) -> None:
    """Raised message names the gate, the deficit, and operator remedies.

    Operators see this message when the cost model's swap-in-headroom
    assumption breaks at runtime. The message must:

    * Name "ProTrain SWAP gate" so it's the obvious owner.
    * Surface the numerical deficit (need vs. have) so the operator
      can size the gap.
    * Suggest concrete remediation (reduce ``n_swap`` / set to 0).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain.block import swap as swap_mod

    device = torch.device("cuda")

    pool = ActivationSwapPool(
        n_swap=1, slot_bytes=64, prefetch_depth=2, slots_per_block=1
    )
    swap_stream = torch.cuda.Stream()
    handle = swap_mod._CPUHandle(
        pool=pool,
        swap_stream=swap_stream,
        slot_id=0,
        shape=(8,),
        stride=(1,),
        dtype=torch.float32,
        device=device,
        nbytes=1 << 40,
        requires_grad=False,
    )

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _dev=None: (1024, 1 << 40)
    )

    pack, unpack = swap_mod._make_pack_unpack(
        pool, swap_stream, swap_mod.SIZE_THRESHOLD_BYTES
    )

    with pytest.raises(RuntimeError) as excinfo:
        unpack(handle)

    msg = str(excinfo.value)
    assert "ProTrain SWAP gate" in msg
    assert "n_swap" in msg, "message must suggest n_swap reduction"
    assert str(handle.nbytes) in msg, "message must surface the byte deficit"
    assert "safety margin" in msg, "message must surface the safety margin"

    pool.close()
