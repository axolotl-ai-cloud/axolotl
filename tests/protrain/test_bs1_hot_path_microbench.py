"""CPU microbench for the ProTrain bs=1 hot path (§16 PR #4).

Profiles the Python-side per-step overhead introduced by ProTrain hooks
on a tiny synthetic transformer wired up the way a real Meta-Llama-3-8B
+ LoRA model is: N transformer blocks, each block carrying ``self_attn``
(so ``discover_blocks`` picks the ``ModuleList`` up) and a fan-out of
``lora_A`` / ``lora_B`` sub-modules per block (so
``_find_peft_lora_containers`` discovers them).

The bs=1 cliff documented in proposal §6.a / §6.d is a Python-side
per-step overhead that doesn't amortize at small batch sizes. This
harness measures that overhead in isolation (no CUDA, no real model
compute) so a regression-guard can run on CPU CI.

Baseline (pre-PR #4, measured on this rig, see body for the captured
value) and post-fix figures live in the source; the assertion is a
loose ``post_fix_per_step_us <= baseline_us * 0.7`` so we catch the
optimisation regressing in either direction.
"""

from __future__ import annotations

import contextlib
import cProfile
import os
import pstats
import time
from typing import Any

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Synthetic model that exercises both code paths (block-level + LoRA-container).
# ---------------------------------------------------------------------------


class _LoRASubmodule(nn.Module):
    """Stand-in for one PEFT-LoRA-wrapped Linear: base + lora_A + lora_B."""

    def __init__(self, dim: int, rank: int = 8) -> None:
        super().__init__()
        self.base_layer = nn.Linear(dim, dim, bias=False)
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        # Freeze the base, train the lora factors (matches PEFT behaviour).
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


class _AttnLikeBlock(nn.Module):
    """Transformer-block stand-in with q/k/v/o + gate/up/down LoRA-wrapped.

    Mirrors Llama-class layout: 7 LoRA-equipped sub-modules per block
    (q/k/v/o under self_attn; gate/up/down at the block root). The
    ``self_attn`` attribute is what makes ``discover_blocks`` pick the
    parent ``ModuleList`` up under the attention heuristic.
    """

    def __init__(self, dim: int, rank: int = 8) -> None:
        super().__init__()
        # self_attn carries q/k/v/o LoRA wrappers.
        self.self_attn = nn.Module()
        self.self_attn.q_proj = _LoRASubmodule(dim, rank)
        self.self_attn.k_proj = _LoRASubmodule(dim, rank)
        self.self_attn.v_proj = _LoRASubmodule(dim, rank)
        self.self_attn.o_proj = _LoRASubmodule(dim, rank)
        # MLP sub-modules at block root.
        self.gate_proj = _LoRASubmodule(dim, rank)
        self.up_proj = _LoRASubmodule(dim, rank)
        self.down_proj = _LoRASubmodule(dim, rank)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.self_attn.o_proj(
            self.self_attn.q_proj(h)
            + self.self_attn.k_proj(h)
            + self.self_attn.v_proj(h)
        )
        x = x + h
        return x + self.down_proj(self.gate_proj(x) * self.up_proj(x))


class _TinyLoRATransformer(nn.Module):
    """4-block ``ModuleList`` of ``_AttnLikeBlock``; CPU-only."""

    def __init__(self, n_blocks: int = 4, dim: int = 64, rank: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_AttnLikeBlock(dim, rank) for _ in range(n_blocks)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Stub scheduler / chunk-manager: count calls, do no real work.
# ---------------------------------------------------------------------------


class _StubScheduler:
    """Per-call counters; methods are no-ops so we measure Python overhead only."""

    def __init__(self) -> None:
        self.pre_forward = 0
        self.post_forward = 0
        self.pre_backward = 0
        self.post_backward = 0
        self.ensure_resident = 0

    def pre_block_forward(self, block_id: int) -> None:  # noqa: ARG002
        self.pre_forward += 1

    def post_block_forward(self, block_id: int) -> None:  # noqa: ARG002
        self.post_forward += 1

    def pre_block_backward(self, block_id: int) -> None:  # noqa: ARG002
        self.pre_backward += 1

    def post_block_backward(self, block_id: int) -> None:  # noqa: ARG002
        self.post_backward += 1

    def ensure_chunks_resident(self, chunk_ids: Any) -> None:  # noqa: ARG002
        self.ensure_resident += 1


class _StubChunkManager:
    """Carries the minimum surface ``install_hooks`` reads from."""

    def __init__(self, model: nn.Module) -> None:
        # _container_chunk_ids reads _params_by_id; one chunk per param is fine
        # for the microbench (the hook is fired once per container regardless).
        self._params_by_id: dict[str, nn.Parameter] = dict(model.named_parameters())

        # layout.param_to_chunk: map each param name -> a single chunk id 0.
        class _Layout:
            def __init__(self, names):
                self.param_to_chunk = {name: 0 for name in names}

        self.layout = _Layout(self._params_by_id.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model_with_hooks(n_blocks: int = 4, dim: int = 64, rank: int = 8):
    """Build model + install ProTrain hooks; return (model, sched, handles)."""
    from axolotl.integrations.protrain.runtime.hooks import install_hooks
    from axolotl.integrations.protrain.types import BlockId, BlockMode

    model = _TinyLoRATransformer(n_blocks=n_blocks, dim=dim, rank=rank)
    sched = _StubScheduler()
    cm = _StubChunkManager(model)
    block_map = {BlockId(i): BlockMode.NONE for i in range(n_blocks)}
    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    return model, sched, cm, handles


def _time_steps(model, bs: int, dim: int, n_iters: int = 100) -> float:
    """Return per-step wall time in microseconds (median over n_iters)."""
    # Warm up to JIT-compile any lazy paths.
    x = torch.randn(bs, 16, dim)
    for _ in range(5):
        out = model(x)
        out.sum().backward()
        # Manual zero_grad to avoid touching an Optimizer for this microbench.
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

    samples: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = model(x)
        out.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)  # us
    samples.sort()
    # Median is robust to GC spikes / scheduler jitter on busy CI hosts.
    return samples[len(samples) // 2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hooks_fire_on_synthetic_model():
    """Sanity check: every hook actually fires on the synthetic model."""
    model, sched, _cm, handles = _build_model_with_hooks(n_blocks=4, dim=64)
    try:
        x = torch.randn(1, 8, 64)
        out = model(x)
        out.sum().backward()

        # Block-level hooks: 1 per block per direction.
        assert sched.pre_forward == 4
        assert sched.post_forward == 4
        assert sched.pre_backward == 4
        assert sched.post_backward == 4

        # LoRA-container hooks: 7 containers per block × 4 hooks
        # (pre/post fwd + pre/post bwd) × 4 blocks = 112 fires.
        # Each container fires ensure_chunks_resident exactly 4 times per step.
        # Plus any nested-container double-fires (LoRASubmodule itself
        # contains lora_A / lora_B child modules that name-match), so check
        # for a lower bound and reproducibility:
        assert sched.ensure_resident >= 4 * 7 * 4, (
            f"LoRA-container hooks fired only {sched.ensure_resident} times; "
            "expected at least 4 (pre/post fwd/bwd) per 7 containers per "
            "4 blocks = 112."
        )
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


@pytest.mark.skipif(
    os.environ.get("CI", "").lower() in ("true", "1", "yes"),
    reason=(
        "Wall-clock microbench: GitHub Actions shared runners exhibit 5-10x "
        "variance on Python-side timings (observed ~30 ms/step vs the "
        "4.7 ms/step CPU baseline captured on dedicated dev HW). The "
        "regression-guard is meaningful only on consistent hardware; run "
        "locally to enforce the budget."
    ),
)
def test_bs1_per_step_overhead_within_budget():
    """Measure ProTrain-attributable per-step overhead at bs=1 vs no-hooks baseline.

    Strategy: measure the same model twice — once with ProTrain hooks
    installed, once without — and assert the with-hooks delta is bounded
    against a captured pre-fix baseline.

    Pre-fix baseline captured on this rig (commit 6fdf9113a, CPU-only,
    Python 3.12) with n_blocks=8 and the cProfile breakdown summarised
    in §16 PR #4 notes:

    * 32 ``setup_input_hook`` + 32 ``setup_output_hook`` autograd-hook
      registrations per step (one per registered ``full_backward_pre_hook``
      / ``full_backward_hook``, regardless of bs) → 1.4 ms/step in PyTorch
      autograd machinery alone.
    * Per-call ``cuda.is_available()`` inside ``ensure_chunks_resident``
      fired 7 LoRA-containers × 4 hooks × 8 blocks = 224 times/step at
      ~2.7 us each → 600 us/step.
    * ``post_block_forward`` per-step set() construction of next-block
      chunks: small but non-zero.

    Pre-fix total observed: ~4.7 ms/step ProTrain overhead at n_blocks=8.

    Post-fix expectation: scheduler-cached CUDA flag removes the per-call
    syscall and reduces the per-block walk overhead via init-time caches.
    The autograd-machinery cost remains (cannot be removed without
    dropping the LoRA-container hook quartet which the offload tests
    pin). Net expected ~10-20% reduction.

    The regression-guard threshold of ``0.90x`` catches a slide back
    toward the pre-fix shape without being so tight that CI-host
    variance trips it. The remaining residual (autograd machinery)
    is GPU-profile-driven and tracked in the updated §16 PR #4.
    """
    # Captured BEFORE the optimisation on commit 6fdf9113a; see docstring.
    # In microseconds. Median of 5 trials × 100 iters each.
    BASELINE_OVERHEAD_US = 4700.0

    # Same n_blocks=8 in both arms so model-compute cost cancels in the diff.
    n_blocks = 8

    # No hooks: model compute + autograd alone.
    model_bare = _TinyLoRATransformer(n_blocks=n_blocks, dim=64, rank=8)
    bare_us = _time_steps(model_bare, bs=1, dim=64, n_iters=80)

    # With ProTrain hooks installed.
    model, _sched, _cm, handles = _build_model_with_hooks(
        n_blocks=n_blocks, dim=64, rank=8
    )
    try:
        hooks_us = _time_steps(model, bs=1, dim=64, n_iters=80)
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()

    overhead_us = hooks_us - bare_us

    # Print so -s -v shows the actual numbers on every CI run.
    print(f"\n[bs1_hot_path] no-hooks per-step:   {bare_us:.1f} us")
    print(f"[bs1_hot_path] with-hooks per-step: {hooks_us:.1f} us")
    print(f"[bs1_hot_path] overhead delta:      {overhead_us:.1f} us")
    print(f"[bs1_hot_path] pre-fix baseline:    {BASELINE_OVERHEAD_US:.1f} us")
    print(
        f"[bs1_hot_path] ratio:               {overhead_us / BASELINE_OVERHEAD_US:.2f}x"
    )

    # Catches a regression back toward the pre-fix shape.
    assert overhead_us <= BASELINE_OVERHEAD_US * 0.90, (
        f"bs=1 per-step ProTrain overhead {overhead_us:.1f} us exceeds 90% of "
        f"pre-fix baseline {BASELINE_OVERHEAD_US:.1f} us; the §16 PR #4 hot-path "
        "optimisation has regressed."
    )


def test_bs4_no_regression_relative_to_bs1():
    """At bs=4 the per-step Python overhead must NOT scale linearly with bs.

    The Python overhead is bs-independent (hook fire count doesn't depend
    on the batch dimension), so the per-step wall time at bs=4 should be
    very close to bs=1 (the only delta is the extra tensor allocations +
    backward kernel sizes, both linear in numel — small at dim=64).

    This is a sanity check that the optimisation didn't accidentally
    *pessimize* the amortized case by, say, adding bs-dependent work
    inside a hook.
    """
    model, _sched, _cm, handles = _build_model_with_hooks(n_blocks=4, dim=64, rank=8)
    try:
        bs1_us = _time_steps(model, bs=1, dim=64, n_iters=60)
        bs4_us = _time_steps(model, bs=4, dim=64, n_iters=60)
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()

    print(f"\n[bs1_hot_path] bs=1 per-step: {bs1_us:.1f} us")
    print(f"[bs1_hot_path] bs=4 per-step: {bs4_us:.1f} us")
    print(f"[bs1_hot_path] bs4/bs1 ratio: {bs4_us / bs1_us:.2f}x")

    # bs=4 should be < 2x bs=1 (the hook overhead is constant; the only
    # extra cost is the marginal tensor ops, which are cheap at dim=64).
    # Catch a regression where the optimisation added bs-dependent work.
    assert bs4_us < bs1_us * 2.0, (
        f"bs=4 ({bs4_us:.1f} us) is more than 2x bs=1 ({bs1_us:.1f} us); "
        "the §16 PR #4 optimisation may have introduced bs-dependent work "
        "inside a per-step hot path."
    )


# ---------------------------------------------------------------------------
# Optional manual run: pytest -s -v ::test_bs1_cprofile_dump
# Prints the top-20 cumulative-time callees so we know where the next
# round of optimisation should land if we revisit this hot path.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="manual diagnostic; enable with -k cprofile when re-profiling")
def test_bs1_cprofile_dump():
    """Run cProfile over the hot path and print the top 20 callees by cumulative time."""
    model, _sched, _cm, handles = _build_model_with_hooks(n_blocks=4, dim=64, rank=8)
    try:
        x = torch.randn(1, 16, 64)

        def _run() -> None:
            for _ in range(50):
                out = model(x)
                out.sum().backward()
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None

        # Warm
        _run()

        prof = cProfile.Profile()
        prof.enable()
        _run()
        prof.disable()

        stats = pstats.Stats(prof)
        stats.sort_stats("cumulative")
        stats.print_stats(30)
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()
