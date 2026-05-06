"""Unit + GPU tests for the ProTrain M1 profiler."""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.profiler import (
    ProfilerCacheKey,
    load_cached_trace,
    measure_pcie,
    reconstruct_peak_bytes,
    run_trace,
    save_cached_trace,
)
from axolotl.integrations.protrain.profiler.on_demand import OnDemandTensorMgr
from axolotl.integrations.protrain.types import (
    BlockId,
    OpId,
    OpRecord,
    ProfilerConfig,
    ProfilerTrace,
)

_TINY_MODEL_CANDIDATES = (
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-gpt2",
)


def _load_tiny_gpt2():
    """Try the canonical tiny-GPT2 checkpoint, fall back to the HF-internal one."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    last_exc: Exception | None = None
    for name in _TINY_MODEL_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name)
            return name, tok, model
        except Exception as exc:  # pragma: no cover - network-dependent
            last_exc = exc
            continue
    raise RuntimeError(f"no tiny-GPT2 checkpoint available: {last_exc}")


def _build_batch(tok, bs: int, seq: int, device):

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    text = ["hello world"] * bs
    enc = tok(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@pytest.mark.gpu
def test_reconstruct_peak_within_10pct_tiny_gpt2(gpu_device):
    """The M1 accuracy contract: simplified peak within 10% of max_memory_allocated."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    name, tok, model = _load_tiny_gpt2()
    model = model.to(device)

    bs, seq = 2, 128
    batch = _build_batch(tok, bs, seq, device)

    cfg = ProfilerConfig(
        batch_size=bs,
        seq_len=seq,
        device=str(device),
        include_backward=True,
        on_demand=False,
    )

    # First: profiled run. Hooks add a small constant; we care about the
    # reconstructed number, not the measured peak during this call.
    trace = run_trace(model, batch, cfg)
    peak_est = reconstruct_peak_bytes(trace)

    # Second: ground-truth run with no hooks. Fresh zero for peak stats.
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    # Re-fetch a batch tied to no retained autograd graph from the first pass.
    batch2 = _build_batch(tok, bs, seq, device)
    output = model(**batch2)
    loss = output.loss if hasattr(output, "loss") else output[0].sum()
    loss.backward()
    torch.cuda.synchronize(device)
    ground_truth = int(torch.cuda.max_memory_allocated(device))

    assert ground_truth > 0, "ground truth peak should be positive"
    rel_err = abs(peak_est - ground_truth) / ground_truth
    assert rel_err < 0.10, (
        f"reconstructed peak {peak_est} vs ground truth {ground_truth} "
        f"rel_err={rel_err:.3f} on model {name!r}"
    )


def _minimal_trace() -> ProfilerTrace:
    """Build a tiny valid ProfilerTrace for cache round-trip testing."""
    op = OpRecord(
        op_id=OpId(0),
        module_path="root.layer0",
        qualified_name="Linear",
        shape_signature=((2, 128, 16),),
        block_id=BlockId(0),
        is_forward=True,
    )
    return ProfilerTrace(
        op_order=(op,),
        intra_op_delta={OpId(0): 1024},
        inter_op_delta={OpId(0): 512},
        activation_sizes={BlockId(0): 2048},
        model_state_bytes=1 << 20,
        pcie_h2d_bps=25e9,
        pcie_d2h_bps=23e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="deadbeef",
        bs=2,
        seq=128,
        sku="NVIDIA GeForce RTX 3090",
        world=1,
    )


def test_cache_roundtrip(tmp_path, monkeypatch):
    """save -> load must return an equal ProfilerTrace."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = ProfilerCacheKey(
        arch_hash="deadbeef",
        bs=2,
        seq=128,
        sku="NVIDIA GeForce RTX 3090",
        world=1,
    )
    trace = _minimal_trace()
    path = save_cached_trace(key, trace)
    assert path.exists()

    loaded = load_cached_trace(key)
    assert loaded is not None
    assert loaded == trace

    # Missing key returns None.
    other = ProfilerCacheKey(
        arch_hash="feedface", bs=2, seq=128, sku="NVIDIA GeForce RTX 3090", world=1
    )
    assert load_cached_trace(other) is None


def test_minimal_trace_defaults_new_bwd_peak_fields_to_zero():
    """A minimally-constructed trace defaults the v19 backward-peak fields.

    Regression guard: pre-v19 callers that built ``ProfilerTrace`` directly
    (cost-model unit tests, search/exhaustive smoke tests) did not populate
    ``steady_bwd_peak_bytes`` / ``steady_bwd_block_peak_bytes``. The dataclass
    must default those to 0 / empty so the cost model continues to fall back
    to the analytical bwd estimate when a trace was captured without
    ``include_backward=True``.
    """
    trace = _minimal_trace()
    assert trace.steady_bwd_peak_bytes == 0
    assert trace.steady_bwd_block_peak_bytes == {}


def test_cache_roundtrip_preserves_bwd_peak_fields(tmp_path, monkeypatch):
    """save -> load preserves ``steady_bwd_peak_bytes`` + per-block dict.

    Locks in the TRACE_VERSION 19 cache schema additions: a backward-aware
    profile records ``steady_bwd_peak_bytes`` (aggregate) and
    ``steady_bwd_block_peak_bytes`` (per-block) alongside the forward-side
    fields, and the JSON cache must round-trip both bit-for-bit.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = ProfilerCacheKey(
        arch_hash="cafebabe",
        bs=2,
        seq=128,
        sku="NVIDIA GeForce RTX 3090",
        world=1,
    )
    op = OpRecord(
        op_id=OpId(0),
        module_path="root.layer0",
        qualified_name="Linear",
        shape_signature=((2, 128, 16),),
        block_id=BlockId(0),
        is_forward=True,
    )
    trace = ProfilerTrace(
        op_order=(op,),
        intra_op_delta={OpId(0): 1024},
        inter_op_delta={OpId(0): 512},
        activation_sizes={BlockId(0): 2048},
        model_state_bytes=1 << 20,
        pcie_h2d_bps=25e9,
        pcie_d2h_bps=23e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="cafebabe",
        bs=2,
        seq=128,
        sku="NVIDIA GeForce RTX 3090",
        world=1,
        # Forward-side measurements (existing schema)
        steady_fwd_peak_bytes=8 * (1 << 20),
        steady_fwd_block_peak_bytes={BlockId(0): 6 * (1 << 20)},
        # Backward-side measurements — NEW in v19
        steady_bwd_peak_bytes=12 * (1 << 20),
        steady_bwd_block_peak_bytes={BlockId(0): 10 * (1 << 20)},
    )
    save_cached_trace(key, trace)
    loaded = load_cached_trace(key)
    assert loaded is not None
    assert loaded.steady_bwd_peak_bytes == trace.steady_bwd_peak_bytes
    assert loaded.steady_bwd_block_peak_bytes == trace.steady_bwd_block_peak_bytes
    # Full equality also covers backward compat with the rest of the schema.
    assert loaded == trace


@pytest.mark.gpu
def test_trace_records_steady_bwd_peak_and_per_block(gpu_device):
    """Backward-aware trace populates ``steady_bwd_peak_bytes`` + per-block dict.

    Locks in the v19 capture path: when ``cfg.include_backward=True`` AND
    on-demand does not engage (small model fits with headroom), the steady
    hot loop records both the whole-bwd peak (cumulative
    ``max_memory_allocated`` across the backward window) and the per-block
    bwd peaks via ``register_full_backward_hook``. Both must be strictly
    positive on a tiny GPT-2 forward+backward.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    _name, tok, model = _load_tiny_gpt2()
    model = model.to(device)

    bs, seq = 1, 64
    batch = _build_batch(tok, bs, seq, device)

    cfg = ProfilerConfig(
        batch_size=bs,
        seq_len=seq,
        device=str(device),
        include_backward=True,
        on_demand=False,
    )
    trace = run_trace(model, batch, cfg)

    # Aggregate bwd peak: must be > 0 when the steady-bwd loop ran.
    assert trace.steady_bwd_peak_bytes > 0, (
        "include_backward=True must populate steady_bwd_peak_bytes"
    )
    # Backward peak should be at least as large as the forward peak —
    # backward holds gradients alongside saved activations. Give a 5%
    # tolerance to absorb measurement noise on tiny models.
    assert trace.steady_bwd_peak_bytes >= int(trace.steady_fwd_peak_bytes * 0.95), (
        f"bwd peak {trace.steady_bwd_peak_bytes}B unexpectedly below "
        f"fwd peak {trace.steady_fwd_peak_bytes}B"
    )
    # Per-block dict: at least one block recorded a bwd peak when the
    # block discovery succeeded. Tiny-GPT-2 has H blocks; if discovery
    # failed (e.g. unrecognised layout) the dict may be empty — only
    # assert positivity when it's populated.
    if trace.steady_fwd_block_peak_bytes:
        assert trace.steady_bwd_block_peak_bytes, (
            "fwd per-block peaks recorded but bwd per-block peaks empty — "
            "register_full_backward_hook may not be firing on the discovered blocks"
        )
        for bid, peak in trace.steady_bwd_block_peak_bytes.items():
            assert peak > 0, f"block {bid} has non-positive bwd peak {peak}"


@pytest.mark.gpu
def test_measure_compute_rate_returns_sane_tflops(gpu_device):
    """measure_compute_rate must return a positive TFLOPS measurement."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from axolotl.integrations.protrain.profiler.hw_bench import (
        measure_compute_rate,
    )

    tflops = measure_compute_rate(gpu_device, matrix_size=2048, n_iters=4)
    # Sanity range only — the test's job is to verify ``measure_compute_rate``
    # returns a positive, physically plausible number, not to validate any
    # specific GPU model. Original bracket was 3090-class (5–200 TFLOPS at
    # 2048²); broadened to span a much wider hardware envelope:
    #
    # * Lower bound 1.0: smaller-than-tuned matrices on cuBLAS/cuDNN paths
    #   that haven't been kernel-tuned for the live SKU (e.g. brand-new
    #   sm_120 / Blackwell on a torch wheel built against an older CUDA
    #   minor) hit a fixed kernel-launch floor (~6 ms / 2048² fp16 GEMM
    #   ≈ 2–3 TFLOPS). Still a real, non-broken measurement; just slow.
    # * Upper bound 1000: leaves headroom above peak fp16 tensor-core
    #   rates of current Blackwell-class workstation cards (~200–250
    #   TFLOPS rated) to absorb future SKUs without re-touching the
    #   bracket. Anything above this would indicate event-timer division
    #   blowing up (e.g. ``median_iter ≈ 0``).
    assert 1.0 < tflops < 1000.0, (
        f"compute rate {tflops:.1f} TFLOPS outside sane physical range"
    )


def test_measure_nccl_single_rank_returns_empty_tuple():
    """Single-rank fast path: ``({}, {})`` so the searcher's collective term collapses."""
    from axolotl.integrations.protrain.profiler.hw_bench import measure_nccl

    gather, reduce = measure_nccl(world_size=1)
    assert gather == {}
    assert reduce == {}


def test_measure_nccl_multi_rank_without_dist_raises():
    """world_size>1 without an initialized process group must fail loudly."""
    import torch.distributed as dist

    from axolotl.integrations.protrain.profiler.hw_bench import measure_nccl

    if dist.is_available() and dist.is_initialized():
        pytest.skip(
            "torch.distributed is initialized in this environment; "
            "cannot validate the not-initialized error path."
        )
    with pytest.raises(RuntimeError, match="not initialized|torchrun"):
        measure_nccl(world_size=2)


@pytest.mark.gpu
def test_hw_bench_pcie_returns_positive(gpu_device):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    h2d, d2h = measure_pcie(gpu_device, n_bytes=16 * 1024 * 1024, n_iters=2)
    assert h2d > 0
    assert d2h > 0
    # 200 GB/s is well above PCIe 5.0 x16 theoretical (~63 GB/s); trips if we
    # accidentally divide by the wrong unit.
    assert h2d < 200e9
    assert d2h < 200e9


@pytest.mark.gpu
def test_trace_records_op_latencies(gpu_device):
    """Profiler must populate ``trace.op_latencies`` with measured per-op times."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    _name, tok, model = _load_tiny_gpt2()
    model = model.to(device)

    bs, seq = 1, 64
    batch = _build_batch(tok, bs, seq, device)

    cfg = ProfilerConfig(
        batch_size=bs,
        seq_len=seq,
        device=str(device),
        include_backward=True,
        on_demand=False,
    )

    trace = run_trace(model, batch, cfg)

    # Must be non-empty — if this fails we regressed the capture path.
    assert trace.op_latencies, "trace.op_latencies must be populated"

    # Every recorded latency is positive and well under 1s on tiny-GPT-2;
    # the latter trips if elapsed_ms is not converted to seconds.
    for op_id, lat in trace.op_latencies.items():
        assert lat > 0.0, f"op {op_id} has non-positive latency {lat}"
        assert lat < 1.0, f"op {op_id} latency {lat}s exceeds sanity ceiling"

    # Coverage: at least 80% of ops in op_order must have a latency entry.
    # (Some edge-case modules may fire a pre-hook but no post-hook if
    # forward re-enters the same module id; skip those.)
    n_ops = len(trace.op_order)
    n_covered = sum(1 for op in trace.op_order if op.op_id in trace.op_latencies)
    assert n_covered / max(1, n_ops) >= 0.80, (
        f"only {n_covered}/{n_ops} ops have latencies — coverage too low"
    )


def test_on_demand_disabled_fast_path():
    """Disabled OnDemandTensorMgr must be a no-op context manager."""
    mgr = OnDemandTensorMgr(device="cuda:0", disabled=True)
    with mgr as entered:
        assert entered is mgr
        # Disabled path must not raise on allocate/free.
        fake_op = OpRecord(
            op_id=OpId(0),
            module_path="x",
            qualified_name="X",
            shape_signature=((),),
            block_id=None,
            is_forward=True,
        )
        mgr.allocate_inputs(fake_op)
        mgr.free_after(fake_op)
    assert tuple(mgr.live_tensor_ids()) == ()


def test_on_demand_enabled_requires_model():
    """Enabled mode must reject construction without a model."""
    mgr = OnDemandTensorMgr(device="cuda:0", disabled=False)
    with pytest.raises(ValueError, match="requires a model"):
        mgr.__enter__()


@pytest.mark.gpu
def test_on_demand_enabled_param_offload_and_restore(gpu_device):
    """Enabled OnDemandTensorMgr offloads params and restores them byte-exact."""
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    ).to(device)

    # Snapshot original params so we can verify byte-exact restore later.
    original_state = {name: p.detach().clone() for name, p in model.named_parameters()}

    from axolotl.integrations.protrain.profiler.on_demand import (
        OnDemandTensorMgr,
    )

    mgr = OnDemandTensorMgr(device=device, disabled=False, model=model)

    x = torch.randn(4, 64, device=device)
    with mgr:
        # Inside the context, before any forward, params should be empty
        # placeholders (storage of size 0). The pre-forward hooks will
        # gather them just before each Linear's forward.
        for _name, p in model.named_parameters():
            assert p.data.numel() == 0, (
                f"expected empty placeholder under on-demand, got "
                f"{p.data.numel()} elements"
            )

        out = model(x)
        # Forward must produce a sane output of the right shape.
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

    # After exiting, params restored to GPU with original values.
    for name, p in model.named_parameters():
        assert p.device.type == "cuda"
        assert torch.allclose(p, original_state[name], atol=0, rtol=0), (
            f"param {name} did not restore byte-exact under OnDemandTensorMgr"
        )


@pytest.mark.gpu
def test_on_demand_engaged_path_in_run_trace(gpu_device, monkeypatch):
    """run_trace engages on-demand when params exceed the size threshold.

    Forces the threshold down to ~0% so a tiny model takes the on-demand
    branch. The trace must still complete and populate op records.
    """
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")

    # Simple two-block "transformer" — enough to exercise multiple modules
    # under the on-demand gather/release path. Use a non-Linear container
    # so the trace's block heuristic still picks it up.
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 32)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

        def forward(self, input_ids=None, **kwargs):
            x = input_ids.to(torch.float32)
            for layer in self.layers:
                x = layer(x)
            return type("Out", (), {"loss": x.sum()})()

    model = TinyModel().to(device)
    batch = {
        "input_ids": torch.randn(2, 32, device=device),
    }

    # Force on-demand to engage by dropping the threshold to 0%.
    from axolotl.integrations.protrain.profiler import trace as trace_mod

    monkeypatch.setattr(trace_mod, "ON_DEMAND_STATE_BYTES_FRACTION", 0.0)

    cfg = ProfilerConfig(
        batch_size=2,
        seq_len=32,
        device=str(device),
        include_backward=False,
        on_demand=True,
    )
    trace = run_trace(model, batch, cfg)

    # Trace must have op records — the on-demand path didn't drop ops.
    assert len(trace.op_order) > 0
    # Forward-only trace: no <backward> op record expected.
    assert all(op.is_forward for op in trace.op_order)
    # Activation sizes captured for at least the inferred blocks (the layers
    # ModuleList children get block_id=0, 1 via the ``layers.<i>`` heuristic).
    assert len(trace.activation_sizes) >= 1, (
        "on-demand trace did not record any activation sizes"
    )


@pytest.mark.gpu
def test_on_demand_engaged_cost_model_finite(gpu_device, monkeypatch):
    """Cost model must produce a finite, positive iter-time on an on-demand trace.

    Smoke-test, not calibration: assert ``estimate_runtime`` is in
    ``(0, 60s)`` so we catch the "roofline collapse predicts hours"
    failure mode when on-demand traces feed inflated peak / activation /
    delta numbers into the cost model. The 60s ceiling is loose enough
    to absorb measurement noise on tiny models without ever masking the
    nonsense-prediction regression.
    """
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")

    # Same shape as test_on_demand_engaged_path_in_run_trace — two stacked
    # tiny blocks so block-id inference picks them up at indices 0, 1.
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 32)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

        def forward(self, input_ids=None, **kwargs):
            x = input_ids.to(torch.float32)
            for layer in self.layers:
                x = layer(x)
            return type("Out", (), {"loss": x.sum()})()

    model = TinyModel().to(device)
    batch = {"input_ids": torch.randn(2, 32, device=device)}

    # Force on-demand engagement after Fix 3's rename.
    from axolotl.integrations.protrain.profiler import trace as trace_mod

    monkeypatch.setattr(trace_mod, "ON_DEMAND_STATE_BYTES_FRACTION", 0.0)

    cfg_profile = ProfilerConfig(
        batch_size=2,
        seq_len=32,
        device=str(device),
        include_backward=False,
        on_demand=True,
    )
    trace = run_trace(model, batch, cfg_profile)

    # Build a tiny synthetic ChunkLayout that's consistent with the trace's
    # block count. The cost model only cares that block_to_chunks covers
    # every block in trace.activation_sizes; a 1-chunk-per-block layout is
    # the simplest valid topology for this smoke test.
    from axolotl.integrations.protrain.block.layout_rules import assign_modes
    from axolotl.integrations.protrain.cost import estimate_runtime
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        ChunkLayout,
        CostConfig,
        HardwareProfile,
        ParamId,
    )

    block_ids = sorted(trace.activation_sizes.keys())
    n_block = len(block_ids)
    assert n_block >= 1, "trace must have at least one inferred block"

    n_chunk = max(n_block, 1)
    chunks = tuple((ParamId(f"p.{i}"),) for i in range(n_chunk))
    param_to_chunk = {ParamId(f"p.{i}"): i for i in range(n_chunk)}
    block_to_chunks = {_BlockId(int(bid)): (i,) for i, bid in enumerate(block_ids)}
    layout = ChunkLayout(
        S_chunk=4 * (1 << 20),  # 4 MiB; tiny but positive
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks=block_to_chunks,
    )

    hw = HardwareProfile(
        gpu_sku=trace.sku,
        gpu_memory_bytes=int(torch.cuda.get_device_properties(device).total_memory),
        gpu_count=1,
        pcie_h2d_bps=trace.pcie_h2d_bps if trace.pcie_h2d_bps > 0 else 12e9,
        pcie_d2h_bps=trace.pcie_d2h_bps if trace.pcie_d2h_bps > 0 else 12e9,
        has_nvlink=False,
        cpu_adam_bytes_per_sec=trace.cpu_adam_bytes_per_sec,
        gpu_adam_bytes_per_sec=trace.gpu_adam_bytes_per_sec,
        gpu_compute_tflops=trace.compute_rate_tflops,
    )

    cost_cfg = CostConfig(n_persist=1, n_buffer=1, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(0, 0, n_block)

    iter_s = estimate_runtime(cost_cfg, trace, layout, block_map, hw)

    import math

    assert math.isfinite(iter_s), f"iter_s is not finite: {iter_s}"
    assert iter_s > 0.0, f"iter_s must be positive, got {iter_s}"
    # 60s ceiling: a tiny model on a 3090 should never predict more than
    # seconds. Trips if on-demand traces feed inflated peak / activation
    # numbers into the cost model and the roofline collapses to hours.
    assert iter_s < 60.0, (
        f"iter_s={iter_s:.2f}s exceeds 60s ceiling — on-demand trace "
        "may have produced inflated activation/delta numbers that broke "
        "the cost model's roofline. Inspect trace.activation_sizes / "
        "intra_op_delta / inter_op_delta."
    )


@pytest.mark.gpu
def test_on_demand_backward_under_unpack_hook(gpu_device):
    """Backward under on-demand must not crash on CPU/CUDA mismatch.

    Regression: ``_unpack_hook`` previously returned the spilled CPU tensor
    as-is, so a CUDA backward landed on a CPU saved tensor and exploded
    deep in autograd C++. The fix routes the unpack copy through
    ``self.device`` so backward sees a CUDA tensor.
    """
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    ).to(device)

    mgr = OnDemandTensorMgr(device=device, disabled=False, model=model)

    # x must require grad so the full_backward_pre_hooks fire on the first
    # Linear (PyTorch skips them when no input gradient flow is needed).
    x = torch.randn(2, 32, device=device, requires_grad=True)

    with mgr:
        out = model(x)
        loss = out.sum()
        loss.backward()

    # Every trainable param must have a finite, non-None grad after backward.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no grad after backward"
        assert torch.isfinite(p.grad).all(), f"{name} grad is not finite"


@pytest.mark.gpu
def test_on_demand_intra_delta_excludes_gather(gpu_device, monkeypatch):
    """Regression: on-demand pre-gather hook must fire BEFORE trace's pre_forward.

    Pre-fix, the trace driver registered its ``_pre_forward`` hook on every
    module before ``OnDemandTensorMgr.__enter__`` ran. PyTorch fires
    forward_pre hooks in registration order, so the trace's
    ``allocated_before`` snapshot was taken BEFORE on-demand's ``_pre_gather``
    materialized the GPU param. ``intra_op_delta = peak − allocated_before``
    therefore absorbed the gather bytes (full weight + bias) for every leaf
    op, inflating the cost model's peak reconstruction.

    The fix: register the on-demand pre-gather hook with ``prepend=True`` so
    it fires first; the trace's baseline then already includes the gather,
    and intra_op_delta captures only workspace + output.

    This test forces on-demand engagement on a tiny model whose Linear
    weight bytes (256 * 256 * 4 = 256 KiB) dwarf the per-op output bytes
    (2 * 256 * 4 = 2 KiB). After the fix, at least one leaf-Linear op must
    have ``intra_op_delta`` strictly less than the per-leaf gather bytes —
    something that was structurally impossible pre-fix because every leaf
    op's delta included the gather as a floor.
    """
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")

    # Param-heavy / output-light Linear so the gather signal dwarfs the
    # legitimate workspace + output. Two stacked layers so the SECOND
    # Linear's intra_op_delta (post cuBLAS workspace warmup) drops to
    # near-zero in the post-fix world but stays >= gather_bytes pre-fix.
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 256)

        def forward(self, x):
            return self.fc(x)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

        def forward(self, input_ids=None, **kwargs):
            x = input_ids.to(torch.float32)
            for layer in self.layers:
                x = layer(x)
            return type("Out", (), {"loss": x.sum()})()

    model = TinyModel().to(device)
    batch = {"input_ids": torch.randn(2, 256, device=device)}

    # Force on-demand to engage on this tiny model.
    from axolotl.integrations.protrain.profiler import trace as trace_mod

    monkeypatch.setattr(trace_mod, "ON_DEMAND_STATE_BYTES_FRACTION", 0.0)

    cfg = ProfilerConfig(
        batch_size=2,
        seq_len=256,
        device=str(device),
        include_backward=False,
        on_demand=True,
    )
    trace = run_trace(model, batch, cfg)

    # Per-leaf gather floor: weight (256 * 256 * 4) + bias (256 * 4)
    # = 263168 bytes. Pre-fix every leaf op's intra_op_delta absorbed at
    # least this. Post-fix, leaf ops past the cuBLAS workspace warmup
    # carry only output + scratch (~2 KiB).
    gather_bytes_floor = 256 * 256 * 4 + 256 * 4

    # Pull intra_op_delta values for every leaf-Linear op in the trace.
    leaf_intra = [
        trace.intra_op_delta[op.op_id]
        for op in trace.op_order
        if op.qualified_name == "Linear" and op.is_forward
    ]
    assert len(leaf_intra) >= 2, (
        f"expected >=2 leaf Linear ops in trace, got {len(leaf_intra)}: "
        f"{[op.qualified_name for op in trace.op_order]}"
    )

    # Sanity ceiling: at least one leaf-Linear's intra_op_delta must be
    # strictly less than the gather floor. Pre-fix every leaf op had at
    # least gather_bytes worth of inflation; post-fix the second leaf
    # (after cuBLAS workspace warmup) lands at ~output bytes only.
    assert min(leaf_intra) < gather_bytes_floor, (
        f"leaf-Linear intra_op_delta values {leaf_intra} all exceed the "
        f"per-leaf gather floor of {gather_bytes_floor} bytes — on-demand "
        f"pre-gather hook is firing AFTER the trace's pre_forward (regression "
        f"of the prepend=True fix)."
    )


def test_delta_since_last_resets_peak_window_no_stale_carryover():
    """``delta_since_last`` must reset the peak window between calls.

    Without the reset, a high-water mark from a prior interval keeps
    re-charging subsequent ``peak_allocated_bytes`` reads, so an
    interval with zero allocation activity would still report a stale
    positive delta. This test models the allocator with a tiny in-test
    fake that respects ``reset_peak_memory_stats`` semantics, so it
    runs on CPU-only hosts.
    """
    import types

    from axolotl.integrations.protrain.profiler.memory_deltas import (
        MemoryDeltaTracker,
    )

    tracker = MemoryDeltaTracker(device=None)

    # Faithful model of the CUDA allocator's bookkeeping: ``current``
    # tracks resident bytes, ``peak`` is the high-water mark since the
    # last ``reset``. The scripted op story:
    #   call 1: baseline (1024 resident, peak=1024)
    #   between 1 and 2: an op spikes to 4096 then frees back to 1024
    #   call 2: should report a 4096-1024 = 3072 delta
    #   between 2 and 3: nothing happens
    #   call 3: should report 0 because the prior call reset the peak
    state = {"current": 1024, "peak": 1024}
    reset_calls = {"n": 0}
    # Indexed pre-call hooks: simulate any allocator activity that
    # happened just before each ``delta_since_last`` invocation.
    pre_call_events = [
        lambda: None,  # before call 1
        lambda: state.update(peak=max(state["peak"], 4096)),  # transient spike
        lambda: None,  # before call 3 — no activity
    ]
    call_idx = {"n": 0}

    def fake_stats(self):
        # Run the pre-call event for this invocation, then surface state.
        pre_call_events[call_idx["n"]]()
        call_idx["n"] += 1
        return {
            "allocated_bytes.all.current": state["current"],
            "allocated_bytes.all.peak": state["peak"],
        }

    def fake_reset(self):
        reset_calls["n"] += 1
        # ``torch.cuda.reset_peak_memory_stats`` clamps the peak down
        # to current; future spikes raise it again.
        state["peak"] = state["current"]

    tracker._stats = types.MethodType(fake_stats, tracker)
    tracker.reset = types.MethodType(fake_reset, tracker)

    # Call 1: establishes baseline, returns 0, AND must reset the peak.
    assert tracker.delta_since_last() == 0
    assert reset_calls["n"] == 1, (
        "first delta_since_last() must reset the peak window so the next "
        "interval starts fresh"
    )

    # Call 2: observes the transient spike, returns the legitimate delta,
    # and resets the peak again so call 3 doesn't see a stale 4096.
    assert tracker.delta_since_last() == 4096 - 1024
    assert reset_calls["n"] == 2

    # Call 3: no new activity. Pre-fix this would return 4096-1024 = 3072
    # because the high-water mark from call 2's interval lingered. Post-
    # fix, the reset cleared peak back to current, so delta == 0.
    assert tracker.delta_since_last() == 0
    assert reset_calls["n"] == 3


@pytest.mark.gpu
def test_measure_chunked_steady_restores_model_and_optimizer_state(gpu_device):
    """``measure_chunked_steady`` must roll back parameters + optimizer state.

    Regression for the round-3 finding: the round-2 fix snapshotted via
    ``state_dict()`` which returns *aliased* tensor references — the
    warmup + timed loops' ``optimizer.step()`` calls then mutated those
    tensors in place, so the snapshot tracked the live (mutated) values
    and ``load_state_dict()`` restored from already-advanced state. The
    fix is to deep-clone every tensor at snapshot time
    (``_clone_state_dict`` walks model + optimizer state_dict() and
    ``.detach().clone()``s each tensor) so restoration writes the true
    pre-measurement values back into the live params.
    """
    import torch
    from torch import nn

    from axolotl.integrations.protrain.profiler.phase2 import measure_chunked_steady

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    device = torch.device(f"cuda:{gpu_device}")

    # Tiny linear model — small enough that a non-zero LR optimizer.step
    # produces a numerically detectable parameter delta in one iteration,
    # so any regression in the snapshot/restore plumbing fails loudly.
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    ).to(device)
    # Use a relatively large LR so step() shifts parameters by a clearly
    # non-trivial amount — equality (rather than near-equality) on the
    # pre/post comparison then proves the snapshot was independent.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    # Pre-call snapshot of param values, kept on CPU and detached so the
    # step()s inside measure_chunked_steady cannot poison the reference.
    pre_params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Tickle the optimizer so its state has Adam moment buffers populated
    # before the snapshot — this exercises the recursive clone path
    # (``optimizer.state_dict()['state']`` is nested) and ensures the
    # restore round-trips both the param tensors AND the moment tensors.
    optimizer.zero_grad(set_to_none=True)
    seed_input = torch.randn(2, 16, device=device)
    seed_loss = model(seed_input).sum()
    seed_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Re-snapshot AFTER the seed step so the test's "pre" baseline is the
    # state immediately before measure_chunked_steady is called.
    pre_params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_optim = {
        pid: {k: v.detach().cpu().clone() for k, v in st.items() if torch.is_tensor(v)}
        for pid, st in optimizer.state_dict()["state"].items()
    }

    # Build a forward batch that produces a backwards-able loss. The
    # tiny module's forward returns a tensor; ``_extract_loss`` accepts a
    # raw scalar tensor, so we wrap the module so loss = output.sum().
    class _LossWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x).sum()

    wrapped = _LossWrap(model).to(device)
    batch = {"x": torch.randn(4, 16, device=device)}

    # ``measure_chunked_steady`` requires backward through ``wrapped``;
    # only the inner module's params are in the optimizer, so the inner
    # step() updates those (the wrap has no extra params).
    fwd_s, bwd_s, step_s, peak = measure_chunked_steady(
        model=wrapped,
        batch=batch,
        optimizer=optimizer,
        n_warmup=2,
        n_iters=2,
    )

    # Sanity: timings + peak must be populated and finite.
    assert fwd_s > 0.0 and bwd_s > 0.0 and step_s >= 0.0
    assert peak > 0

    # Critical assertion: every parameter must be EXACTLY equal to the
    # pre-call value. Pre-fix this fails because the aliased snapshot
    # was mutated by warmup+timed step()s, so load_state_dict restored
    # an already-advanced state.
    post_params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for k, pre in pre_params.items():
        post = post_params[k]
        assert torch.equal(pre, post), (
            f"param {k!r} not restored after measure_chunked_steady: "
            f"max abs diff = {(pre - post).abs().max().item():g}"
        )

    # Optimizer moments must also be exactly restored (Adam tracks
    # ``exp_avg`` / ``exp_avg_sq`` per param; both are mutated by step()).
    post_optim_state = optimizer.state_dict()["state"]
    for pid, pre_state in pre_optim.items():
        post_state = post_optim_state[pid]
        for k, pre_t in pre_state.items():
            post_t = post_state[k].detach().cpu()
            assert torch.equal(pre_t, post_t), (
                f"optimizer state[{pid}][{k!r}] not restored: "
                f"max abs diff = {(pre_t - post_t).abs().max().item():g}"
            )
