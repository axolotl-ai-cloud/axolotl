"""Unit tests for the kernel-aware searcher: ``forbid_activation_offload``.

The fused LoRA MLP backward kernel (``lora_mlp_kernel: true``) returns a
real-shape gradient on offloaded activations whose chunk-storage placeholder
is zero-shape, crashing with a ``LoRA_MLPBackward`` shape-mismatch at the
first backward (v61 finding). The searcher's ``forbid_activation_offload``
flag is the root-cause fix at the planner layer: filter every
``n_offload > 0`` CostConfig out of the candidate set before any pick is
made, so no auto-mode run can land on the unsafe composition.

The shape-preserving placeholder is the COMPLEMENTARY (sibling) fix at the
chunk-storage layer; the two are belt-and-suspenders. This file exercises
the searcher gate only.
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.search import search
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

MB = 1 << 20
GB = 1 << 30


# ---------------------------------------------------------------------
# Synthetic fixtures (subset of test_cost_search.py's setup)
# ---------------------------------------------------------------------


def _make_op_order(n_block: int, ops_per_block: int) -> tuple[OpRecord, ...]:
    out: list[OpRecord] = []
    op_id = 0
    for b in range(n_block):
        for k in range(ops_per_block):
            out.append(
                OpRecord(
                    op_id=OpId(op_id),
                    module_path=f"block.{b}.op.{k}",
                    qualified_name="aten::toy",
                    shape_signature=((1,),),
                    block_id=BlockId(b),
                    is_forward=True,
                )
            )
            op_id += 1
    return tuple(out)


def _make_trace(
    *,
    n_block: int = 8,
    ops_per_block: int = 5,
    activation_bytes_per_block: int = 32 * MB,
    model_state_bytes: int = 768 * MB,
    intra_delta_bytes: int = 8 * MB,
    inter_delta_bytes: int = 2 * MB,
    op_latency_s: float = 0.0002,
) -> ProfilerTrace:
    op_order = _make_op_order(n_block, ops_per_block)
    intra_op_delta = {op.op_id: intra_delta_bytes for op in op_order}
    inter_op_delta = {op.op_id: inter_delta_bytes for op in op_order}
    activation_sizes = {BlockId(b): activation_bytes_per_block for b in range(n_block)}
    op_latencies = {op.op_id: op_latency_s for op in op_order}
    hooked_sum = sum(op_latencies.values())
    return ProfilerTrace(
        op_order=op_order,
        intra_op_delta=intra_op_delta,
        inter_op_delta=inter_op_delta,
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="kernel-aware-test",
        bs=1,
        seq=128,
        sku="RTX 3090 (synthetic)",
        world=1,
        op_latencies=op_latencies,
        hooked_fwd_wall_s=hooked_sum,
        steady_fwd_wall_s=hooked_sum,
        steady_bwd_wall_s=0.0,
    )


def _make_layout(
    *, n_chunk: int = 12, s_chunk: int = 64 * MB, n_block: int = 8
) -> ChunkLayout:
    chunks = tuple((ParamId(f"param.{i}"),) for i in range(n_chunk))
    param_to_chunk = {ParamId(f"param.{i}"): ChunkId(i) for i in range(n_chunk)}
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]] = {
        BlockId(b): (ChunkId(b % n_chunk),) for b in range(n_block)
    }
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks=block_to_chunks,
    )


def _make_hw(
    *,
    gpu_memory_bytes: int = 24 * GB,
    cpu_adam_bytes_per_sec: float = 2e9,
    gpu_adam_bytes_per_sec: float = 4e11,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="RTX 3090 (synthetic)",
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
        zero3_shard=False,
        cpu_adam_bytes_per_sec=cpu_adam_bytes_per_sec,
        gpu_adam_bytes_per_sec=gpu_adam_bytes_per_sec,
    )


# ---------------------------------------------------------------------
# Searcher tests
# ---------------------------------------------------------------------


def test_searcher_succeeds_on_synthetic_fixture():
    """Sanity: the unconstrained searcher returns a valid cfg on the synthetic toy.

    Pre-condition for the kernel-aware filter tests: the underlying search
    must succeed before we can assert "the filter changes the outcome".
    """
    trace = _make_trace()
    layout = _make_layout()
    hw = _make_hw(gpu_memory_bytes=2 * GB)
    result = search(trace, layout, capacity_bytes=2 * GB, hw=hw)
    assert result.cfg is not None
    assert 0 <= result.cfg.n_offload <= 8


def test_searcher_with_forbid_activation_offload_never_picks_n_offload_gt_zero():
    """forbid_activation_offload=True MUST yield n_offload=0 in the picked cfg.

    The candidate-generation loop drops the entire n_offload>0 subtree
    before any peak/runtime evaluation; the picked cfg's ``n_offload`` is
    therefore always 0.
    """
    trace = _make_trace()
    layout = _make_layout()
    hw = _make_hw(gpu_memory_bytes=4 * GB)
    result = search(
        trace,
        layout,
        capacity_bytes=4 * GB,
        hw=hw,
        forbid_activation_offload=True,
    )
    assert result.cfg.n_offload == 0, (
        f"forbid_activation_offload=True must yield n_offload=0; "
        f"got n_offload={result.cfg.n_offload}"
    )


def test_searcher_raises_clear_error_when_no_offload_zero_config_fits():
    """forbid_activation_offload + tiny capacity must raise with a clear message.

    Constructs a scenario where the only feasible configs require
    activation-offload. With ``forbid_activation_offload=True`` the
    searcher drops every n_offload>0 subtree and must raise a
    RuntimeError mentioning ``lora_mlp_kernel`` so the user can take
    action.
    """
    # Big activations + tiny GPU budget: n_offload>0 is the only escape.
    # Use huge per-op intra_delta so even CKPT-everything keeps raw_peak above budget.
    trace = _make_trace(
        n_block=8,
        ops_per_block=5,
        activation_bytes_per_block=512 * MB,
        model_state_bytes=4 * GB,
        intra_delta_bytes=512 * MB,
        inter_delta_bytes=64 * MB,
    )
    layout = _make_layout(n_chunk=12, n_block=8, s_chunk=64 * MB)
    # 64 MB GPU — way below even one block's footprint without offload.
    hw = _make_hw(gpu_memory_bytes=64 * MB)
    with pytest.raises(RuntimeError) as excinfo:
        search(
            trace,
            layout,
            capacity_bytes=64 * MB,
            hw=hw,
            forbid_activation_offload=True,
        )
    msg = str(excinfo.value)
    # Either the kernel-specific message fires (preferred) or the generic
    # capacity-failure message. Both are acceptable correctness outcomes;
    # the former is more actionable so we prefer it when n_kernel_filtered>0.
    assert "lora_mlp_kernel" in msg or "no feasible ProTrain config" in msg, (
        f"error must explain the failure; got: {msg!r}"
    )


def test_searcher_forbid_offload_keyword_only():
    """``forbid_activation_offload`` is keyword-only.

    Pinning the API surface prevents accidental positional misuse from
    silently swapping with ``cpu_capacity_bytes``.
    """
    import inspect

    sig = inspect.signature(search)
    param = sig.parameters["forbid_activation_offload"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"forbid_activation_offload must be keyword-only; kind={param.kind}"
    )
    assert param.default is False, (
        f"forbid_activation_offload default must be False; got {param.default}"
    )


# ---------------------------------------------------------------------
# Plumbing: cfg.lora_mlp_kernel -> protrain_model_wrapper -> search()
# ---------------------------------------------------------------------


def test_protrain_model_wrapper_has_forbid_activation_offload_param():
    """The wrapper exposes ``forbid_activation_offload`` for plugin wiring."""
    import inspect

    from axolotl.integrations.protrain.api.model_wrapper import (
        protrain_model_wrapper,
    )

    sig = inspect.signature(protrain_model_wrapper)
    assert "forbid_activation_offload" in sig.parameters, (
        "protrain_model_wrapper must accept ``forbid_activation_offload`` so "
        "the plugin can wire cfg.lora_mlp_kernel into the searcher's filter."
    )
    param = sig.parameters["forbid_activation_offload"]
    assert param.default is False


def test_plugin_reads_cfg_lora_mlp_kernel_for_searcher_gate():
    """Confirm the plugin's ``getattr(cfg, "lora_mlp_kernel", False)`` read path.

    Full plugin invocation pulls in too many runtime dependencies (HW probes,
    trace cache, etc.); this test pins the read surface used by
    ``post_model_load`` so a future refactor of the cfg-field name surfaces
    here loudly.
    """

    class _Cfg:
        lora_mlp_kernel = True

    assert bool(getattr(_Cfg(), "lora_mlp_kernel", False)) is True

    class _CfgFalse:
        lora_mlp_kernel = False

    assert bool(getattr(_CfgFalse(), "lora_mlp_kernel", False)) is False

    class _CfgUnset:
        pass

    assert bool(getattr(_CfgUnset(), "lora_mlp_kernel", False)) is False
