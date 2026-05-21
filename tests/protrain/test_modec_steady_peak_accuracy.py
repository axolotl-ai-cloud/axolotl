"""bnb-4-bit Mode-C steady-peak: predictor must charge the full ckpt-chain residual sum across all CKPT blocks."""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.cost.memory import estimate_peak
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ParamId,
    ProfilerTrace,
)

GiB = 1 << 30


# Llama-30B (huggyllama/llama-30b) architecture from
# ``m0_artifacts/ext_30b_seq{512,1024,2048}.yml``:
#     num_hidden_layers     = 60
#     hidden_size           = 6656
#     intermediate_size     = 17920
#     num_attention_heads   = 52
#     vocab_size            = 32000
LLAMA_30B_N_BLOCK = 60
LLAMA_30B_INTERMEDIATE = 17920

# Audit Mode-C cfg knobs (identical across the three seq runs; see
# ``m0_artifacts/ext_30b_seq2048.yml``):
#     protrain_n_persist_override:    0
#     protrain_n_buffer_override:     12
#     protrain_n_swap_override:       0
#     protrain_n_checkpoint_override: 60
N_PERSIST = 0
N_BUFFER = 12
N_SWAP = 0
N_CHECKPOINT = 60

# Layout knobs observed in every log: ``layout built: S_chunk=67108864
# N_chunk=302``. ``layout.mandatory_persistent`` was [0, 300, 301] per
# the wrapper's residency = prefix[0..0) | mandatory line — 3 chunks
# pinned by layout regardless of n_persist.
S_CHUNK = 67108864  # 64 MiB
N_CHUNK = 302
MANDATORY_PERSISTENT_IDS = (0, 300, 301)

# Measured steady-state peaks (GiB) from empirical 30B 4-bit Mode-C runs at three seq lengths.
MEASURED_STEADY_GIB = {
    512: 2.91,
    1024: 3.50,
    2048: 4.68,
}

# 30B QLoRA model-state aggregate seen in the audit runs. Approximate:
# frozen base @ 4-bit ≈ 15 GiB; tiny LoRA adapters ≈ 100 MiB x 16 bytes
# (param+grad+fp32 master+m+v) ≈ 1.6 GiB. The trace's
# ``_count_model_state_bytes`` records these as a single aggregate; the
# cost model's ``model_state_present_bytes`` clamps
# ``persistent_factor = max(1.0, model_state_bytes / fp16_total)`` so
# the exact value matters only when it exceeds ``N_chunk * S_chunk``
# (18.875 GiB here). 16 GiB lands BELOW that threshold ⇒
# ``persistent_factor`` clamps to 1.0 — matching the audit logs'
# implicit assumption (the wrapper's ``peak prediction calibrated
# 0.00 -> 2.54 GB`` line ONLY makes sense at ``persistent_factor=1.0``).
MODEL_STATE_BYTES_30B_QLORA = 16 * GiB


def _build_layout() -> ChunkLayout:
    """Reconstruct the audit's chunk layout (N_chunk=302 x 64 MiB) with the three layout-mandatory chunks pinned.
    """
    chunks = tuple((ParamId(f"p.{cid}"),) for cid in range(N_CHUNK))
    param_to_chunk = {ParamId(f"p.{cid}"): ChunkId(cid) for cid in range(N_CHUNK)}
    # Single dummy block_to_chunks entry (the audit n_offload=0 cfg
    # never reads this map — estimate_peak only walks
    # trace.activation_sizes and trace.op_order).
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]] = {
        BlockId(b): (ChunkId(b % N_CHUNK),) for b in range(LLAMA_30B_N_BLOCK)
    }
    return ChunkLayout(
        S_chunk=S_CHUNK,
        N_chunk=N_CHUNK,
        chunks=chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks=block_to_chunks,
        mandatory_persistent=frozenset(
            ChunkId(cid) for cid in MANDATORY_PERSISTENT_IDS
        ),
    )


def _build_synth_trace(seq_len: int) -> ProfilerTrace:
    """Reconstruct synth_trace_from_overrides output (empty op_order, FFN-intermediate activation proxy)."""
    bs = 1  # audit cfg: micro_batch_size: 1
    per_block_act_bytes = int(bs) * int(seq_len) * int(LLAMA_30B_INTERMEDIATE) * 2
    activation_sizes = {
        BlockId(b): per_block_act_bytes for b in range(LLAMA_30B_N_BLOCK)
    }
    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes=activation_sizes,
        model_state_bytes=int(MODEL_STATE_BYTES_30B_QLORA),
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="huggyllama/llama-30b-qlora-modec",
        bs=bs,
        seq=int(seq_len),
        sku="NVIDIA RTX PRO 6000 Blackwell (audit)",
        world=1,
    )


def _build_hw_4bit() -> HardwareProfile:
    """HW profile with dominant_param_bytes_per_element=0.5 to route estimate_peak through the 4-bit alpha branch."""
    return HardwareProfile(
        gpu_sku="NVIDIA RTX PRO 6000 Blackwell (audit)",
        gpu_memory_bytes=24 * GiB,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        zero3_shard=False,
        cpu_adam_bytes_per_sec=2e9,
        gpu_adam_bytes_per_sec=4e11,
        dominant_param_bytes_per_element=0.5,
    )


# Band absorbs wrapper-side calibration offset, intermediate-vs-hidden proxy slack, and per-dtype alpha shift.
TOLERANCE_FRAC = 0.35


@pytest.mark.parametrize("seq_len", [512, 1024, 2048])
def test_modec_steady_peak_within_tolerance(seq_len: int) -> None:
    """estimate_peak lands within +/-35% of the measured steady peak across seq=512/1024/2048."""
    layout = _build_layout()
    trace = _build_synth_trace(seq_len)
    hw = _build_hw_4bit()
    cfg = CostConfig(
        n_persist=N_PERSIST,
        n_buffer=N_BUFFER,
        n_swap=N_SWAP,
        n_checkpoint=N_CHECKPOINT,
        n_offload=0,
    )
    block_map = assign_modes(N_SWAP, N_CHECKPOINT, LLAMA_30B_N_BLOCK)

    predicted_bytes = estimate_peak(cfg, trace, layout, block_map, hw)
    predicted_gib = predicted_bytes / GiB
    measured_gib = MEASURED_STEADY_GIB[seq_len]
    relative_error = abs(predicted_gib - measured_gib) / measured_gib

    assert relative_error <= TOLERANCE_FRAC, (
        f"30B 4-bit Mode-C seq={seq_len}: predicted_peak={predicted_gib:.3f} GiB "
        f"vs measured_steady={measured_gib:.3f} GiB; relative_error={relative_error:.3f} "
        f"(tolerance +/-{TOLERANCE_FRAC:.2f}). "
        f"Check the ckpt_chain_bytes accumulator in cost/memory.py::estimate_peak "
        f"and the raw_peak == 0 fallback."
    )


def test_modec_steady_peak_scales_with_seq() -> None:
    """Predicted peak must grow with sequence length on Mode-C; flat-output regression is the failure mode."""
    layout = _build_layout()
    hw = _build_hw_4bit()
    cfg = CostConfig(
        n_persist=N_PERSIST,
        n_buffer=N_BUFFER,
        n_swap=N_SWAP,
        n_checkpoint=N_CHECKPOINT,
        n_offload=0,
    )
    block_map = assign_modes(N_SWAP, N_CHECKPOINT, LLAMA_30B_N_BLOCK)

    predictions: list[tuple[int, int]] = []
    for seq_len in (512, 1024, 2048):
        trace = _build_synth_trace(seq_len)
        peak_bytes = estimate_peak(cfg, trace, layout, block_map, hw)
        predictions.append((seq_len, peak_bytes))

    # Strict monotonicity in seq_len. Each doubling of seq_len doubles
    # the per-block activation contribution (synth proxy is linear in
    # seq); the CKPT-chain sum across 60 blocks therefore doubles too,
    # and the prediction must grow.
    for (seq_a, peak_a), (seq_b, peak_b) in zip(
        predictions, predictions[1:], strict=False
    ):
        assert peak_b > peak_a, (
            f"predicted peak must grow with sequence length: "
            f"seq={seq_a} -> {peak_a / GiB:.3f} GiB but "
            f"seq={seq_b} -> {peak_b / GiB:.3f} GiB (expected strict increase). "
            f"This breaks the per-seq scaling guarantee."
        )

    # Sanity: the seq=2048 prediction must grow by at least
    # ``2 * N_block * (1024 * intermediate * 2 bytes) * alpha_4bit``
    # relative to seq=1024 — the chain contribution scales linearly
    # with seq, so doubling seq adds at least that much to raw_peak.
    expected_min_delta = int(
        0.75  # ALPHA_FRAGMENTATION_4BIT
        * LLAMA_30B_N_BLOCK
        * 1024
        * LLAMA_30B_INTERMEDIATE
        * 2
        * 0.5  # half-credit slack for cap / rounding interactions
    )
    actual_delta = predictions[2][1] - predictions[1][1]
    assert actual_delta >= expected_min_delta, (
        f"seq=1024 -> 2048 should add >= "
        f"{expected_min_delta / GiB:.2f} GiB via the CKPT-chain term; "
        f"got delta={actual_delta / GiB:.2f} GiB. Suggests the "
        f"``ckpt_chain_bytes`` accumulator is dropping CKPT blocks."
    )
