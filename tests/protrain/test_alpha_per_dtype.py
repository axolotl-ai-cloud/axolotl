"""Pin the per-dtype α fragmentation factor lookup.

Coverage audit Block G (Phase 2) re-derived the empirical α=1.10
fragmentation factor against the M5 / M0-spike / Block-A matrices
and found:

- fp16 / bf16 (2 B/element): α_measured ≈ 0.96 → α=1.10 is mildly
  conservative; keep.
- bnb 8-bit (1 B/element): α_measured ≈ 0.93 → α=1.10 is mildly
  conservative; keep. (Activation / gradient streams stay fp16
  even when base weights are int8, so the fragmentation profile
  is fp16-like.)
- bnb 4-bit Mode-A (0.5 B/element via ``Params4bit``'s
  2-elements-per-uint8 packing): α_measured ≈ 0.70 → α=1.10
  over-predicts by ~37%. Drop to α=0.75 (slightly conservative
  vs the empirical floor).

This test pins the per-dtype lookup in
``cost/memory.py::alpha_fragmentation_for_dtype`` so a future
recalibration cannot silently regress the 4-bit branch.
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.cost.memory import (
    ALPHA_FRAGMENTATION,
    ALPHA_FRAGMENTATION_4BIT,
    alpha_fragmentation_for_dtype,
)


def test_constants_have_expected_values():
    """Lock the two named constants so unrelated edits cannot drift
    the calibration silently."""
    assert ALPHA_FRAGMENTATION == pytest.approx(1.10)
    assert ALPHA_FRAGMENTATION_4BIT == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("bpe", "expected_alpha", "description"),
    [
        # fp32 — α=1.10 (the >=1.0 branch).
        (4.0, ALPHA_FRAGMENTATION, "fp32 weights → α=1.10"),
        # fp16 / bf16 — α=1.10 (paper default; Block G α_measured ≈ 0.96).
        (2.0, ALPHA_FRAGMENTATION, "fp16/bf16 weights → α=1.10"),
        # bnb 8-bit — α=1.10 (Block G α_measured ≈ 0.93; mildly conservative).
        (1.0, ALPHA_FRAGMENTATION, "bnb 8-bit weights → α=1.10"),
        # bnb 4-bit (Params4bit) — α=0.75 (Block G α_measured ≈ 0.70).
        (0.5, ALPHA_FRAGMENTATION_4BIT, "bnb 4-bit weights → α=0.75"),
    ],
)
def test_alpha_lookup_by_dtype(bpe: float, expected_alpha: float, description: str):
    assert alpha_fragmentation_for_dtype(bpe) == pytest.approx(expected_alpha), (
        description
    )


def test_alpha_lookup_threshold_is_one_byte():
    """The fp16/8-bit-vs-4-bit cutoff is exactly 1.0 B/element.

    Values < 1.0 are routed to the 4-bit α; values >= 1.0 (including
    exactly 1.0 for bnb int8) are routed to the fp16 α.
    """
    # Strictly below the cutoff — 4-bit branch.
    assert alpha_fragmentation_for_dtype(0.99) == pytest.approx(
        ALPHA_FRAGMENTATION_4BIT
    )
    # Exactly at the cutoff — fp16 branch (8-bit is conservative-ish, keep α=1.10).
    assert alpha_fragmentation_for_dtype(1.0) == pytest.approx(ALPHA_FRAGMENTATION)
    # Strictly above the cutoff — fp16 branch.
    assert alpha_fragmentation_for_dtype(1.01) == pytest.approx(ALPHA_FRAGMENTATION)


def test_alpha_lookup_extreme_bpe_does_not_crash():
    """Boundary / out-of-range inputs land in one of the two known branches.

    A future calibration may add bands (e.g. fp4 vs nf4 at 0.5
    B/element, fp8 at 1.0 B/element with a tighter α), but today
    the function is binary: 4-bit branch (<1.0) vs fp16 branch
    (>=1.0). Pin both extremes so a future refactor that introduces
    NaN / zero / negative handling has to update this test on
    purpose.
    """
    # Tiny positive value — still routes to 4-bit branch.
    assert alpha_fragmentation_for_dtype(0.001) == pytest.approx(
        ALPHA_FRAGMENTATION_4BIT
    )
    # Zero — by the documented rule (< 1.0) routes to 4-bit branch.
    assert alpha_fragmentation_for_dtype(0.0) == pytest.approx(ALPHA_FRAGMENTATION_4BIT)
    # Negative — by the documented rule (< 1.0) routes to 4-bit branch.
    # Real callers should never pass negative; this just locks behaviour
    # so a future ``max(0, bpe)`` guard is opt-in.
    assert alpha_fragmentation_for_dtype(-1.0) == pytest.approx(
        ALPHA_FRAGMENTATION_4BIT
    )
    # Very large value — fp16 branch.
    assert alpha_fragmentation_for_dtype(1024.0) == pytest.approx(ALPHA_FRAGMENTATION)


def test_dominant_param_dtype_detector_default_for_fp16_model():
    """The detector in ``model_wrapper`` returns 2.0 (fp16) for a
    typical bf16 model — keeping the α=1.10 ceiling unchanged for
    non-quantized callers.
    """
    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.model_wrapper import (
        _detect_dominant_param_bytes_per_element,
    )

    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Two layers' worth of bf16 weights — dominant by aggregate count.
            self.w1 = nn.Parameter(torch.zeros(128, 64, dtype=torch.bfloat16))
            self.w2 = nn.Parameter(torch.zeros(64, 32, dtype=torch.bfloat16))
            # A small fp32 buffer (layer-norm-scale-shaped) that should NOT
            # flip the dominant classification despite element_size=4.
            self.ln = nn.Parameter(torch.zeros(32, dtype=torch.float32))

    bpe = _detect_dominant_param_bytes_per_element(_Toy())
    assert bpe == pytest.approx(2.0), (
        f"bf16 model with a small fp32 LN param should classify as bpe=2.0, got {bpe}"
    )


def test_dominant_param_dtype_detector_returns_default_on_empty_model():
    """The detector falls back to 2.0 (fp16/bf16) when the model has
    no parameters — matches the HardwareProfile default so the
    cost model picks α=1.10 in the absence of signal."""
    from torch import nn

    from axolotl.integrations.protrain.api.model_wrapper import (
        _detect_dominant_param_bytes_per_element,
    )

    class _Empty(nn.Module):
        pass

    assert _detect_dominant_param_bytes_per_element(_Empty()) == pytest.approx(2.0)


def test_dominant_param_dtype_detector_classifies_int8_dominant_model():
    """A model where the bulk of the logical-element mass is int8
    (e.g. bnb 8-bit base) but with bf16 LoRA factors on top classifies
    as bpe=1.0, landing on the conservative α=1.10."""
    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.model_wrapper import (
        _detect_dominant_param_bytes_per_element,
    )

    class _Int8Heavy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Large int8-storage weight (analog for bnb int8 base) — the
            # numel here is the logical-element count too (int8 is 1:1).
            self.base_w = nn.Parameter(
                torch.zeros(4096, 4096, dtype=torch.uint8), requires_grad=False
            )
            # Small bf16 LoRA factors on top.
            self.lora_a = nn.Parameter(torch.zeros(16, 4096, dtype=torch.bfloat16))
            self.lora_b = nn.Parameter(torch.zeros(4096, 16, dtype=torch.bfloat16))

    bpe = _detect_dominant_param_bytes_per_element(_Int8Heavy())
    assert bpe == pytest.approx(1.0), (
        f"int8-dominant model should classify as bpe=1.0, got {bpe}"
    )
    # And the lookup routes it to the conservative α=1.10.
    assert alpha_fragmentation_for_dtype(bpe) == pytest.approx(ALPHA_FRAGMENTATION)


def test_estimate_peak_uses_per_dtype_alpha():
    """End-to-end pin: a HardwareProfile with bpe=0.5 makes
    ``estimate_peak`` return the raw peak scaled by 0.75 (the 4-bit
    α) instead of 1.10. With the default bpe=2.0 the existing 1.10
    ceiling is preserved — matching every legacy test.
    """
    from axolotl.integrations.protrain.cost.memory import estimate_peak
    from axolotl.integrations.protrain.types import (
        BlockId,
        BlockMode,
        BlockStrategyMap,
        ChunkLayout,
        CostConfig,
        HardwareProfile,
        ProfilerTrace,
    )

    # Minimal viable trace + layout — one block, one tiny op. No
    # measured per-block peaks, no measured deltas, so the op-walk
    # raw peak is dominated by ``model_state_present`` (which is 0
    # because ``model_state_bytes`` is 0) plus the persistent /
    # buffer pool terms.
    # We arrange S_chunk * (n_persist + n_buffer) = 1 GiB so the raw
    # peak is large and easy to multiply against α.
    s_chunk = 1 << 28  # 256 MiB
    n_chunk = 4
    layout = ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple(tuple() for _ in range(n_chunk)),  # type: ignore[arg-type]
        param_to_chunk={},
        block_to_chunks={BlockId(0): ()},
    )
    trace = ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes={BlockId(0): 0},
        model_state_bytes=0,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="test",
        bs=1,
        seq=16,
        sku="test",
        world=1,
    )
    cfg = CostConfig(n_persist=2, n_buffer=2, n_swap=0, n_checkpoint=0)
    block_map: BlockStrategyMap = {BlockId(0): BlockMode.NONE}

    # Default HW profile — bpe=2.0 lands on α=1.10.
    hw_fp16 = HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )
    # 4-bit HW profile — bpe=0.5 lands on α=0.75.
    hw_4bit = HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        dominant_param_bytes_per_element=0.5,
    )

    peak_fp16 = estimate_peak(cfg, trace, layout, block_map, hw_fp16)
    peak_4bit = estimate_peak(cfg, trace, layout, block_map, hw_4bit)

    # The α=0.75 branch must return strictly less peak than the
    # α=1.10 branch on the same raw inputs — concrete value depends
    # on the op-walk's exact accounting, so assert the relative
    # contract.
    assert peak_4bit < peak_fp16, (
        f"per-dtype α should yield smaller peak for 4-bit "
        f"(α=0.75): got peak_4bit={peak_4bit}, peak_fp16={peak_fp16}"
    )
    # Ratio is 0.75 / 1.10 modulo int() rounding (cost model
    # casts the alpha-scaled value to int). Use 1% slack.
    expected_ratio = ALPHA_FRAGMENTATION_4BIT / ALPHA_FRAGMENTATION
    observed_ratio = peak_4bit / max(peak_fp16, 1)
    assert observed_ratio == pytest.approx(expected_ratio, rel=0.01), (
        f"peak_4bit / peak_fp16 = {observed_ratio:.4f} should match "
        f"α_4bit / α_fp16 = {expected_ratio:.4f}"
    )
