# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU tests for the sonicmoe fp4_cute host-prep layer.

Covers the NVFP4 quantize/dequantize reference, the blocked scale-factor
layout (checked against an independent per-element formula), the varlen
dQaccum SFA padding, per-tensor-scale folding, and the gate/up interleave.
The kernel itself is GPU-only and validated by benchmarks/sonicmoe_nvfp4/.
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
    FP4_CODE_VALUES,
    SF_VEC_SIZE,
    dequantize_nvfp4_ref,
    encode_e2m1,
    fp4_code_to_value,
    pack_fp4_codes,
    quantize_nvfp4_ref,
    simulate_nvfp4_quant,
    unpack_fp4_codes,
)
from axolotl.integrations.kernels.libs.sonicmoe.sf_layout import (
    build_varlen_sfa,
    fold_per_tensor_scale,
    gate_up_interleave_perm,
    pack_scales_blocked,
    varlen_padded_num_row_tiles,
)


def _naive_blocked_index(m: int, k: int):
    """Independent formula for the 512-byte swizzled SF atom position."""
    tile_m, tile_k = m // 128, k // 4
    byte = (m % 32) * 16 + ((m // 32) % 4) * 4 + (k % 4)
    return tile_m, tile_k, byte


def test_fp4_code_value_roundtrip():
    # -0.0 (code 8) encodes back to +0 (code 0); every other code roundtrips.
    codes = torch.tensor([c for c in range(16) if c != 8], dtype=torch.uint8)
    vals = fp4_code_to_value(codes)
    assert torch.equal(encode_e2m1(vals), codes)
    assert FP4_CODE_VALUES[3] == 1.5 and FP4_CODE_VALUES[11] == -1.5


def test_pack_unpack_roundtrip():
    codes = torch.randint(0, 16, (5, 32), dtype=torch.uint8)
    assert torch.equal(unpack_fp4_codes(pack_fp4_codes(codes)), codes)
    # low nibble first: byte 0 holds elements 0 (low) and 1 (high)
    two = torch.tensor([[3, 9]], dtype=torch.uint8)
    assert int(pack_fp4_codes(two)[0, 0]) == 3 | (9 << 4)


def test_quantize_exact_grid_roundtrips():
    # Values on the E2M1 grid times an exactly-representable e4m3 scale. The
    # stored scale is block_amax/6, so pin each 16-block's amax to code 6.0
    # to make the scale land exactly on 2.0.
    scale = 2.0
    codes = torch.randint(0, 16, (4, 32), dtype=torch.uint8)
    x = fp4_code_to_value(codes) * scale
    x[..., 0::SF_VEC_SIZE] = 6.0 * scale
    assert torch.equal(simulate_nvfp4_quant(x), x)


def test_quantize_error_bound_and_zeros():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 0.5
    x[3] = 0.0
    packed, scale, pts = quantize_nvfp4_ref(x)
    sim = dequantize_nvfp4_ref(packed, scale, pts)
    # Max quantization error is half the largest code gap (2.0) times the scale.
    bound = scale.float().repeat_interleave(SF_VEC_SIZE, dim=-1) * 1.0 + 1e-6
    assert bool(((x - sim).abs() <= bound).all())
    assert torch.equal(sim[3], torch.zeros_like(sim[3]))
    assert not sim.isnan().any()


def test_quantize_applies_per_tensor_scale():
    torch.manual_seed(1)
    x = torch.randn(4, 32)
    packed, scale, pts = quantize_nvfp4_ref(x, per_tensor_scale=0.25)
    assert float(pts) == 0.25
    sim = dequantize_nvfp4_ref(packed, scale, pts)
    bound = scale.float().repeat_interleave(SF_VEC_SIZE, dim=-1) * float(pts) + 1e-6
    assert bool(((x - sim).abs() <= bound).all())


def test_pack_scales_blocked_matches_naive_formula():
    torch.manual_seed(2)
    l, mn, sf_k = 2, 130, 6  # exercises both pad directions
    raw = torch.randint(0, 256, (l, mn, sf_k), dtype=torch.uint8)
    blocked = pack_scales_blocked(raw)
    assert blocked.shape == (l, 2, 2, 512)
    for li in range(l):
        for m in range(mn):
            for k in range(sf_k):
                tm, tk, byte = _naive_blocked_index(m, k)
                assert int(blocked[li, tm, tk, byte]) == int(raw[li, m, k])
    # padding is zero
    tm, tk, byte = _naive_blocked_index(131, 0)
    assert int(blocked[0, tm, tk, byte]) == 0


def test_pack_scales_blocked_preserves_dtype():
    raw = torch.randint(0, 256, (1, 128, 4), dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    out = pack_scales_blocked(raw)
    assert out.dtype == torch.float8_e4m3fn and out.shape == (1, 1, 1, 512)


def test_build_varlen_sfa_offsets():
    torch.manual_seed(3)
    seqlens = [5, 0, 130, 3]
    total_m = sum(seqlens)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()))
    sf_k = 4
    rows = torch.randint(1, 255, (total_m, sf_k), dtype=torch.uint8)
    blocked = build_varlen_sfa(rows, cu)

    e = len(seqlens)
    assert varlen_padded_num_row_tiles(total_m, e) == (total_m + 127) // 128 + e - 1
    assert blocked.shape == (1, varlen_padded_num_row_tiles(total_m, e), 1, 512)

    for i in range(e):
        start = int(cu[i])
        row0 = (start // 128 + i) * 128
        for r in range(seqlens[i]):
            padded_row = row0 + r
            for k in range(sf_k):
                tm, tk, byte = _naive_blocked_index(padded_row, k)
                assert int(blocked[0, tm, tk, byte]) == int(rows[start + r, k])


def test_build_varlen_sfa_rejects_bad_cu():
    rows = torch.zeros(10, 4, dtype=torch.uint8)
    with pytest.raises(AssertionError):
        build_varlen_sfa(rows, torch.tensor([0, 4, 9]))  # cu[-1] != total_m


def test_fold_per_tensor_scale():
    scale = torch.full((2, 4, 4), 2.0).to(torch.float8_e4m3fn)
    pts = torch.tensor([0.5, 2.0])
    folded, rel_err = fold_per_tensor_scale(scale, pts)
    assert rel_err == 0.0
    assert torch.equal(folded[0].float(), torch.full((4, 4), 1.0))
    assert torch.equal(folded[1].float(), torch.full((4, 4), 4.0))
    # accepts (E,1,1) shape and scalar
    folded2, _ = fold_per_tensor_scale(scale, pts.view(2, 1, 1))
    assert torch.equal(folded2.float(), folded.float())
    folded3, _ = fold_per_tensor_scale(scale, torch.tensor(1.0))
    assert torch.equal(folded3.float(), scale.float())


def test_fold_per_tensor_scale_saturation_and_underflow():
    big = torch.full((1, 2, 2), 448.0).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="saturates"):
        fold_per_tensor_scale(big, torch.tensor([2.0]))
    tiny = torch.full((1, 2, 2), 2.0**-9).to(torch.float8_e4m3fn)  # e4m3 min subnormal
    with pytest.raises(ValueError, match="underflows"):
        fold_per_tensor_scale(tiny, torch.tensor([0.25]))


def test_gate_up_interleave_perm():
    perm = gate_up_interleave_perm(8)
    assert perm.tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    w = torch.arange(8).unsqueeze(-1)  # concat: gate rows 0-3, up rows 4-7
    wi = w[perm]
    # pair j = (gate_j, up_j)
    assert wi[0::2].squeeze(-1).tolist() == [0, 1, 2, 3]
    assert wi[1::2].squeeze(-1).tolist() == [4, 5, 6, 7]
    # invertible
    inv = torch.argsort(perm)
    assert torch.equal(wi[inv], w)


def test_fp4_cute_unavailable_without_sm100():
    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute import fp4_cute_available

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11):
        pytest.skip("running on SM100/SM110")
    assert fp4_cute_available() is False


def _make_nvfp4_weight(E, N, K, pts=None, seed=0):
    """Random dense [E, N, K] quantized per expert into a torchao NVFP4Tensor."""
    nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    torch.manual_seed(seed)
    pts_t = torch.ones(E) if pts is None else torch.as_tensor(pts, dtype=torch.float32)
    qs, ss = [], []
    for e in range(E):
        q, s, _ = quantize_nvfp4_ref(torch.randn(N, K) * K**-0.5, pts_t[e])
        qs.append(q)
        ss.append(s)
    return nvfp4_mod.NVFP4Tensor(
        torch.stack(qs),
        torch.stack(ss),
        16,
        torch.float32,
        per_tensor_scale=pts_t.view(-1, 1, 1),
    )


def test_torchao_dequant_matches_ref():
    # The kernel path assumes torchao's scheme (low nibble first, value =
    # code * e4m3_scale * pts) is identical to our reference.
    w = _make_nvfp4_weight(3, 8, 32, pts=[0.5, 1.3, 2.0])
    ref = dequantize_nvfp4_ref(w.qdata, w.scale, w.per_tensor_scale)
    assert torch.equal(w.dequantize(torch.float32), ref)


def test_dequantize_expert_slice():
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import (
        dequantize_expert_slice,
    )

    w = _make_nvfp4_weight(3, 8, 32, pts=[0.5, 1.0, 2.0])
    full = w.dequantize(torch.float32)
    for e in range(3):
        assert torch.equal(dequantize_expert_slice(w, e), full[e])

    dense = torch.randn(3, 8, 32)
    assert dequantize_expert_slice(dense, 1) is not None
    assert torch.equal(dequantize_expert_slice(dense, 1), dense[1])


def test_unpack_nvfp4_components():
    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute_ops import (
        unpack_nvfp4_components,
    )

    w = _make_nvfp4_weight(2, 8, 32, pts=[1.0, 0.5])
    qdata, scale, pts = unpack_nvfp4_components(w)
    assert qdata.dtype == torch.uint8 and qdata.shape == (2, 8, 16)
    assert scale.dtype == torch.float8_e4m3fn and scale.shape == (2, 8, 2)
    assert pts.shape == (2, 1, 1)
    with pytest.raises(TypeError):
        unpack_nvfp4_components(torch.randn(2, 8, 32))


def test_fp4_cute_dims_ok():
    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute_ops import (
        fp4_cute_dims_ok,
    )

    w1 = torch.empty(2, 16, 64)  # [E, 2I, H]
    w2 = torch.empty(2, 64, 32)  # [E, H, I]
    assert fp4_cute_dims_ok(w1, w2) is True
    assert fp4_cute_dims_ok(torch.empty(2, 16, 48), w2) is False  # K % 32
    assert fp4_cute_dims_ok(torch.empty(2, 12, 64), w2) is False  # N % 8
    assert fp4_cute_dims_ok(w1, torch.empty(2, 64, 24)) is False


def test_lora_backward_nvfp4_base_matches_dense():
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        _lora_backward_per_group,
    )

    torch.manual_seed(4)
    E, d1, d2, r = 3, 8, 32, 2
    w = _make_nvfp4_weight(E, d1, d2, pts=[0.5, 1.0, 2.0])
    counts = [4, 0, 3]
    offsets = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()))
    T = sum(counts)
    x = torch.randn(T, d2)
    grad_h = torch.randn(T, d1)
    lora_A = torch.randn(r * E, d2)
    lora_B = torch.randn(d1, r * E)

    got = _lora_backward_per_group(grad_h, x, offsets, lora_A, lora_B, w, 0.5)
    ref = _lora_backward_per_group(
        grad_h, x, offsets, lora_A, lora_B, w.dequantize(torch.float32), 0.5
    )
    for g, rf in zip(got, ref, strict=True):
        torch.testing.assert_close(g, rf)


def test_select_nvfp4_backend(monkeypatch):
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        _select_nvfp4_backend,
    )

    dense1, dense2 = torch.empty(2, 16, 64), torch.empty(2, 64, 32)
    monkeypatch.delenv("AXOLOTL_SONICMOE_NVFP4_BACKEND", raising=False)
    assert _select_nvfp4_backend(dense1, dense2) == "dequant"
    monkeypatch.setenv("AXOLOTL_SONICMOE_NVFP4_BACKEND", "fp4_cute")
    assert _select_nvfp4_backend(dense1, dense2) == "fp4_cute"
