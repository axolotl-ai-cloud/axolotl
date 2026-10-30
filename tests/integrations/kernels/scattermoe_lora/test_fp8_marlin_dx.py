"""Block-fp8 transposed-Marlin backward dX (W8A16, exact transpose): correctness + mode wiring.

128x128 block-fp8 scales are transpose-symmetric, so the dX pack is a byte-permute + scale-grid
transpose with ZERO requant error; dX runs as a marlin W8A16 grouped GEMM with bf16 group-128
scales (exponent bias 2^120 fused into the scales, per vLLM fp8 marlin).
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import RUNTIME

DEV = "cuda"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _marlin_ok():
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import (
        marlin_w4a16_available,
    )

    return marlin_w4a16_available()


def _make_blockfp8(E, N, K, block=128):
    W = torch.randn(E, N, K, device=DEV, dtype=torch.float32) * 0.02
    wb = W.reshape(E, N // block, block, K // block, block)
    amax = wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12)
    ws = (amax / 448.0).squeeze(2).squeeze(-1)
    wq = (wb / amax * 448.0).reshape(E, N, K).to(torch.float8_e4m3fn)
    return wq, ws


def _dequant(wq, ws, block=128):
    E, N, K = wq.shape
    return (
        wq.float().reshape(E, N // block, block, K // block, block)
        * ws[:, :, None, :, None]
    ).reshape(E, N, K)


def test_scalar_type_id_packing():
    """The hardcoded vllm ScalarType ids must match the C++ bit packing."""

    def pack(exponent, mantissa, signed, bias, finite, nan):
        fields = [
            (exponent, 8),
            (mantissa, 8),
            (int(signed), 1),
            (bias & 0xFFFFFFFF, 32),
            (int(finite), 1),
            (nan, 8),
        ]
        id_, off = 0, 0
        for val, bits in fields:
            id_ |= (val & ((1 << bits) - 1)) << off
            off += bits
        return id_

    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.prep import (
        FLOAT4_E2M1F_ID,
        FLOAT8_E4M3FN_ID,
    )

    assert pack(2, 1, True, 0, True, 0) == FLOAT4_E2M1F_ID
    assert pack(4, 3, True, 0, True, 2) == FLOAT8_E4M3FN_ID


def test_fp8_marlin_dx_matches_exact_reference(monkeypatch):
    """Transposed-pack fp8 marlin dX == exact dequant contraction within bf16-scale rounding."""
    if not _marlin_ok():
        pytest.skip("marlin backend unavailable")
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import load_ext
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.backend import (
        marlin_dx_gemm,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.prep import (
        FLOAT8_E4M3FN_ID,
        prepare_blockfp8_dx_weight,
    )

    monkeypatch.setattr(RUNTIME, "marlin_dx", True)
    torch.manual_seed(0)
    E, N, K, rows = 4, 256, 128, 64
    wq, ws = _make_blockfp8(E, N, K)
    Wd = _dequant(wq, ws)

    ext = load_ext()
    pack = prepare_blockfp8_dx_weight(wq, ws, ext.gptq_marlin_repack)

    Mt = E * rows
    g = torch.randn(Mt, N, device=DEV, dtype=torch.bfloat16) * 0.1
    m_indices = torch.arange(E, device=DEV, dtype=torch.int32).repeat_interleave(
        rows // 64
    )
    out = marlin_dx_gemm(pack, g, m_indices, K, N, FLOAT8_E4M3FN_ID)
    ref = torch.cat([g[e * rows : (e + 1) * rows].float() @ Wd[e] for e in range(E)])
    relerr = (out.float() - ref).norm() / ref.norm()
    assert relerr < 0.01, f"fp8 marlin dX rel {relerr:.5f}"


def test_fp8_grouped_train_grads_with_dx_packs(monkeypatch):
    """grouped_fp8_moe_train grads with the marlin dX packs match the Triton-dX path (the fp8
    transpose is exact — only bf16 scale rounding differs)."""
    if not _marlin_ok():
        pytest.skip("marlin backend unavailable")
    from axolotl.integrations.kernels.libs.scattermoe_lora.fp8_grouped import (
        grouped_fp8_moe_train,
    )

    torch.manual_seed(1)
    E, H, I, N, topk, r = 4, 256, 128, 32, 2, 8  # noqa: E741
    wq_gu, ws_gu = _make_blockfp8(E, 2 * I, H)
    wq_dn, ws_dn = _make_blockfp8(E, H, I)
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.randint(0, E, (N, topk), device=DEV)
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1)

    def run(dx_mode):
        monkeypatch.setattr(RUNTIME, "marlin_dx", dx_mode == "1")
        torch.manual_seed(2)
        Ag = (
            torch.randn(r * E, H, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        Bg = (
            torch.randn(2 * I, r * E, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        Ad = (
            torch.randn(r * E, I, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        Bd = (
            torch.randn(H, r * E, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        hk = hidden.clone().requires_grad_(True)
        cache = {}
        out = grouped_fp8_moe_train(
            hk,
            idx,
            wts,
            wq_gu,
            ws_gu,
            wq_dn,
            ws_dn,
            (Ag, Bg, 2.0),
            (Ad, Bd, 2.0),
            None,
            act_type="silu",
            dx_cache=cache,
        )
        out.float().pow(2).mean().backward()
        built = "fp8_gate_up_dxT" in cache
        return built, hk.grad.clone(), Ag.grad.clone()

    built_on, dh_on, dag_on = run("1")
    assert built_on, "fp8 dX packs not built with moe_marlin_dx on"
    built_off, dh_off, dag_off = run("0")
    assert not built_off

    cos = torch.nn.functional.cosine_similarity
    c_h = cos(dh_on.float().flatten(), dh_off.float().flatten(), dim=0).item()
    c_a = cos(dag_on.float().flatten(), dag_off.float().flatten(), dim=0).item()
    assert c_h > 0.999, f"d_hidden cos {c_h:.5f}"
    assert c_a > 0.999, f"d_Agu cos {c_a:.5f}"


def test_batched_repack_bit_exact_vs_loop():
    """The one-launch batched transpose+repack must match the per-expert
    gptq_marlin_repack loop bit-for-bit (qdata) and the per-expert scale permute exactly."""
    if not _marlin_ok():
        pytest.skip("marlin backend unavailable")
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import load_ext
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.batched_repack import (
        batched_repack8_transposed,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.prep import (
        FP8_BLOCK,
        marlin_permute_scales,
        prepare_blockfp8_dx_weight,
    )

    torch.manual_seed(3)
    E, N, K = 3, 384, 256
    wq, ws = _make_blockfp8(E, N, K)
    ext = load_ext()
    perm = torch.empty(0, dtype=torch.int, device=DEV)

    # reference: the original per-expert loop
    ref_qw, ref_sc = [], []
    for e in range(E):
        bT = wq[e].t().contiguous()
        qw_i = bT.view(torch.int32).T.contiguous()
        ref_qw.append(ext.gptq_marlin_repack(qw_i, perm, N, K, 8, False))
        s = (ws[e].float() * 2.0**120).to(torch.bfloat16)
        s = s.repeat_interleave(FP8_BLOCK, dim=1)
        ref_sc.append(
            marlin_permute_scales(s, size_k=N, size_n=K, group_size=FP8_BLOCK)
        )
    ref_qw = torch.stack(ref_qw, 0)
    ref_sc = torch.stack(ref_sc, 0)

    got_qw = batched_repack8_transposed(wq, ext)
    assert got_qw.shape == ref_qw.shape, (got_qw.shape, ref_qw.shape)
    assert torch.equal(got_qw, ref_qw), "batched repack differs from per-expert loop"

    got_qw2, got_sc, gs = prepare_blockfp8_dx_weight(wq, ws, ext.gptq_marlin_repack)
    assert gs is None
    assert torch.equal(got_qw2, ref_qw)
    assert torch.equal(got_sc.view(torch.uint8), ref_sc.view(torch.uint8))


def test_dx_pack_build_failure_is_cached(monkeypatch):
    """A failed pack build must not retry every step: the cache records the failure."""
    if not _marlin_ok():
        pytest.skip("marlin backend unavailable")
    import axolotl.integrations.kernels.libs.scattermoe_lora.fp8_grouped as fg

    monkeypatch.setattr(RUNTIME, "marlin_dx", True)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(
        "axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.prep.prepare_blockfp8_dx_weight",
        boom,
    )
    wq, ws = _make_blockfp8(2, 256, 256)
    cache = {}
    assert fg._build_fp8_dx_packs((wq, ws), (wq, ws), cache) == (None, None)
    assert cache.get("fp8_dxT_failed")
    assert fg._build_fp8_dx_packs((wq, ws), (wq, ws), cache) == (None, None)
    assert calls["n"] == 1


def test_fp8_marlin_forward_matches_triton(monkeypatch):
    """moe_grouped_backend: marlin routes the fp8 forward through the W8A16 marlin packs;
    output must match the Triton forward within bf16-scale rounding, grads unaffected."""
    if not _marlin_ok():
        pytest.skip("marlin backend unavailable")
    from axolotl.integrations.kernels.libs.scattermoe_lora import fp8_grouped as fg

    monkeypatch.setattr(RUNTIME, "marlin_dx", False)
    torch.manual_seed(4)
    # H/I >= 256 so both marlin reduction dims span >=2 scale groups (single-group -> skipped)
    E, H, I, N, topk, r = 4, 256, 256, 32, 2, 8  # noqa: E741
    wq_gu, ws_gu = _make_blockfp8(E, 2 * I, H)
    wq_dn, ws_dn = _make_blockfp8(E, H, I)
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.randint(0, E, (N, topk), device=DEV)
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1)

    def run(backend):
        monkeypatch.setattr(RUNTIME, "grouped_backend", backend)
        fg._BACKEND_LOGGED = True
        torch.manual_seed(5)
        Ag = (
            torch.randn(r * E, H, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        Bg = (
            torch.randn(2 * I, r * E, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        Ad = (
            torch.randn(r * E, I, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        Bd = (
            torch.randn(H, r * E, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        hk = hidden.clone().requires_grad_(True)
        cache = {}
        out = fg.grouped_fp8_moe_train(
            hk,
            idx,
            wts,
            wq_gu,
            ws_gu,
            wq_dn,
            ws_dn,
            (Ag, Bg, 2.0),
            (Ad, Bd, 2.0),
            None,
            act_type="silu",
            dx_cache=cache,
        )
        out.float().pow(2).mean().backward()
        return out.detach(), hk.grad.clone(), Ag.grad.clone(), cache

    # default: transient per-forward packs (memory-neutral) -> nothing cached
    out_m, dh_m, dag_m, cache_m = run("marlin")
    assert "fp8_gate_up_fwd" not in cache_m, "fwd packs cached without headroom opt-in"
    # headroom opt-in -> resident cache
    monkeypatch.setenv("AXOLOTL_MARLIN_CACHE_MIN_FREE_GB", "0")
    out_c, _, _, cache_c = run("marlin")
    assert "fp8_gate_up_fwd" in cache_c, "fwd packs not cached with headroom opt-in"
    monkeypatch.delenv("AXOLOTL_MARLIN_CACHE_MIN_FREE_GB")
    torch.testing.assert_close(out_c, out_m)
    out_t, dh_t, dag_t, cache_t = run(None)
    assert "fp8_gate_up_fwd" not in cache_t

    rel = (out_m.float() - out_t.float()).norm() / out_t.float().norm()
    assert rel < 0.01, f"marlin fwd rel {rel:.5f}"
    cos = torch.nn.functional.cosine_similarity
    assert cos(dh_m.float().flatten(), dh_t.float().flatten(), dim=0) > 0.999
    assert cos(dag_m.float().flatten(), dag_t.float().flatten(), dim=0) > 0.999


def test_batched_repack8_fwd_bit_exact():
    """The coalesced forward-orientation repack == the strided transposed-view path byte-for-byte
    (which is itself loop-validated above)."""
    if not _marlin_ok():
        pytest.skip("marlin backend unavailable")
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import load_ext
    from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16.batched_repack import (
        batched_repack8_fwd,
        batched_repack8_transposed,
    )

    torch.manual_seed(6)
    ext = load_ext()
    for E, N, K in ((4, 512, 256), (3, 384, 768)):
        wq = torch.randint(0, 255, (E, N, K), device=DEV, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        a = batched_repack8_fwd(wq, ext)
        b = batched_repack8_transposed(wq.transpose(1, 2), ext)
        assert a.shape == (E, K // 16, N * 4)
        assert torch.equal(a, b), f"fwd repack differs at {(E, N, K)}"


def test_fp8_marlin_forward_chunked_matches_unchunked(monkeypatch):
    """Chunked transient fwd packs == unchunked within marlin tile-scheduling rounding."""
    if not _marlin_ok():
        pytest.skip("marlin backend unavailable")
    from axolotl.integrations.kernels.libs.scattermoe_lora import fp8_grouped as fg

    monkeypatch.setattr(RUNTIME, "grouped_backend", "marlin")
    monkeypatch.setattr(fg, "_BACKEND_LOGGED", True)
    torch.manual_seed(7)
    E, H, I, N, topk, r = 4, 512, 384, 64, 2, 16  # noqa: E741
    wq_gu, ws_gu = _make_blockfp8(E, 2 * I, H)
    wq_dn, ws_dn = _make_blockfp8(E, H, I)
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.randint(0, E, (N, topk), device=DEV)
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1)
    g = torch.Generator(device=DEV).manual_seed(8)
    mk = lambda a, b: (
        torch.randn(a, b, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
    )  # noqa: E731
    args = (
        hidden,
        idx,
        wts,
        wq_gu,
        ws_gu,
        wq_dn,
        ws_dn,
        (mk(r * E, H), mk(2 * I, r * E), 2.0),
        (mk(r * E, I), mk(H, r * E), 2.0),
        None,
    )
    outs = {}
    for c in ("1", "3"):
        monkeypatch.setenv("AXOLOTL_MARLIN_FWD_CHUNKS", c)
        outs[c] = fg.grouped_fp8_moe_train(*args, act_type="silu", dx_cache={})
    rel = (outs["3"].float() - outs["1"].float()).norm() / outs["1"].float().norm()
    assert rel < 1e-3, f"chunked fwd rel {rel:.2e}"
