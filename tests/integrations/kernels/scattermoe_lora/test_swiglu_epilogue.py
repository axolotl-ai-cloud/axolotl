"""moe_swiglu_epilogue: fused LoRA-B + activation epilogue vs the unfused two-pass path."""

import pytest
import torch

DEV = "cuda"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_blockfp8(E, N, K, block=128):
    W = torch.randn(E, N, K, device=DEV, dtype=torch.float32) * 0.02
    wb = W.reshape(E, N // block, block, K // block, block)
    amax = wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12)
    ws = (amax / 448.0).squeeze(2).squeeze(-1)
    wq = (wb / amax * 448.0).reshape(E, N, K).to(torch.float8_e4m3fn)
    return wq, ws


@pytest.mark.parametrize("act_type,limit", [("silu", 7.0), ("gelu_tanh", None)])
def test_fused_matches_unfused_fp8_path(act_type, limit, monkeypatch):
    from axolotl.integrations.kernels.libs.scattermoe_lora.fp8_grouped import (
        grouped_fp8_moe_train,
    )

    torch.manual_seed(0)
    E, H, I, N, topk, r = 4, 256, 256, 64, 2, 24  # noqa: E741  (r non-pow2 on purpose)
    wq_gu, ws_gu = _make_blockfp8(E, 2 * I, H)
    wq_dn, ws_dn = _make_blockfp8(E, H, I)
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.randint(0, E, (N, topk), device=DEV)
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1)

    def run(fused):
        monkeypatch.setenv("AXOLOTL_SWIGLU_EPILOGUE", "1" if fused else "0")
        torch.manual_seed(1)
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
            limit,
            act_type=act_type,
            dx_cache={},
        )
        out.float().pow(2).mean().backward()
        return out.detach(), hk.grad, Ag.grad, Bg.grad, Ad.grad, Bd.grad

    ref = run(False)
    got = run(True)
    names = ["out", "d_hidden", "dAgu", "dBgu", "dAdn", "dBdn"]
    for name, a, b in zip(names, got, ref, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2, msg=f"{name} mismatch")


def test_config_wiring(monkeypatch):
    from axolotl.integrations.kernels.args import KernelsArgs
    from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import (
        RUNTIME,
        configure_scattermoe_runtime,
    )

    assert KernelsArgs.model_validate({"moe_swiglu_epilogue": True}).moe_swiglu_epilogue
    assert KernelsArgs.model_validate({}).moe_swiglu_epilogue is None

    class _C(dict):
        get = dict.get

    monkeypatch.delenv("AXOLOTL_SWIGLU_EPILOGUE", raising=False)
    configure_scattermoe_runtime(_C({"moe_swiglu_epilogue": True}))
    assert RUNTIME.swiglu_epilogue is True
    configure_scattermoe_runtime(_C({}))
    assert RUNTIME.swiglu_epilogue is False
    monkeypatch.setenv("AXOLOTL_SWIGLU_EPILOGUE", "1")
    configure_scattermoe_runtime(_C({}))
    assert RUNTIME.swiglu_epilogue is True
    RUNTIME.reset()
