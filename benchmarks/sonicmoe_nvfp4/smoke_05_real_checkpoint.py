"""Smoke 5: real-checkpoint numerics on one MoE layer of nvidia/Qwen3-30B-A3B-NVFP4.

Pulls one layer's 128 routed experts straight from the checkpoint with the same helpers the
qwen3_moe adapter's load path uses (``inspect_nvfp4_layout`` + ``_build_expert_nvfp4``, i.e.
the shared ``fuse_nvfp4_experts`` core), then reruns the smoke-4 end-to-end MoE-LoRA
comparison on the REAL weights: ``backend="fp4_cute"`` vs the STE oracle (tight: forward
bitwise, grads by norm) and vs the ``dequant`` backend (info: the W4A4 activation-quant
error at real weight magnitudes, the honest OQ1 number vs smoke 4's random-weight 17%).

Also prints the fraction of e4m3 block scales that would land in e4m3's subnormal range if
the per-expert ``per_tensor_scale`` were folded in, documenting on a real checkpoint why
fp4_cute_ops applies pts post-GEMM instead.

Env: AXOLOTL_SMOKE05_REPO (default nvidia/Qwen3-30B-A3B-NVFP4), AXOLOTL_SMOKE05_LAYER
(default 24). First run downloads the shard(s) holding that layer's experts.
"""

import json
import os

from _common import check, finish, report, report_norm, require_sm100

REPO = os.environ.get("AXOLOTL_SMOKE05_REPO", "nvidia/Qwen3-30B-A3B-NVFP4")
LAYER = int(os.environ.get("AXOLOTL_SMOKE05_LAYER", "24"))


def main():
    import torch

    require_sm100()

    from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
        _build_expert_nvfp4,
        _load_index,
        _resolve_repo_file,
        inspect_nvfp4_layout,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute_ops import (
        fp4_cute_dims_ok,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import gated_activation
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        _lora_delta_per_group,
        combine_expert_outputs,
        grouped_moe_reference_forward,
        route_and_group,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        dequantize_nvfp4_ref,
        quantize_nvfp4_ref,
    )

    with open(_resolve_repo_file(REPO, "config.json")) as f:
        hf_cfg = json.load(f)
    E = hf_cfg["num_experts"]
    top_k = hf_cfg["num_experts_per_tok"]
    H = hf_cfg["hidden_size"]
    I = hf_cfg["moe_intermediate_size"]  # noqa: E741
    assert LAYER < hf_cfg["num_hidden_layers"]
    print(f"{REPO} layer {LAYER}: E={E} top_k={top_k} H={H} I={I}")

    layout = inspect_nvfp4_layout(REPO)
    check("layout: modelopt naming", layout["naming"] == "modelopt")
    check(
        "layout: routed gate/up/down present",
        layout["routed_present"]
        and set(layout["routed_projs"]) >= {"gate_proj", "up_proj", "down_proj"},
    )

    dev = "cuda"
    dtype = torch.bfloat16
    wmap = _load_index(REPO)
    base_fmt = "model.layers.{layer}.mlp.experts.{e}.{proj}"
    w1 = _build_expert_nvfp4(
        REPO, wmap, base_fmt, LAYER, ("gate_proj", "up_proj"), E, dev
    )
    w2 = _build_expert_nvfp4(REPO, wmap, base_fmt, LAYER, ("down_proj",), E, dev)
    check(
        "fused shapes",
        tuple(w1.shape) == (E, 2 * I, H) and tuple(w2.shape) == (E, H, I),
    )
    check("dims ok for fp4_cute", fp4_cute_dims_ok(w1, w2))

    for name, w in (("gate_up", w1), ("down", w2)):
        pts = w.per_tensor_scale.view(-1)
        folded = w.scale.float() * w.per_tensor_scale
        nz = folded > 0
        sub = float(
            ((folded < 2**-6) & nz).float().sum() / nz.float().sum().clamp(min=1)
        )
        print(
            f"[info] {name}: pts range [{pts.min():.3e}, {pts.max():.3e}]; "
            f"{sub:.1%} of nonzero block scales would be e4m3-SUBNORMAL if pts were folded"
        )

    torch.manual_seed(5)
    T, r = 2048, 8
    scaling1, scaling2 = 0.5, 0.25
    A1 = torch.randn(r * E, H, device=dev, dtype=dtype) * H**-0.5
    B1 = torch.randn(2 * I, r * E, device=dev, dtype=dtype) * 0.02
    A2 = torch.randn(r * E, I, device=dev, dtype=dtype) * I**-0.5
    B2 = torch.randn(H, r * E, device=dev, dtype=dtype) * 0.02
    # post-RMSNorm hidden states are ~unit RMS; N(0,1) is the cheap stand-in
    hidden = torch.randn(T, H, device=dev, dtype=dtype)
    router = torch.randn(T, E, device=dev)
    vals, top_k_index = router.topk(top_k, dim=-1)
    top_k_weights = torch.softmax(vals, dim=-1)
    g_out = torch.randn(T, H, device=dev, dtype=dtype)

    def leaves():
        return tuple(t.clone().requires_grad_() for t in (hidden, A1, B1, A2, B2))

    # --- implementation: fp4_cute backend ---
    h_i, A1_i, B1_i, A2_i, B2_i = leaves()
    out = grouped_moe_reference_forward(
        h_i,
        top_k_index,
        top_k_weights,
        w1,
        None,
        w2,
        None,
        (A1_i, B1_i),
        (A2_i, B2_i),
        E,
        act="silu",
        backend="fp4_cute",
        concat=True,
        scaling1=scaling1,
        scaling2=scaling2,
    )
    (out.float() * g_out.float()).sum().backward()
    torch.cuda.synchronize()

    # --- STE oracle (same construction as smoke 4, real weights) ---
    def ste_quant(x):
        q, s, _ = quantize_nvfp4_ref(x.detach())
        dq = dequantize_nvfp4_ref(q, s).to(x.dtype)
        return x + (dq - x).detach()

    def base_gemm(x, w, offsets):
        # kernel view of the weights: stored scales, NO pts; then the same
        # post-GEMM per-expert row scaling in fp32 that fp4_cute_ops applies
        w_np = dequantize_nvfp4_ref(w.qdata, w.scale)
        pts = w.per_tensor_scale.view(-1)
        xq = ste_quant(x)
        outs = []
        for e in range(E):
            s0, e0 = int(offsets[e]), int(offsets[e + 1])
            o = (xq[s0:e0].float() @ w_np[e].t()).to(dtype)
            outs.append((o.float() * pts[e]).to(dtype))
        return torch.cat(outs, dim=0)

    h_o, A1_o, B1_o, A2_o, B2_o = leaves()
    xg, offsets, gidx, wg = route_and_group(h_o, top_k_index, top_k_weights, E)
    h = base_gemm(xg, w1, offsets) + _lora_delta_per_group(
        xg, offsets, A1_o, B1_o, scaling1, E, 2 * I, H
    )
    a = gated_activation(h, "silu", concat=True)
    y = base_gemm(a, w2, offsets) + _lora_delta_per_group(
        a, offsets, A2_o, B2_o, scaling2, E, H, I
    )
    out_ref = combine_expert_outputs(y, gidx, wg, T)
    (out_ref.float() * g_out.float()).sum().backward()

    report("real ckpt fp4_cute: forward", out, out_ref)
    report_norm("real ckpt fp4_cute: d hidden", h_i.grad, h_o.grad)
    report_norm("real ckpt fp4_cute: d lora_A1", A1_i.grad, A1_o.grad)
    report_norm("real ckpt fp4_cute: d lora_B1", B1_i.grad, B1_o.grad)
    report_norm("real ckpt fp4_cute: d lora_A2", A2_i.grad, A2_o.grad)
    report_norm("real ckpt fp4_cute: d lora_B2", B2_i.grad, B2_o.grad)

    # --- dequant backend, info only: W4A4 activation-quant error on REAL weights (OQ1) ---
    h_d, A1_d, B1_d, A2_d, B2_d = leaves()
    out_dq = grouped_moe_reference_forward(
        h_d,
        top_k_index,
        top_k_weights,
        w1,
        None,
        w2,
        None,
        (A1_d, B1_d),
        (A2_d, B2_d),
        E,
        act="silu",
        backend="dequant",
        concat=True,
        scaling1=scaling1,
        scaling2=scaling2,
    )
    diff = (out.float() - out_dq.float()).abs()
    denom = out_dq.float().abs()
    rel = float(diff.mean() / denom.mean())
    rel_fro = float((out.float() - out_dq.float()).norm() / out_dq.float().norm())
    print(
        f"[info] fp4_cute vs dequant on real weights (W4A4 error, OQ1): "
        f"mean_rel={rel:.4e} rel_fro={rel_fro:.4e}"
    )

    finish()


if __name__ == "__main__":
    main()
