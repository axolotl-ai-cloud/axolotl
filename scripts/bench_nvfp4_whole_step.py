# mypy: disable-error-code="operator"
#!/usr/bin/env python
"""Whole-step profile of an NVFP4 transformer-block stack (Qwen3-8B shapes).

Profiles fwd+bwd+optimizer over N compiled decoder blocks — LoRA-r16 over
frozen NVFP4 compute-base linears (the benchmark-yaml path) or full-FFT
NVFP4Linear — and ranks every CUDA kernel by self time. This is the step-level
complement to the MLP-only profile in bench_nvfp4_epilogue_fusion.py.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
    python scripts/bench_nvfp4_whole_step.py --mode lora --layers 2
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.integrations.nvfp4.nvfp4_training import (
    NVFP4FastComputeBaseLinear,
    NVFP4Linear,
    NVFP4Recipe,
    nvfp4_supported,
)

HIDDEN = 4096
INTER = 12288
N_HEADS = 32
N_KV = 8
HEAD_DIM = 128
EPS = 1e-6


class LoRALinear(nn.Module):
    """PEFT lora.Linear equivalent: frozen NVFP4 base + bf16 r16 adapters."""

    def __init__(self, in_f, out_f, r=16, alpha=32):
        super().__init__()
        base = nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.bfloat16)
        self.base = NVFP4FastComputeBaseLinear.from_linear(base, NVFP4Recipe())
        self.lora_A = nn.Linear(
            in_f, r, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.lora_B = nn.Linear(
            r, out_f, bias=False, device="cuda", dtype=torch.bfloat16
        )
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


class FFTLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        lin = nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.bfloat16)
        self.inner = NVFP4Linear.from_linear(lin, NVFP4Recipe())

    def forward(self, x):
        return self.inner(x)


def rms_norm(x, w, eps=EPS):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


class Block(nn.Module):
    def __init__(self, make):
        super().__init__()
        self.input_ln = nn.Parameter(
            torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        )
        self.post_ln = nn.Parameter(
            torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        )
        self.q_norm = nn.Parameter(
            torch.ones(HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        self.k_norm = nn.Parameter(
            torch.ones(HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        self.q_proj = make(HIDDEN, N_HEADS * HEAD_DIM)
        self.k_proj = make(HIDDEN, N_KV * HEAD_DIM)
        self.v_proj = make(HIDDEN, N_KV * HEAD_DIM)
        self.o_proj = make(N_HEADS * HEAD_DIM, HIDDEN)
        self.gate_proj = make(HIDDEN, INTER)
        self.up_proj = make(HIDDEN, INTER)
        self.down_proj = make(INTER, HIDDEN)

    def forward(self, x, cos, sin):
        from flash_attn import flash_attn_func

        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        b, s, _ = x.shape
        h = rms_norm(x, self.input_ln)
        q = self.q_proj(h).view(b, s, N_HEADS, HEAD_DIM)
        k = self.k_proj(h).view(b, s, N_KV, HEAD_DIM)
        v = self.v_proj(h).view(b, s, N_KV, HEAD_DIM)
        q = fused_rms_norm_rope(q, self.q_norm, cos, sin, eps=EPS)
        k = fused_rms_norm_rope(k, self.k_norm, cos, sin, eps=EPS)
        attn = flash_attn_func(q, k, v, causal=True)
        x = x + self.o_proj(attn.reshape(b, s, -1))
        h = rms_norm(x, self.post_ln)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class Stack(nn.Module):
    def __init__(self, n_layers, make):
        super().__init__()
        self.blocks = nn.ModuleList([Block(make) for _ in range(n_layers)])
        inv = 1.0 / (
            1e6 ** (torch.arange(0, HEAD_DIM, 2, device="cuda").float() / HEAD_DIM)
        )
        t = torch.arange(4096, device="cuda").float()
        f = torch.outer(t, inv)
        emb = torch.cat((f, f), dim=-1)
        self.register_buffer("cos", emb.cos().to(torch.bfloat16))
        self.register_buffer("sin", emb.sin().to(torch.bfloat16))

    def forward(self, x):
        s = x.shape[1]
        cos, sin = self.cos[None, :s], self.sin[None, :s]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


def _categorize(name: str) -> str:
    low = name.lower()
    if "flash" in low or "fmha" in low:
        return "attention"
    if (
        "nvjet" in low
        or "cutlass" in low
        or "cublas" in low
        or "gemm" in low
        or ("matmul" in low and "triton" not in low)
    ):
        return "gemm"
    if "adam" in low or "foreach" in low or "lerp" in low:
        return "optimizer"
    if "quantize" in low or "nvfp4" in low or "recipe" in low or "amax" in low:
        return "quant"
    if "rope" in low or "norm_rope" in low:
        return "rope/qknorm"
    if "reduce" in low or "norm" in low:
        return "reduction"
    if "memcpy" in low or "memset" in low:
        return "memcpy"
    if low.startswith("triton_") or "elementwise" in low or "vectorized" in low:
        return "elementwise/inductor"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lora", "fft"], default="lora")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trace", type=str, default="", help="chrome trace output path")
    args = ap.parse_args()

    ok, why = nvfp4_supported()
    assert ok, why
    torch.manual_seed(0)

    make = LoRALinear if args.mode == "lora" else FFTLinear
    model = Stack(args.layers, make)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(
        f"mode={args.mode} layers={args.layers} trainable params={n_train / 1e6:.2f}M"
    )
    opt = torch.optim.AdamW(params, lr=1e-4, fused=True)

    x = torch.randn(args.batch, args.seq, HIDDEN, device="cuda", dtype=torch.bfloat16)
    fwd = model if args.no_compile else torch.compile(model)

    def step():
        opt.zero_grad(set_to_none=True)
        out = fwd(x)
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
        return loss

    for _ in range(8):
        step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        step()
    end.record()
    torch.cuda.synchronize()
    wall = start.elapsed_time(end) / args.iters
    print(f"wall per step: {wall:.3f} ms  ({args.batch * args.seq} tokens)")

    from torch.profiler import ProfilerActivity, profile

    prof_iters = 10
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(prof_iters):
            step()
        torch.cuda.synchronize()

    cats: dict[str, float] = defaultdict(float)
    names: dict[str, float] = defaultdict(float)
    for evt in prof.key_averages():
        if evt.device_type.name != "CUDA" or evt.self_device_time_total == 0:
            continue
        cats[_categorize(evt.key)] += evt.self_device_time_total
        names[evt.key] += evt.self_device_time_total

    total = sum(cats.values())
    print(
        f"\nprofiled CUDA busy: {total / 1e3 / prof_iters:.3f} ms/iter "
        f"(wall {wall:.3f} -> gap/launch {wall - total / 1e3 / prof_iters:.3f} ms)"
    )
    print(f"per-layer CUDA busy: {total / 1e3 / prof_iters / args.layers:.3f} ms")
    for cat, us in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(
            f"  {cat:22s} {us / 1e3 / prof_iters:8.3f} ms/iter  {100 * us / total:5.1f}%"
        )
    print("\n  -- top kernels (self CUDA, ms/iter) --")
    for name, us in sorted(names.items(), key=lambda kv: -kv[1])[:40]:
        print(
            f"  [{_categorize(name):>20s}] {us / 1e3 / prof_iters:7.3f}  {name[:110]}"
        )

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"trace -> {args.trace}")


if __name__ == "__main__":
    main()
