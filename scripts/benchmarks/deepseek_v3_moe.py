#!/usr/bin/env python
# mypy: ignore-errors
"""Microbenchmark for DeepSeek V3 MoE block comparing baseline vs Triton CG kernels."""

from __future__ import annotations

import argparse
import time
from types import MethodType

import torch

try:
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config,
    )
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
except ImportError as exc:  # pragma: no cover - utility script
    raise SystemExit(
        "Transformers with DeepSeek-V3 support must be available in PYTHONPATH"
    ) from exc

from axolotl.monkeypatch.deepseek_v3 import patch_deepseek_v3_moe

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--seq-len", type=int, default=2048, help="sequence length")
    parser.add_argument("--hidden-size", type=int, default=4096, help="MoE hidden size")
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=8192,
        help="MoE intermediate projection size",
    )
    parser.add_argument(
        "--n-experts",
        type=int,
        default=256,
        help="Number of routed experts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of experts per token",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=8,
        help="Router groups (must divide n-experts)",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="bf16",
        help="Computation dtype",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=25, help="Benchmark iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--uniform-routing",
        action="store_true",
        help="Override router to distribute tokens evenly across experts",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="GROUP_SIZE_M used by the Triton kernel",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_module(args: argparse.Namespace) -> DeepseekV3MoE:
    config = DeepseekV3Config(
        hidden_size=args.hidden_size,
        intermediate_size=args.moe_intermediate_size,
        moe_intermediate_size=args.moe_intermediate_size,
        n_routed_experts=args.n_experts,
        num_experts_per_tok=args.top_k,
        n_group=args.groups,
        topk_group=max(1, min(args.groups, args.top_k)),
        n_shared_experts=1,
    )
    module = DeepseekV3MoE(config)
    module.eval()
    return module


@torch.no_grad()
def timed_forward(
    module: DeepseekV3MoE, inputs: torch.Tensor, iters: int, warmup: int
) -> float:
    for _ in range(warmup):
        module(inputs)
    if inputs.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        module(inputs)
    if inputs.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def benchmark_deepseek_v3(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    print(f"device: {device}, dtype: {dtype}")

    if args.n_experts % args.groups != 0:
        raise SystemExit("n-experts must be divisible by groups")
    if args.top_k > args.n_experts:
        raise SystemExit("top-k cannot exceed number of experts")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    baseline_module = build_module(args)
    original_moe = DeepseekV3MoE.moe
    baseline_module.moe = MethodType(original_moe, baseline_module)
    state_dict = baseline_module.state_dict()

    patch_deepseek_v3_moe(group_size_m=args.group_size)
    patched_module = build_module(args)
    patched_module.load_state_dict(state_dict)

    baseline_module.to(device=device, dtype=dtype)
    patched_module.to(device=device, dtype=dtype)

    tokens = args.batch * args.seq_len
    routed_tokens = tokens * args.top_k
    avg_tokens_per_expert = routed_tokens / args.n_experts

    inputs = torch.randn(
        args.batch,
        args.seq_len,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        flat_inputs = inputs.view(-1, args.hidden_size)
        if args.uniform_routing:
            total_assignments = flat_inputs.size(0) * args.top_k
            base = total_assignments // args.n_experts
            remainder = total_assignments % args.n_experts
            counts = torch.full(
                (args.n_experts,),
                base,
                dtype=torch.int64,
                device=device,
            )
            if remainder:
                counts[:remainder] += 1
            assignments = torch.repeat_interleave(
                torch.arange(args.n_experts, device=device), counts
            )
            assignments = assignments[torch.randperm(assignments.size(0))]
            topk_idx = assignments.view(flat_inputs.size(0), args.top_k)
        else:
            topk_idx, _ = patched_module.gate(flat_inputs)

        tokens_per_expert = torch.bincount(
            topk_idx.reshape(-1), minlength=args.n_experts
        )
        min_tokens = int(tokens_per_expert.min().item())
        max_tokens = int(tokens_per_expert.max().item())

    if args.uniform_routing:
        weights = torch.full(
            topk_idx.shape,
            1.0 / args.top_k,
            device=device,
            dtype=torch.float32,
        )

        def _uniform_gate(self, hidden_states):
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            token_count = flat.shape[0]
            return topk_idx[:token_count], weights[:token_count]

        patched_module.gate.forward = _uniform_gate.__get__(
            patched_module.gate, patched_module.gate.__class__
        )
        baseline_module.gate.forward = _uniform_gate.__get__(
            baseline_module.gate, baseline_module.gate.__class__
        )

    with torch.no_grad():
        ref_output = baseline_module(inputs)
        patched_output = patched_module(inputs)
        max_diff = (ref_output - patched_output).abs().max().item()

    baseline_ms = timed_forward(baseline_module, inputs, args.iters, args.warmup)
    patched_ms = timed_forward(patched_module, inputs, args.iters, args.warmup)

    speedup = baseline_ms / patched_ms if patched_ms > 0 else float("nan")

    return {
        "device": device,
        "dtype": dtype,
        "baseline_ms": baseline_ms,
        "patched_ms": patched_ms,
        "speedup": speedup,
        "max_diff": max_diff,
        "routed_tokens": routed_tokens,
        "avg_tokens": avg_tokens_per_expert,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
    }


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = parse_args()
    result = benchmark_deepseek_v3(args)

    print(
        f"Device={result['device'].type} dtype={result['dtype']} batch={args.batch} seq={args.seq_len} hidden={args.hidden_size}"
    )
    print(
        f"routed tokens={result['routed_tokens']} avg tokens/expert={result['avg_tokens']:.1f} group_size={args.group_size}"
    )
    print(f"min/max tokens per expert: {result['min_tokens']}/{result['max_tokens']}")
    print(
        f"Baseline: {result['baseline_ms']:.3f} ms | Patched: {result['patched_ms']:.3f} ms | x{result['speedup']:.2f}"
    )
    print(f"Max |Δ| between outputs: {result['max_diff']:.2e}")


if __name__ == "__main__":
    main()
