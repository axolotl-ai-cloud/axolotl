#!/usr/bin/env python3
"""Instantiate a ~8.3B DeepSeek-V3 MoE model with random weights.

Run this on a GPU-equipped machine (e.g. 1× NVL H100) so the dense
initialization completes quickly:

    python scripts/benchmarks/build_deepseek_v3_8b.py --output deepseek-v3-8b-moe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def build_config() -> DeepseekV3Config:
    """Return a DeepSeek V3 configuration totaling ~8.3B parameters."""

    return DeepseekV3Config(
        vocab_size=32_000,
        hidden_size=3_072,
        intermediate_size=8_192,
        moe_intermediate_size=2_560,
        num_hidden_layers=20,
        num_attention_heads=24,
        num_key_value_heads=24,
        n_routed_experts=18,
        num_experts_per_tok=4,
        n_group=6,
        topk_group=4,
        kv_lora_rank=192,
        q_lora_rank=384,
        max_position_embeddings=2_048,
        rope_theta=10_000.0,
        rope_interleave=True,
        hidden_act="silu",
        initializer_range=0.02,
        attention_dropout=0.0,
        attention_bias=False,
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save the generated model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=DTYPE_MAP.keys(),
        help="Storage dtype for the checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config()
    model = DeepseekV3ForCausalLM(config)

    dtype = DTYPE_MAP[args.dtype]
    model.to(dtype=dtype)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Initialized DeepSeek-V3 MoE with {param_count / 1e9:.3f}B parameters")

    model.save_pretrained(output_dir, safe_serialization=True)
    config.save_pretrained(output_dir)
    print(f"Saved model and config to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
