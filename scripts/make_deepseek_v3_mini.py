#!/usr/bin/env python
"""Create a randomly initialized Deepseek-V3-mini model and save to a directory.

Example:
  python scripts/make_deepseek_v3_mini.py --out ./deepseek-v3-mini-1.3b --params 1300m

Or custom sizes:
  python scripts/make_deepseek_v3_mini.py --out ./custom --vocab 32000 --hidden 2048 --layers 24 --heads 16 --ff 8192 --experts 8 --topk 2
"""

import argparse
from pathlib import Path

from axolotl.models.deepseek_v3_mini import (
    DeepseekV3MiniConfig,
    DeepseekV3MiniForCausalLM,
)


PRESETS = {
    # rough ballparks; active params smaller due to MoE sparsity
    "350m": dict(
        vocab=32000, hidden=1024, layers=16, heads=16, ff=4096, experts=4, topk=2
    ),
    "700m": dict(
        vocab=32000, hidden=1536, layers=20, heads=16, ff=6144, experts=4, topk=2
    ),
    "1300m": dict(
        vocab=32000, hidden=2048, layers=24, heads=16, ff=8192, experts=8, topk=2
    ),
    "3000m": dict(
        vocab=32000, hidden=2560, layers=28, heads=20, ff=10240, experts=8, topk=2
    ),
    # aliases and larger presets
    "3b": dict(
        vocab=32000, hidden=2560, layers=28, heads=20, ff=10240, experts=8, topk=2
    ),
    "7000m": dict(
        vocab=32000, hidden=4096, layers=32, heads=32, ff=16384, experts=16, topk=2
    ),
    "7b": dict(
        vocab=32000, hidden=4096, layers=32, heads=32, ff=16384, experts=16, topk=2
    ),
}


def build_config(args: argparse.Namespace) -> DeepseekV3MiniConfig:
    if args.params:
        spec = PRESETS[args.params.lower()]
        vocab = spec["vocab"]
        hidden = spec["hidden"]
        layers = spec["layers"]
        heads = spec["heads"]
        ff = spec["ff"]
        experts = spec["experts"]
        topk = spec["topk"]
    else:
        vocab = args.vocab
        hidden = args.hidden
        layers = args.layers
        heads = args.heads
        ff = args.ff
        experts = args.experts
        topk = args.topk

    return DeepseekV3MiniConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=ff,
        num_experts=experts,
        top_k=topk,
        num_shared_experts=args.shared_experts,
        max_position_embeddings=args.max_pos,
        dropout=args.dropout,
        # Save as safetensors without shared weights between embeddings and lm_head
        tie_word_embeddings=False,
        router_score_fn="sigmoid",
        route_norm=True,
        route_scale=1.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--params", choices=list(PRESETS.keys()))
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--ff", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--shared_experts", type=int, default=0)
    ap.add_argument("--max_pos", type=int, default=4096)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    cfg = build_config(args)
    model = DeepseekV3MiniForCausalLM(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(out)
    # Save weights using safetensors (default) with untied embeddings
    model.save_pretrained(out, safe_serialization=True)
    print(f"Saved randomly initialized model (safetensors, untied) to {out}")


if __name__ == "__main__":
    main()
