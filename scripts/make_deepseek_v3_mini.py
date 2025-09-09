#!/usr/bin/env python
"""Create a randomly initialized Hugging Face DeepSeek-V3 model (smaller sizes) and save it.

Examples:
  # preset ~1.3B
  python scripts/make_deepseek_v3_mini.py --out ./models/deepseek-v3-1_3b --params 1300m --tokenizer axolotl-ai-co/DeepSeek-V3-1B

  # custom small DeepSeek-V3
  python scripts/make_deepseek_v3_mini.py --out ./models/custom --vocab 32000 --hidden 2560 --layers 28 --heads 20 --ff 10240 --experts 8 --topk 2 --tokenizer axolotl-ai-co/DeepSeek-V3-1B
"""

import argparse
import contextlib
from pathlib import Path

import torch
from transformers import AutoTokenizer

from transformers.models.deepseek_v3 import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
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


def build_config(args: argparse.Namespace) -> DeepseekV3Config:
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

    cfg = DeepseekV3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=ff,
        # MoE
        num_experts=experts,
        num_experts_per_tok=topk,
        num_shared_experts=args.shared_experts,
        max_position_embeddings=args.max_pos,
        dropout=args.dropout,
        # Tie embeddings by default to reduce memory unless --no_tie is passed
        tie_word_embeddings=args.tie,
    )

    return cfg


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
    ap.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Optional tokenizer name/path to copy into the output directory. "
            "Use a 32k tokenizer (e.g., axolotl-ai-co/DeepSeek-V3-1B or a local path) to match the default vocab_size."
        ),
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Parameter dtype for initialization to reduce host RAM (default: bf16)",
    )
    ap.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="Sharded safetensors max shard size (e.g., '1GB', '2GB').",
    )
    ap.add_argument(
        "--tie",
        action="store_true",
        help="Tie lm_head to embed_tokens to reduce memory (recommended)",
    )
    ap.add_argument(
        "--no_tie",
        dest="tie",
        action="store_false",
        help="Do not tie embeddings (uses more memory)",
    )
    ap.set_defaults(tie=True)
    args = ap.parse_args()

    cfg = build_config(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Initialize directly in requested dtype to reduce host RAM
    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    with _set_default_dtype(target_dtype if target_dtype != torch.float32 else None):
        model = DeepseekV3ForCausalLM(cfg)
    cfg.save_pretrained(out)
    # Save weights using sharded safetensors
    model.save_pretrained(
        out,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    # Optionally save tokenizer alongside the model
    if args.tokenizer:
        try:
            tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
            tok.save_pretrained(out)
        except Exception as e:
            print(f"Warning: failed to copy tokenizer from {args.tokenizer}: {e}")
    tie_str = "tied" if args.tie else "untied"
    print(
        f"Saved randomly initialized DeepSeek-V3 model (safetensors shards up to {args.max_shard_size}, {args.dtype}, {tie_str}) to {out}"
    )


@contextlib.contextmanager
def _set_default_dtype(dtype: torch.dtype | None):
    if dtype is None:
        yield
        return
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


if __name__ == "__main__":
    main()
