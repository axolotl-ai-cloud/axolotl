# Ternary (1.58-bit) conversion

Converts a full-precision checkpoint into a ternary (`{-1, 0, +1}`) model and heals it
with continued training, then exports artifacts that run today.

The plugin swaps the transformer-block Linears for `TernaryLinear` (per-tensor absmean
weight scale, per-token int8 activations, straight-through estimator on bf16 latents),
warms the quantization strength λ in from 0 so the model is not shocked into ternary,
anneals weight decay to 0 mid-run so the codes settle, and bakes the latents to exact
`{-s, 0, +s}` values at save time. Embeddings, `lm_head` and norms stay full precision.

Ternary conversion is a full-finetune operation: adapters and any other quantization
path (`use_onebitllms`, `qat:`, `quantization:`, `load_in_*bit`) are rejected.

## Minimal config

```yaml
base_model: meta-llama/Llama-3.2-1B

plugins:
  - axolotl.integrations.ternary.TernaryPlugin

ternary:
  lambda_warmup_steps: 1000
  export:
    formats: [master_bf16, hf_bitnet]

pretraining_dataset:
  - path: HuggingFaceFW/fineweb-edu
    type: pretrain
streaming: true

learning_rate: 1e-4
save_only_model: true
```

Module selection defaults to the architecture preset (`llama`, `qwen3`); every `nn.Linear`
must fullmatch exactly one of `target_modules`/`keep_fp_modules` or the swap is a hard
error. Training logs `ternary/flip_rate/<family>` and `ternary/zero_frac/<family>` every
`log_code_flip_every` steps.

## Export

`ternary.export.formats` are written when training finishes. The same artifacts can be
produced later from a saved master:

```bash
axolotl ternary export config.yaml --format gguf_tq2_0 --output-dir ./exports
```

| Format | Runtime |
|---|---|
| `master_bf16` | transformers; the refinetunable interchange artifact |
| `hf_bitnet` | transformers `BitNetForCausalLM` |
| `gguf_tq2_0` / `gguf_tq1_0` | llama.cpp (needs the optional `gguf` package) |
| `i2_s` | bitnet.cpp (needs the optional `gguf` package) |

Every packed export is gated against the master: exact code equality, an fp16-bounded
dequant error, and a smoke eval.

## Layout

| Module | Role |
|---|---|
| `args.py` | `ternary:` schema and cross-config validation |
| `quant.py` | eager fake-quant math — the semantic oracle every kernel must match |
| `modules.py` | `TernaryLinear`, the bake hook |
| `swap.py` | strict regex-enumerated module surgery + swap manifest |
| `callbacks.py` | λ schedule, weight-decay anneal, code-flip/zero-fraction monitoring |
| `distill.py` | in-process frozen-teacher trainer |
| `kernels/` | fused Triton fake-quant/act-quant/stats, int8 W2A8 forward |
| `export/` | bake, packers, parity gates |
| `cli.py` | `axolotl ternary export` |

## Expectations

Healing needs tokens: ~1B is demo quality, ~10B is usable, 30B+ is where a converted
model becomes generally good. Budget accordingly before starting a run.

See [docs/ternary_conversion.qmd](../../../../docs/ternary_conversion.qmd) for the full
config reference, recipes (pure QAT, KD-plugin teacher, in-process distillation) and the
export/parity details.
