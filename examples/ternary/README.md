# Ternary (1.58-bit) conversion

Convert a full-precision checkpoint into a ternary (`{-1, 0, +1}`) model, heal it with
continued training, and export artifacts that run in transformers, llama.cpp or
bitnet.cpp.

| Config | Recipe |
|---|---|
| `llama-3.2-1b-qat.yaml` | Pure QAT heal on streamed pretraining text — no teacher. The default recipe. |
| `qwen3-4b-distill.yaml` | Same heal plus an in-process frozen FP teacher (logits + hidden-state KD). |

```bash
axolotl train examples/ternary/llama-3.2-1b-qat.yaml
axolotl ternary export examples/ternary/llama-3.2-1b-qat.yaml --format gguf_tq2_0
```

Healing needs tokens: ~1B is demo quality, ~10B is usable, 30B+ is where a converted
model becomes generally good. Both configs are sized as starting points — raise
`max_steps` to hit the budget you actually want.

See [docs/ternary_conversion.qmd](../../docs/ternary_conversion.qmd) for the full config
reference, teacher options and export details.
