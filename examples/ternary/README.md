# Ternary (1.58-bit) conversion

Convert a full-precision checkpoint into a ternary (`{-1, 0, +1}`) model, heal it with
continued training, and export artifacts that run in transformers, llama.cpp or
bitnet.cpp.

| Config | Recipe |
|---|---|
| `llama-3.2-1b-qat.yaml` | Pure QAT heal on streamed pretraining text — no teacher. The default recipe. |
| `qwen3-4b-distill.yaml` | Same heal plus an in-process frozen FP teacher (logits + hidden-state KD). |
| `llama-3.2-1b-ptq-init.yaml` | Calibrated PTQ init (`init: ternary_fit_calibrated` with `weight_scale: learnable`), KD anchored to the LR-decay tail, and a sparsity-adaptive `mask_sign` export. |

```bash
axolotl ternary damage-map examples/ternary/llama-3.2-1b-ptq-init.yaml
axolotl train examples/ternary/llama-3.2-1b-ptq-init.yaml
axolotl ternary export examples/ternary/llama-3.2-1b-qat.yaml --format gguf_tq2_0
```

Run the damage map first: it swaps the model and evaluates it without training, so the cost
of ternary weights, int8 activations and each module family is a number before any budget is
committed. It always measures the quantizer against the untouched FP weights — a fitted
`init` and `subln` are switched off for its passes, since neither is function-preserving at
λ=0 and both would move the baseline it compares to. Weight-dominant damage is what the
fitted init and a longer λ warmup buy down; a single outlier family belongs in
`keep_fp_modules`.

Healing needs tokens: ~1B is demo quality, ~10B is usable, 30B+ is where a converted
model becomes generally good. All three configs are sized as starting points — raise
`max_steps` to hit the budget you actually want.

See [docs/ternary_conversion.qmd](../../docs/ternary_conversion.qmd) for the full config
reference, teacher options and export details.
