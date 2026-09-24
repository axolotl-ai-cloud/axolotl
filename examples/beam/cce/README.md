# CCE patch

`beam.py` belongs in [ml-cross-entropy](https://github.com/axolotl-ai-cloud/ml-cross-entropy),
not in this repo — it is kept here so it travels with the rest of the Beam support.

Install:

```bash
cp beam.py <ml-cross-entropy>/cut_cross_entropy/transformers/beam.py
```

and register both spellings in `cut_cross_entropy/transformers/patch.py`:

```python
"beam": ("cut_cross_entropy.transformers.beam", "patch_beam"),
"reflection": ("cut_cross_entropy.transformers.beam", "patch_reflection"),
```

Without a registry entry, enabling CCE raises `Unknown model type`.

Adapted from `gemma3.py` (the softcap reference). Unlike gemma3 this architecture is
MoE, so the patch preserves the router load-balancing aux loss and returns
`MoeCausalLMOutputWithPast`.

Measured on the tiny model, bf16 throughout:

| seq | loss base vs CCE | peak memory | saved |
|---|---|---|---|
| 512 | 12.031174 / 12.031157 (1.7e-05) | 1908 → 716 MiB | 62% |
| 2048 | 12.035582 / 12.035566 (1.5e-05) | 7262 → 1487 MiB | 80% |

Unverified: the `lm_head_dtype: fp32` path, and gradient agreement on real weights
(relative grad diff was 1.6e-02 on random weights, consistent with bf16 accumulation
order but not confirmed).
