# Finetune OpenGV's InternVL with Axolotl

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install Cut Cross Entropy following [docs](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy).

3. Install timm

2. Run the below

```bash
# QLoRA SFT linear layers (1xXYGB @ ~AB GiB)
axolotl train examples/internvl3_5/internvl3_5-8b-qlora.yml
```

Note: Memory usage taken from `device_mem_reserved(gib)` from logs.
