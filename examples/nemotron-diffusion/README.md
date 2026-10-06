# Nemotron diffusion

Native diffusion recipes for Nemotron-Labs-Diffusion. See
[`docs/diffusion_lm.qmd`](../../docs/diffusion_lm.qmd) for installation,
attention, packing, objective options and adapter constraints.

| Config | Purpose |
|---|---|
| `lora-smoke.yaml` | Pinned 3B, two-step chat LoRA smoke with packing; not a quality recipe |

```bash
axolotl preprocess examples/nemotron-diffusion/lora-smoke.yaml
axolotl train examples/nemotron-diffusion/lora-smoke.yaml
```
