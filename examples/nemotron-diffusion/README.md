# Nemotron diffusion

Native diffusion recipes for Nemotron-Labs-Diffusion. See
[`docs/diffusion_lm.qmd`](../../docs/diffusion_lm.qmd) for installation,
attention, packing, objective options, adapter constraints and the
typed-decision dataset format.

| Config | Purpose |
|---|---|
| `lora-smoke.yaml` | Pinned 3B, two-step chat LoRA smoke with packing; not a quality recipe |
| `decision-lora-8b.yaml` | 8B typed-decision LoRA starting recipe; set your own train/dev JSONL paths |
| `decision-lora-8b-public-procedural.yaml` | 8B typed-decision LoRA on the public procedural mix from `scripts/diffusion_lm/build_public_decision_mix.py` |

```bash
axolotl preprocess examples/nemotron-diffusion/lora-smoke.yaml
axolotl train examples/nemotron-diffusion/lora-smoke.yaml
```
