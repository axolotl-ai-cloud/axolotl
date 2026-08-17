# Preference Learning (RLHF) — Agent Reference

Reference for DPO, IPO, KTO, ORPO, and SimPO. For config templates and dataset format examples, see [rlhf.qmd](../rlhf.qmd). For GRPO, see [grpo.qmd](../grpo.qmd). For EBFT, see [ebft.qmd](../ebft.qmd).

## Method Overview

| Method | Data Requirement | Key Idea | Best For |
|--------|-----------------|----------|----------|
| **DPO** | Paired (chosen + rejected) | Implicit reward via preference pairs | General alignment, most common |
| **IPO** | Paired (chosen + rejected) | DPO with different loss (avoids overfitting) | When DPO overfits |
| **KTO** | Unpaired (completion + binary label) | Kahneman-Tversky loss, no pairs needed | When you only have thumbs-up/down |
| **ORPO** | Paired (chosen + rejected) | Combined SFT + preference, no ref model | Single-stage alignment, saves VRAM |
| **SimPO** | Paired (chosen + rejected) | Length-normalized, no ref model | Simple setup, length-robust |

Default: start with DPO. All methods require `sample_packing: false`.

## Architecture

```
┌──────────────┐   ┌───────────────┐   ┌───────────────┐
│ Policy Model │   │ Reference     │   │ Preference    │
│ (trainable)  │   │ Model (frozen)│   │ Dataset       │
└──────┬───────┘   └──────┬────────┘   └──────┬────────┘
       └──────────┬───────┘                    │
                  v                            │
       Forward pass on chosen + rejected <─────┘
                  │
       Preference Loss (DPO/IPO/KTO/...)
                  │
       Backprop + Update

Exception: ORPO and SimPO do NOT use a reference model (~50% less VRAM).
```

No vLLM server needed (unlike GRPO). Offline RL with pre-collected preference data.

## Method Selection

1. Paired preference data (chosen + rejected)?
   - Default → `rl: dpo`
   - Overfitting → `rl: dpo, dpo_loss_type: ["ipo"]`
   - VRAM-limited → `rl: orpo` (no ref model)
   - Length-sensitive → `rl: simpo` (no ref model)
2. Only binary labels (good/bad)? → `rl: kto`
3. Single-stage training (no separate SFT)? → `rl: orpo`

| | DPO | IPO | KTO | ORPO | SimPO |
|---|---|---|---|---|---|
| **Reference model** | Yes | Yes | Yes | No | No |
| **VRAM overhead** | ~2x model | ~2x model | ~2x model | ~1x model | ~1x model |
| **TRL trainer class** | DPOTrainer | DPOTrainer | KTOTrainer | ORPOTrainer | CPOTrainer |

## Prompt Strategy Resolution

The `type` field resolves to a Python function:

```
type: "chatml.intel"
  → axolotl.prompt_strategies.dpo.chatml.intel(cfg, **kwargs)
  → returns transform_fn(sample) → {"prompt", "chosen", "rejected"}

type: "chat_template.default"
  → axolotl.prompt_strategies.dpo.chat_template.default(cfg, dataset_idx, **kwargs)

type: {"field_prompt": "prompt", ...}   (dict)
  → axolotl.prompt_strategies.dpo.user_defined.default(...)
```

Module base: `axolotl.prompt_strategies.{rl_method}` — replace `dpo` with `kto` or `orpo`.

## Healthy Training Indicators

| Metric | Healthy Range | Problem |
|--------|--------------|---------|
| `train/loss` | Decreasing, 0.3-0.7 | Flat or increasing = broken data or too high LR |
| `rewards/chosen` | Increasing | Flat = model not learning preferences |
| `rewards/rejected` | Decreasing | Increasing = model prefers wrong responses |
| `rewards/margins` | Positive and increasing | Negative = prefers rejected over chosen |
| `rewards/accuracies` | > 0.5, toward 0.7+ | < 0.5 = worse than random |
| `logps/rejected` | Decreasing | Increasing = reward hacking |
| `grad_norm` | 0.01 - 10.0 | > 100 = exploding gradients |

Method-specific: DPO/IPO watch `rewards/margins`; KTO loss is noisier; ORPO monitor SFT + odds ratio components; SimPO check length-normalized reward separation.

## Known Issues

| Issue | Fix |
|-------|-----|
| Sample packing crash | Set `sample_packing: false` (required for all preference methods) |
| KTO `KeyError: 'label'` | Ensure dataset has boolean `label` column |
| ORPO/KTO `KeyError` during tokenization | Add `remove_unused_columns: false` |
| ORPO template not applied | ORPO requires explicit `chat_template` setting |
| OOM with ref model (DPO/IPO/KTO) | Use LoRA/QLoRA, or switch to ORPO/SimPO (no ref model) |
| IPO + label_smoothing | Do not set `dpo_label_smoothing` when `rl: ipo` |

Full troubleshooting: [training_stability.qmd](../training_stability.qmd)

## File Map

```
src/axolotl/
  core/trainers/dpo/              # DPO trainer, args, strategy
  core/builders/rl.py             # HFRLTrainerBuilder — routes rl type → trainer class
  core/training_args.py           # AxolotlKTOConfig, AxolotlORPOConfig, AxolotlCPOConfig
  prompt_strategies/
    dpo/                          # DPO/IPO/SimPO dataset strategies
      chat_template.py            # chat_template.default, chat_template.argilla_chat
      chatml.py                   # chatml.default/intel/icr/argilla_chat/prompt_pairs/ultra
      llama3.py                   # llama3 variants (same subtypes as chatml)
      user_defined.py             # Custom field mapping
      passthrough.py              # No transform
    kto/                          # KTO dataset strategies (chatml, llama3, user_defined)
    orpo/                         # ORPO dataset strategies (chat_template.argilla)
  utils/schemas/enums.py          # RLType enum (dpo, ipo, kto, orpo, simpo, grpo, gdpo, ebft)
  utils/schemas/config.py         # All rl/dpo/kto/orpo/simpo config fields

docs/rlhf.qmd                    # Full user docs: all dataset formats, config templates
docs/choosing_method.qmd          # SFT vs DPO vs GRPO decision guide
examples/qwen2/dpo.yaml           # DPO example
examples/llama-3/qlora-1b-kto.yaml  # KTO example
```
