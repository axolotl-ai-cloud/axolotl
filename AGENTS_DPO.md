# Preference Learning (DPO/IPO/KTO/ORPO/SimPO) -- Agent Reference

Reference for AI agents working with axolotl's preference-based alignment training. Describes methods, dataset formats, configuration, and known issues.

## Overview -- When to Use What

| Method | Data Requirement | Key Idea | Best For |
|--------|-----------------|----------|----------|
| **DPO** | Paired (chosen + rejected) | Implicit reward via preference pairs | General alignment, most common |
| **IPO** | Paired (chosen + rejected) | DPO with different loss (avoids overfitting) | When DPO overfits |
| **KTO** | Unpaired (completion + binary label) | Kahneman-Tversky loss, no pairs needed | When you only have thumbs-up/down |
| **ORPO** | Paired (chosen + rejected) | Combined SFT + preference, no ref model | Single-stage alignment, saves VRAM |
| **SimPO** | Paired (chosen + rejected) | Length-normalized, no ref model | Simple setup, length-robust |

**Default recommendation**: Start with DPO. It has the most dataset support, is well-understood, and works reliably.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Single GPU (or multi-GPU with FSDP/DeepSpeed)             │
│                                                             │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────┐  │
│  │ Policy Model │   │ Reference     │   │ Preference    │  │
│  │ (trainable)  │   │ Model (frozen)│   │ Dataset       │  │
│  └──────┬───────┘   └──────┬────────┘   └──────┬────────┘  │
│         │                  │                    │           │
│         └──────────┬───────┘                    │           │
│                    v                            │           │
│         ┌──────────────────┐                    │           │
│         │ Forward pass on  │<───────────────────┘           │
│         │ chosen + rejected│                                │
│         └────────┬─────────┘                                │
│                  v                                          │
│         ┌──────────────────┐                                │
│         │ Preference Loss  │                                │
│         │ (DPO/IPO/KTO/..) │                                │
│         └────────┬─────────┘                                │
│                  v                                          │
│         ┌──────────────────┐                                │
│         │ Backprop + Update│                                │
│         └──────────────────┘                                │
└─────────────────────────────────────────────────────────────┘

Exception: ORPO and SimPO do NOT use a reference model.
```

Unlike GRPO, preference learning does not require a vLLM server. It is offline RL: the dataset contains pre-collected responses. The reference model is a frozen copy of the initial policy, loaded automatically by TRL.

## Components Required

A preference training run requires two things:

1. **A YAML config** -- model, dataset, RL method, training hyperparameters
2. **A preference dataset** -- HuggingFace dataset with the right fields for the chosen prompt strategy

No reward functions, no vLLM server, no custom Python modules (unless using `user_defined` format).

## Quick Start -- DPO with chat_template

Minimal working config for DPO:

```yaml
base_model: Qwen/Qwen2.5-0.5B

chat_template: qwen_25
rl: dpo

datasets:
  - path: fozziethebeat/alpaca_messages_2k_dpo_test
    type: chat_template.default
    field_messages: conversation
    field_chosen: chosen
    field_rejected: rejected
    message_property_mappings:
      role: role
      content: content
    roles:
      system:
        - system
      user:
        - user
      assistant:
        - assistant

sequence_len: 2048
sample_packing: false
output_dir: ./outputs/dpo-out

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
gradient_checkpointing: true
flash_attention: true

warmup_ratio: 0.1
evals_per_epoch: 4
saves_per_epoch: 1
```

Run:
```bash
axolotl train config.yaml
```

## Dataset Formats

### DPO / IPO / SimPO

All paired-preference methods share the same dataset strategies. The transform must produce three fields: `prompt`, `chosen`, `rejected`.

#### chat_template.default (recommended)

The most flexible format. Uses the model's chat template. Supports custom field names and role mappings.

Config:
```yaml
rl: dpo
chat_template: llama3  # or qwen_25, chatml, etc.
datasets:
  - path: my_dataset
    type: chat_template.default
    field_messages: messages       # field containing conversation history
    field_chosen: chosen           # field containing chosen response
    field_rejected: rejected       # field containing rejected response
    message_property_mappings:
      role: role
      content: content
    roles:
      user: [user, human]          # map source roles to standard roles
      assistant: [assistant, gpt]
      system: [system]
```

Dataset JSON:
```json
{
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "chosen": {"role": "assistant", "content": "4"},
    "rejected": {"role": "assistant", "content": "I think it might be 5."}
}
```

The `chosen`/`rejected` fields accept: a string, a dict with role/content, or a list of messages (last message is used).

#### chat_template.argilla_chat

For datasets where chosen/rejected contain full conversation threads (not just the final response).

Dataset JSON:
```json
{
    "chosen": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ],
    "rejected": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Maybe 5?"}
    ]
}
```

#### chatml.default

Auto-detects prompt field name (`prompt`, `input`, `question`, or `instruction`) and response field names (`chosen`/`chosen_response`, `rejected`/`rejected_response`). Hardcodes chatml tokens.

Dataset JSON:
```json
{
    "system": "...",
    "question": "...",
    "chosen": "...",
    "rejected": "..."
}
```

#### chatml variants

| Type | Prompt Field | Chosen Field | Rejected Field |
|------|-------------|-------------|---------------|
| `chatml.default` | auto-detect | `chosen` or `chosen_response` | `rejected` or `rejected_response` |
| `chatml.intel` | `question` | `chosen` | `rejected` |
| `chatml.icr` | `input` | `chosen` | `rejected` |
| `chatml.argilla_chat` | `chosen[0].content` | `chosen[1].content` | `rejected[1].content` |
| `chatml.prompt_pairs` | `prompt` | `chosen` | `rejected` |
| `chatml.ultra` | `prompt` | `chosen[1].content` | `rejected[1].content` |

#### llama3 variants

Same field mappings as chatml variants, but uses Llama 3 special tokens. Available: `llama3.default`, `llama3.intel`, `llama3.icr`, `llama3.argilla_chat`, `llama3.prompt_pairs`, `llama3.ultra`.

#### user_defined.default

For fully custom field names and format strings. The `type` is a dict instead of a string:

```yaml
rl: dpo
datasets:
  - path: my_dataset
    type:
      field_prompt: "prompt"
      field_system: "system"
      field_chosen: "chosen"
      field_rejected: "rejected"
      prompt_format: "{system}\n{prompt}"
      chosen_format: "{chosen}"
      rejected_format: "{rejected}"
```

#### passthrough.default

Zero processing. Dataset must already contain `prompt`, `chosen`, `rejected` as pre-formatted strings.

```yaml
datasets:
  - path: my_dataset
    type: passthrough.default
```

### KTO

KTO uses unpaired data. Each sample has a completion and a binary label (True = desirable, False = undesirable). The transform must produce: `prompt`, `completion`, `label`.

#### KTO chatml/llama3 variants

| Type | Prompt Field | Completion Field | Label Field |
|------|-------------|-----------------|------------|
| `chatml.argilla` | `instruction` | `completion` | from dataset |
| `chatml.argilla_chat` | `chosen[0].content` | `completion[1].content` | from dataset |
| `chatml.intel` | `question` | `completion` | from dataset |
| `chatml.prompt_pairs` | `prompt` | `completion` | from dataset |
| `chatml.ultra` | `prompt` | `completion` | from dataset |
| `llama3.argilla` | `instruction` | `completion` | from dataset |
| `llama3.argilla_chat` | `completion[0].content` | `completion[1].content` | from dataset |
| `llama3.intel` | `question` | `completion` | from dataset |
| `llama3.prompt_pairs` | `prompt` | `completion` | from dataset |
| `llama3.ultra` | `prompt` | `completion` | from dataset |

KTO dataset JSON (e.g., `chatml.intel`):
```json
{
    "system": "...",
    "question": "What is the capital of France?",
    "completion": "Paris is the capital of France.",
    "label": true
}
```

#### user_defined.default (KTO)

```yaml
rl: kto
datasets:
  - path: my_dataset
    type:
      field_prompt: "prompt"
      field_system: "system"
      field_completion: "completion"
      field_label: "label"
      prompt_format: "{prompt}"
      completion_format: "{completion}"
```

### ORPO

ORPO uses its own tokenization strategy (`chat_template.argilla`) that produces `input_ids`, `labels`, `rejected_input_ids`, `rejected_labels`, and attention masks directly. It requires `chat_template` to be set.

#### chat_template.argilla (ORPO)

```yaml
rl: orpo
chat_template: chatml
datasets:
  - path: argilla/ultrafeedback-binarized-preferences-cleaned
    type: chat_template.argilla
```

Dataset JSON:
```json
{
    "system": "...",
    "chosen": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "rejected": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

If a `prompt` field exists, it overrides the user message extracted from `chosen`.

## YAML Config Templates

### DPO

```yaml
base_model: meta-llama/Meta-Llama-3-8B-Instruct

chat_template: llama3
rl: dpo

# Optional DPO-specific parameters
# rl_beta: 0.1                  # KL penalty strength (default: 0.1)
# dpo_label_smoothing: 0.0      # Label smoothing (incompatible with IPO)
# dpo_use_weighting: false       # Use WPO-style weighting
# dpo_norm_loss: false           # Normalize log-probs like IPO
# dpo_use_liger_kernel: false    # Fused DPO loss kernel
# dpo_padding_free: false        # Padding-free training
# rpo_alpha: null                # NLL regularization term (RPO paper)

datasets:
  - path: fozziethebeat/alpaca_messages_2k_dpo_test
    type: chat_template.default
    field_messages: conversation
    field_chosen: chosen
    field_rejected: rejected
    message_property_mappings:
      role: role
      content: content
    roles:
      system: [system]
      user: [user]
      assistant: [assistant]

adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

sequence_len: 4096
sample_packing: false
gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002
warmup_ratio: 0.1

bf16: auto
gradient_checkpointing: true
flash_attention: true
output_dir: ./outputs/dpo-out
```

### IPO

Identical to DPO except:
```yaml
rl: ipo
# dpo_label_smoothing is NOT compatible with IPO
# rl_beta controls regularization strength
```

IPO uses the same dataset formats as DPO. Internally, it sets `loss_type: "ipo"` on the DPO trainer.

### KTO

```yaml
base_model: meta-llama/Llama-3.2-1B

rl: kto
rl_beta: 0.5                     # KL penalty (default: 0.1)
kto_desirable_weight: 0.2        # Weight for desirable examples (default: 1.0)
kto_undesirable_weight: 1.0      # Weight for undesirable examples (default: 1.0)

remove_unused_columns: false      # Required for KTO

datasets:
  - path: argilla/ultrafeedback-binarized-preferences-cleaned-kto
    type: llama3.ultra
    split: train

adapter: qlora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_linear: true

sequence_len: 2048
sample_packing: false             # Not supported with KTO
pad_to_sequence_len: false

gradient_accumulation_steps: 1
micro_batch_size: 2
num_epochs: 1
optimizer: adamw_8bit
lr_scheduler: cosine
learning_rate: 0.0002
warmup_ratio: 0.1

bf16: auto
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: true             # Recommended for KTO
flash_attention: true
output_dir: ./outputs/kto-out

special_tokens:
  pad_token: "<|end_of_text|>"
```

### ORPO

```yaml
base_model: mistralai/Mistral-7B-v0.1

rl: orpo
orpo_alpha: 0.1                   # Ratio loss weight (mapped to TRL beta)
remove_unused_columns: false       # Required for ORPO

chat_template: chatml
datasets:
  - path: argilla/ultrafeedback-binarized-preferences-cleaned
    type: chat_template.argilla

adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

sequence_len: 4096
sample_packing: false

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 1
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002
warmup_ratio: 0.1

bf16: auto
gradient_checkpointing: true
flash_attention: true
output_dir: ./outputs/orpo-out
```

### SimPO

```yaml
base_model: meta-llama/Meta-Llama-3-8B-Instruct

rl: simpo
simpo_gamma: 1.0                  # Target reward margin
cpo_alpha: 0.0                    # BC regularizer weight (optional)
# rl_beta: 2.0                    # KL penalty strength

# SimPO uses the same dataset formats as DPO
chat_template: llama3
datasets:
  - path: my_preference_dataset
    type: chat_template.default
    field_messages: messages
    field_chosen: chosen
    field_rejected: rejected

sequence_len: 4096
sample_packing: false
# ... standard training params ...
output_dir: ./outputs/simpo-out
```

SimPO internally uses `AxolotlCPOConfig` with `loss_type: "simpo"`. It does not load a reference model.

## Key Parameters

### Common to All Methods

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rl` | (required) | One of: `dpo`, `ipo`, `kto`, `orpo`, `simpo` |
| `rl_beta` | 0.1 | KL penalty coefficient. Higher = more conservative updates |
| `sequence_len` | -- | Max sequence length for chosen + rejected |
| `sample_packing` | -- | Must be `false` for all preference methods |
| `remove_unused_columns` | `false` | Set explicitly to `false` for KTO and ORPO |

### DPO / IPO Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dpo_label_smoothing` | 0.0 | Smooths preference labels. DPO only (not IPO) |
| `dpo_use_weighting` | false | WPO-style weighting of preference pairs |
| `dpo_norm_loss` | false | Normalize log-probs (IPO-style normalization for DPO loss) |
| `dpo_use_liger_kernel` | false | Use Liger fused kernel for DPO loss |
| `dpo_padding_free` | false | Padding-free training |
| `rpo_alpha` | null | NLL regularization weight from RPO paper |

### KTO Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kto_desirable_weight` | 1.0 | Weight for desirable (label=true) examples |
| `kto_undesirable_weight` | 1.0 | Weight for undesirable (label=false) examples |

### ORPO Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `orpo_alpha` | -- | Ratio loss weight. Mapped internally to TRL's `beta` parameter |

### SimPO Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simpo_gamma` | -- | Target reward margin between chosen and rejected |
| `cpo_alpha` | null | Weight of the behavior cloning (BC) regularizer |

## Method Selection Guide

| | DPO | IPO | KTO | ORPO | SimPO |
|---|---|---|---|---|---|
| **Data format** | Paired | Paired | Unpaired | Paired | Paired |
| **Reference model** | Yes | Yes | Yes | No | No |
| **VRAM overhead** | ~2x model | ~2x model | ~2x model | ~1x model | ~1x model |
| **Compute cost** | Medium | Medium | Medium | Low | Low |
| **Overfitting risk** | Higher | Lower | Lower | Medium | Medium |
| **Sample packing** | No | No | No | No | No |
| **Dataset strategies** | Many | Same as DPO | KTO-specific | chat_template.argilla | Same as DPO |
| **TRL trainer class** | DPOTrainer | DPOTrainer | KTOTrainer | ORPOTrainer | CPOTrainer |

Decision tree:
1. Do you have paired preference data (chosen + rejected)? -> DPO (default), IPO (if overfitting), ORPO (if VRAM-limited), SimPO (if length-sensitive)
2. Do you only have binary labels (good/bad) per completion? -> KTO
3. Do you want single-stage training (no separate SFT step)? -> ORPO

## Running

### Single GPU

```bash
axolotl train config.yaml
```

### Multi-GPU (accelerate)

```bash
accelerate launch -m axolotl.cli.train config.yaml
```

### With LoRA + QLoRA

Add to config:
```yaml
adapter: qlora          # or lora
load_in_4bit: true      # only for qlora
lora_r: 32
lora_alpha: 16          # or 2x lora_r
lora_target_linear: true
```

## Healthy Training Indicators

| Metric | Healthy Range | Problem |
|--------|--------------|---------|
| `train/loss` | Decreasing, 0.3-0.7 | Flat or increasing = broken data or too high LR |
| `rewards/chosen` | Increasing over time | Flat = model not learning preferences |
| `rewards/rejected` | Decreasing over time | Increasing = model prefers wrong responses |
| `rewards/margins` | Positive and increasing | Negative = model prefers rejected over chosen |
| `rewards/accuracies` | > 0.5, increasing toward 0.7+ | < 0.5 = worse than random |
| `logps/chosen` | Relatively stable | Collapsing = mode collapse |
| `logps/rejected` | Decreasing | Increasing = reward hacking |
| `grad_norm` | 0.01 - 10.0 | > 100 = exploding gradients, 0 = dead training |

### Method-Specific Signals

- **DPO/IPO**: Watch `rewards/margins`. Should be positive and growing. If it plateaus early, try increasing `rl_beta`.
- **KTO**: Watch loss convergence. KTO loss can be noisier than DPO because samples are unpaired.
- **ORPO**: Monitor both the SFT loss component and the odds ratio. The combined loss should decrease.
- **SimPO**: Check that the length-normalized rewards show clear separation between chosen and rejected.

## Known Issues & Fixes

| Issue | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| Sample packing | Crash or wrong results | Not supported with any preference method | Set `sample_packing: false` |
| KTO missing label | `KeyError: 'label'` | Dataset lacks boolean label field | Ensure dataset has `label` column (True/False) |
| ORPO missing columns | `KeyError` during tokenization | `remove_unused_columns` not set | Add `remove_unused_columns: false` |
| KTO missing columns | `KeyError` during tokenization | `remove_unused_columns` not set | Add `remove_unused_columns: false` |
| ORPO chat_template | Template not applied | `chat_template` not set in config | ORPO requires explicit `chat_template` setting |
| Wrong field names | Empty prompt/chosen/rejected | Dataset fields do not match strategy expectations | Check the field mapping table for your chosen `type` |
| OOM with ref model | CUDA OOM | DPO/IPO/KTO loads two copies of the model | Use LoRA/QLoRA, or switch to ORPO/SimPO (no ref model) |
| IPO + label_smoothing | Unexpected results | `dpo_label_smoothing` is not compatible with IPO | Do not set `dpo_label_smoothing` when `rl: ipo` |
| orpo_alpha mapping | Confusion about beta | `orpo_alpha` is mapped to TRL's `beta` parameter internally | This is expected TRL behavior |

## File Map

```
axolotl/
├── src/axolotl/
│   ├── core/trainers/dpo/
│   │   ├── __init__.py           # DPOStrategy — set_training_args_kwargs, trainer/config selection
│   │   ├── args.py               # AxolotlDPOConfig (extends TRL DPOConfig)
│   │   └── trainer.py            # AxolotlDPOTrainer (extends TRL DPOTrainer)
│   ├── core/builders/rl.py       # HFRLTrainerBuilder — routes rl type to trainer class
│   ├── core/training_args.py     # AxolotlKTOConfig, AxolotlORPOConfig, AxolotlCPOConfig
│   ├── prompt_strategies/
│   │   ├── dpo/
│   │   │   ├── __init__.py       # Loader (delegates to submodules)
│   │   │   ├── chat_template.py  # chat_template.default, chat_template.argilla_chat
│   │   │   ├── chatml.py         # chatml.default/intel/icr/argilla_chat/prompt_pairs/ultra
│   │   │   ├── llama3.py         # llama3.default/intel/icr/argilla_chat/prompt_pairs/ultra
│   │   │   ├── user_defined.py   # user_defined.default (custom field mapping)
│   │   │   ├── passthrough.py    # passthrough.default (no transform)
│   │   │   └── zephyr.py         # zephyr.nectar (ranked answers)
│   │   ├── kto/
│   │   │   ├── __init__.py       # Loader
│   │   │   ├── chatml.py         # chatml.argilla/argilla_chat/intel/prompt_pairs/ultra
│   │   │   ├── llama3.py         # llama3.argilla/argilla_chat/intel/prompt_pairs/ultra
│   │   │   └── user_defined.py   # user_defined.default (custom field mapping)
│   │   └── orpo/
│   │       ├── __init__.py       # Loader
│   │       └── chat_template.py  # chat_template.argilla (tokenizing strategy + transform)
│   └── utils/schemas/
│       ├── enums.py              # RLType enum (dpo, ipo, kto, orpo, simpo, grpo, gdpo, ebft)
│       └── config.py             # All rl/dpo/kto/orpo/simpo config fields
│
├── examples/
│   ├── qwen2/dpo.yaml                           # Qwen2.5-0.5B DPO with chat_template
│   ├── llama-3/instruct-dpo-lora-8b.yml         # Llama-3-8B DPO LoRA with chat_template
│   ├── llama-3/lora-1b-deduplicate-dpo.yml       # Llama-3.2-1B DPO with deduplication
│   ├── llama-3/qlora-1b-kto.yaml                 # Llama-3.2-1B KTO QLoRA
│   ├── llama-3/qlora-1b-gdpo.yaml                # Llama-3.2-1B GDPO (online, not preference)
│   ├── mistral/dpo/mistral-dpo-qlora.yml          # Mistral-7B DPO QLoRA
│   ├── mistral/orpo/mistral-qlora-orpo.yml        # Mistral-7B ORPO QLoRA
│   └── swanlab/dpo-swanlab-completions.yml        # DPO with SwanLab logging
│
└── docs/rlhf.qmd                # User-facing docs for all RLHF methods
```

## Prompt Strategy Resolution

The `type` field in the dataset config resolves to a Python function via this path:

```
type: "chatml.intel"
  -> axolotl.prompt_strategies.dpo.chatml.intel(cfg, **kwargs)
  -> returns transform_fn(sample) -> {"prompt": ..., "chosen": ..., "rejected": ...}

type: "chat_template.default"
  -> axolotl.prompt_strategies.dpo.chat_template.default(cfg, dataset_idx, **kwargs)
  -> returns (transform_fn(sample, tokenizer), {"remove_columns": [...]})

type: {"field_prompt": "prompt", ...}   (dict type)
  -> axolotl.prompt_strategies.dpo.user_defined.default(cfg, dataset_idx, **kwargs)
  -> returns transform_fn(sample) -> {"prompt": ..., "chosen": ..., "rejected": ...}
```

For KTO, replace `dpo` with `kto` in the module path. For ORPO, replace with `orpo`.

The method-specific module is selected based on the `rl` config value. The strategy module base is: `axolotl.prompt_strategies.{rl_method}`.
