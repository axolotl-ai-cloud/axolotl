# Supervised Fine-Tuning (SFT) — Agent Reference

Reference for AI agents working with axolotl's supervised fine-tuning pipeline. Describes the training flow, configuration, dataset formats, optimizations, and known issues.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  YAML Config                                                     │
│  - Model, adapter, dataset, hyperparameters, optimizations       │
└──────────────┬───────────────────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────────────┐
│  axolotl train config.yaml                                       │
│                                                                  │
│  1. Load base model (+ quantization if QLoRA/8-bit)              │
│  2. Apply adapter layers (LoRA/QLoRA) if configured              │
│  3. Load + tokenize dataset(s)                                   │
│     - Apply prompt template (chat_template / alpaca / custom)    │
│     - Mask inputs (train_on_inputs: false)                       │
│     - Pack samples into sequences (sample_packing: true)         │
│  4. Training loop (HuggingFace Trainer)                          │
│     ┌─────────────────────────────────┐                          │
│     │  For each step:                 │                          │
│     │    forward pass → loss          │                          │
│     │    backward pass → gradients    │                          │
│     │    optimizer step               │                          │
│     │    lr scheduler step            │                          │
│     └─────────────────────────────────┘                          │
│  5. Save model / adapter weights + tokenizer                     │
└──────────────────────────────────────────────────────────────────┘

Single-GPU:
  ┌─────────────────────────────────────┐
  │  GPU: Model + Training              │
  │  - Forward/backward pass            │
  │  - Gradient accumulation            │
  │  - Optimizer step                   │
  └─────────────────────────────────────┘

Multi-GPU (FSDP / DeepSpeed):
  ┌────────────────┐  ┌────────────────┐
  │  GPU 0: Shard 0│  │  GPU 1: Shard 1│  ...
  │  - Partial fwd  │  │  - Partial fwd  │
  │  - All-gather   │  │  - All-gather   │
  │  - Reduce-scatter│  │  - Reduce-scatter│
  └────────────────┘  └────────────────┘
```

## Components Required

An SFT training run requires:

1. **A YAML config** — model, dataset(s), adapter settings, training hyperparameters
2. **A dataset** — HuggingFace Hub, local JSONL/JSON/Parquet, or S3/GCS path
3. **(Optional) A custom prompt strategy** — for non-standard dataset formats

No external server processes are needed (unlike GRPO which requires vLLM).

## Quick Start Checklist

1. **Install axolotl** (see `docs/installation.qmd`)
2. **Choose a base model** from HuggingFace (e.g., `meta-llama/Llama-3.1-8B`, `Qwen/Qwen2.5-7B-Instruct`)
3. **Choose an adapter strategy**: LoRA (most common), QLoRA (lower VRAM), or full fine-tune
4. **Prepare your dataset** in a supported format (see Dataset Format Selection below)
5. **Write a YAML config** (see template below)
6. **Validate the config**: `axolotl preprocess config.yaml --debug` — inspect tokenized samples and label masking
7. **Run training**: `axolotl train config.yaml`
8. **Test the model**: `axolotl inference config.yaml --lora-model-dir ./outputs/lora-out`
9. **(If LoRA) Merge weights**: `axolotl merge-lora config.yaml --lora-model-dir ./outputs/lora-out`

## YAML Config Template

### LoRA (recommended starting point)

```yaml
# ---- Model ----
base_model: NousResearch/Meta-Llama-3.1-8B-Instruct

# ---- Adapter ----
adapter: lora
lora_r: 32
lora_alpha: 64                        # Typically 1-2x lora_r
lora_dropout: 0.05
lora_target_linear: true              # Target all linear layers

# ---- Dataset ----
datasets:
  - path: my_org/my_dataset           # HuggingFace Hub or local path
    type: chat_template               # See "Dataset Format Selection"
    split: train

val_set_size: 0.05                    # 5% held out for eval
dataset_prepared_path: last_run_prepared
output_dir: ./outputs/lora-out

# ---- Sequence ----
sequence_len: 4096
sample_packing: true
eval_sample_packing: true
pad_to_sequence_len: true

# ---- Training ----
num_epochs: 1
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.1
optimizer: adamw_8bit
weight_decay: 0.0
train_on_inputs: false                # Mask human/system turns

# ---- Precision ----
bf16: auto
tf32: false

# ---- Optimizations ----
flash_attention: true
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

# ---- Logging & Saving ----
logging_steps: 1
evals_per_epoch: 4
saves_per_epoch: 1
loss_watchdog_threshold: 5.0
loss_watchdog_patience: 3

# ---- Tokens ----
special_tokens:
  pad_token: "<|end_of_text|>"

# ---- Tracking (optional) ----
wandb_project:
wandb_entity:
wandb_name:
```

### QLoRA (low VRAM)

Replace the adapter section:

```yaml
load_in_4bit: true
adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
```

And use a compatible optimizer:

```yaml
optimizer: adamw_bnb_8bit
```

### Full Fine-Tune

Remove all adapter and quantization lines. Adjust learning rate:

```yaml
# No adapter, load_in_8bit, or load_in_4bit lines
learning_rate: 2e-5                   # Lower LR for full fine-tune
optimizer: adamw_torch_fused
```

## Dataset Format Selection

### Decision Guide

```
Is your data in chat/message format?
  ├─ YES: Does it use OpenAI message format (role/content)?
  │   ├─ YES ──────────────────────> type: chat_template  (recommended)
  │   └─ NO (custom field names) ──> type: chat_template + message_property_mappings
  │
  └─ NO: Is it instruction/response pairs?
      ├─ YES ──> type: alpaca       (instruction, input, output)
      └─ NO: Is it raw text?
          ├─ YES with segments ─────> type: input_output  (template-free, fine-grained masking)
          └─ YES continuous ────────> type: completion     (pretraining-style, no masking)
```

### chat_template (recommended default)

The recommended format for instruction/chat fine-tuning. Uses the model's native chat template (or a specified one) to format messages correctly.

**Data format:**
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Config:**
```yaml
datasets:
  - path: my_dataset
    type: chat_template
    # Optional overrides:
    # chat_template: chatml           # Override tokenizer's template
    # roles_to_train: ["assistant"]   # Which roles to compute loss on (default)
    # train_on_eos: turn              # Train on EOS: "all", "turn" (default), "last"
```

**Non-standard field names** (e.g., ShareGPT-style `conversations` with `from`/`value`):
```yaml
datasets:
  - path: my_dataset
    type: chat_template
    field_messages: conversations
    message_property_mappings:
      role: from
      content: value
```

### alpaca

Simple instruction tuning. Good for single-turn tasks.

**Data format:**
```json
{"instruction": "Classify the sentiment", "input": "I love this!", "output": "positive"}
```

**Config:**
```yaml
datasets:
  - path: my_dataset.jsonl
    type: alpaca
```

### input_output (template-free)

Full control over tokenization and masking. No templates applied — you construct the exact text.

**Data format:**
```json
{"segments": [{"label": false, "text": "<s>Question: ...\n"}, {"label": true, "text": "Answer: ...</s>"}]}
```

**Config:**
```yaml
datasets:
  - path: my_dataset.jsonl
    type: input_output
train_on_inputs: false
```

### completion

Pretraining-style. The entire text is trained on with no masking.

**Data format:**
```json
{"text": "The complete document text goes here."}
```

**Config:**
```yaml
datasets:
  - path: my_dataset
    type: completion
    # field: text                     # Column name (default: "text")
```

### Custom Prompt Format

For datasets with non-standard field names that fit the instruction pattern:

```yaml
datasets:
  - path: my_dataset
    type:
      system_prompt: ""
      field_instruction: question
      field_output: answer
      format: "[INST] {instruction} [/INST]"
      no_input_format: "[INST] {instruction} [/INST]"
```

## Key Config Decisions

### Model Size to Adapter Choice

| Model Size | LoRA | QLoRA (4-bit) | Full Fine-Tune | VRAM (approx) |
|-----------|------|---------------|----------------|---------------|
| 1-3B | Preferred | Low-budget option | Single GPU OK | 8-16 GB (LoRA) |
| 7-8B | Preferred | Good balance | Needs multi-GPU | 16-24 GB (LoRA) |
| 13-14B | Preferred | Good balance | Multi-GPU required | 24-40 GB (LoRA) |
| 30-70B | LoRA or QLoRA | Preferred for single GPU | Multi-node | 40-80 GB (QLoRA) |

### Hyperparameter Guidance

| Parameter | LoRA | QLoRA | Full FT | Notes |
|-----------|------|-------|---------|-------|
| `learning_rate` | 1e-4 to 3e-4 | 1e-4 to 3e-4 | 1e-5 to 5e-5 | Lower for larger models |
| `lora_r` | 16-64 | 16-64 | N/A | 32 is a safe default |
| `lora_alpha` | 1-2x `lora_r` | 1-2x `lora_r` | N/A | Scales the adapter contribution |
| `micro_batch_size` | 2-8 | 2-4 | 1-2 | Reduce if OOM |
| `gradient_accumulation_steps` | 2-8 | 4-16 | 4-16 | Effective batch = micro_batch * grad_accum * num_gpus |
| `num_epochs` | 1-3 | 1-3 | 1-3 | Watch for overfitting; 1 epoch often sufficient |
| `sequence_len` | 2048-8192 | 2048-4096 | 2048-8192 | Match model's context window; longer = more VRAM |
| `warmup_ratio` | 0.05-0.1 | 0.05-0.1 | 0.05-0.1 | Fraction of steps for LR warmup |
| `weight_decay` | 0.0 | 0.0 | 0.01-0.1 | Typically only for full FT |
| `optimizer` | `adamw_8bit` | `adamw_bnb_8bit` | `adamw_torch_fused` | 8-bit saves VRAM |

## Optimizations

### flash_attention

Fused attention kernel. Faster and more memory-efficient. Requires Ampere+ GPU (A100, H100, RTX 30xx+).

```yaml
flash_attention: true
```

### gradient_checkpointing

Trades compute for VRAM by recomputing activations during backward pass instead of storing them. Increases training time by ~20-30% but significantly reduces memory.

```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false                # Recommended; `true` required for DeepSpeed ZeRO-3
```

### sample_packing

Packs multiple short samples into a single sequence to avoid wasting compute on padding tokens. Significant speedup when samples are much shorter than `sequence_len`.

```yaml
sample_packing: true
eval_sample_packing: true
pad_to_sequence_len: true             # Reduces memory fragmentation
```

Requires `flash_attention: true` or another compatible attention implementation. Axolotl uses loss masking to ensure cross-sample attention does not leak.

### bf16

Use bfloat16 mixed precision. Recommended on Ampere+ GPUs. Avoids the overflow issues of fp16.

```yaml
bf16: auto                            # Auto-detects GPU capability
```

### Liger Kernel

Fused kernels for RMSNorm, RoPE, SwiGLU, and cross-entropy. Reduces VRAM and improves throughput. Loaded as a plugin.

```yaml
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rope: true
liger_rms_norm: true
liger_glu_activation: true
liger_fused_linear_cross_entropy: true
```

### Batch Flattening

For cases where `sample_packing` is not used, batch flattening concatenates variable-length samples in a batch and processes them without padding.

```yaml
batch_flattening: true                # Alternative to sample_packing
```

## Multi-GPU

### FSDP (Fully Sharded Data Parallel)

Shards model parameters, gradients, and optimizer states across GPUs.

```yaml
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_limit_all_gathers: true
  fsdp_sync_module_states: true
  fsdp_offload_params: false          # true to offload to CPU (slower, saves GPU VRAM)
  fsdp_use_orig_params: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer  # Match your model architecture
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sharding_strategy: FULL_SHARD
```

Launch:
```bash
axolotl train config.yaml
```

For FSDP2 (newer, simpler config):
```yaml
fsdp_version: 2
fsdp:
  - full_shard
```

### DeepSpeed ZeRO

Alternative to FSDP. ZeRO-2 for moderate sharding, ZeRO-3 for full sharding.

```yaml
deepspeed: deepspeed_configs/zero3_bf16.json
```

Launch:
```bash
axolotl train config.yaml
```

Key notes:
- ZeRO-3 requires `use_reentrant: true` for gradient checkpointing
- May require `CUDA_HOME` to be set if nvcc is not in PATH

## Running

### Train

```bash
axolotl train config.yaml
```

Or multi-GPU:
```bash
axolotl train config.yaml
```

### Preprocess (validate data before training)

```bash
axolotl preprocess config.yaml --debug
```

Tokenizes the dataset and shows label masking. Check that:
- Input tokens have label `-100` (masked) when `train_on_inputs: false`
- Output/assistant tokens have their actual token ID as the label
- Special tokens (BOS, EOS, pad) are placed correctly

### Inference

```bash
# CLI interactive mode
axolotl inference config.yaml --lora-model-dir ./outputs/lora-out

# Gradio UI
axolotl inference config.yaml --lora-model-dir ./outputs/lora-out --gradio
```

### Merge LoRA

```bash
axolotl merge-lora config.yaml --lora-model-dir ./outputs/lora-out
```

Output saved to `{output_dir}/merged/`.

### Fetch Example Configs

```bash
axolotl fetch examples
```

Downloads all example configs to `./examples/`.

## Healthy Training Indicators

| Metric | Healthy | Problem |
|--------|---------|---------|
| `train_loss` | Decreasing, starting ~2-4 for chat models | Flat or increasing from step 1 — data or LR issue |
| `eval_loss` | Decreasing, tracks train_loss | Increasing while train_loss decreases — overfitting |
| `learning_rate` | Follows scheduler curve (warmup then decay) | Flat or NaN — config issue |
| `grad_norm` | 0.1-10, relatively stable | Spikes >100 — instability. 0.0 — frozen weights |
| `train_samples_per_second` | Stable after first few steps | Dropping over time — possible memory pressure |
| `epoch` | Increases proportionally | Stuck — data pipeline issue |

### What to Watch For

- **Loss spikes**: Occasional small spikes are normal. Sustained increase suggests LR too high or data quality issues.
- **Loss goes to 0 quickly**: Likely overfitting on a very small dataset. Reduce epochs or increase data.
- **Loss never decreases**: Check that `train_on_inputs: false` is set correctly, the dataset is not empty after filtering, and the learning rate is not too low.
- **OOM after some steps**: Enable `gradient_checkpointing`, reduce `micro_batch_size`, reduce `sequence_len`, or use QLoRA.
- **eval_loss diverges from train_loss**: Overfitting. Reduce `num_epochs`, increase `val_set_size`, or add regularization (LoRA dropout, weight decay).

## Known Issues & Fixes

| Issue | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| OOM during training | CUDA out of memory | Batch too large or sequence too long | Reduce `micro_batch_size`, enable `gradient_checkpointing`, reduce `sequence_len` |
| OOM during preprocessing | RAM exhaustion | Large dataset tokenization | Set `dataset_prepared_path` to cache, use `preprocess_shards`, reduce `dataset_num_proc` |
| `sample_packing` + SDPA + bf16 | 0.0 loss | Known incompatibility | Use `flash_attention: true` or disable `sample_packing` |
| eval errors with packing | Shape mismatch during eval | `eval_sample_packing` mismatch | Set `eval_sample_packing: false` if errors occur |
| Missing chat template | `chat_template is null` error | Tokenizer has no default template | Set `chat_template: chatml` (or another template) explicitly |
| Label masking wrong | Model trains on prompts or ignores responses | Incorrect `type` or field mapping | Run `axolotl preprocess config.yaml --debug` and inspect labels |
| LoRA target mismatch | Warning about missing modules | Model architecture changed | Use `lora_target_linear: true` instead of listing specific modules |
| Loss NaN | NaN in training loss | LR too high, fp16 overflow, or data issues | Use `bf16: auto`, lower LR, check data for empty samples |
| Tokenizer pad token | Infinite loss or garbage output | Model has no pad token | Set `special_tokens: pad_token: "<|end_of_text|>"` or equivalent unused token |
| Loss watchdog triggered | Training stops early | Loss exceeded threshold | Increase `loss_watchdog_threshold` or fix underlying data/LR issue |
| FSDP save errors | Checkpoint save hangs or fails | State dict type mismatch | Use `fsdp_state_dict_type: FULL_STATE_DICT` |
| DeepSpeed gradient checkpoint | `CheckpointError` tensor mismatch | Non-reentrant checkpointing | Set `use_reentrant: true` in `gradient_checkpointing_kwargs` |
| Slow first step | Step 1 takes much longer | Compilation, caching, data loading | Normal. Subsequent steps will be faster. |

## File Map

```
axolotl/
├── src/axolotl/
│   ├── cli/
│   │   ├── train.py                 # Entry point for `axolotl train`
│   │   ├── preprocess.py            # Entry point for `axolotl preprocess`
│   │   ├── inference.py             # Entry point for `axolotl inference`
│   │   └── merge_lora.py            # Entry point for `axolotl merge-lora`
│   │
│   ├── core/
│   │   ├── builders/
│   │   │   ├── causal.py            # HFCausalTrainerBuilder — wires config → SFT trainer
│   │   │   └── base.py              # Base trainer builder
│   │   └── trainers/
│   │       ├── base.py              # AxolotlTrainer — base trainer class
│   │       ├── mixins/
│   │       │   ├── packing.py       # Sample packing logic
│   │       │   ├── optimizer.py     # Optimizer setup
│   │       │   ├── scheduler.py     # LR scheduler setup
│   │       │   └── checkpoints.py   # Checkpoint save/load
│   │       └── utils.py             # Trainer utility functions
│   │
│   ├── utils/
│   │   ├── schemas/
│   │   │   ├── config.py            # AxolotlInputConfig — main config schema
│   │   │   ├── datasets.py          # SFTDataset, DatasetConfig — dataset config
│   │   │   ├── training.py          # HyperparametersConfig — LR, batch size, etc.
│   │   │   ├── model.py             # ModelInputConfig — base_model, dtype, etc.
│   │   │   ├── peft.py              # LoraConfig — LoRA parameters
│   │   │   └── fsdp.py              # FSDPConfig — FSDP settings
│   │   ├── callbacks/               # Training callbacks (logging, profiling, etc.)
│   │   └── data/                    # Data loading and processing utilities
│   │
│   ├── prompt_strategies/           # Prompt formatting strategies (chat_template, alpaca, etc.)
│   ├── monkeypatch/                 # Runtime patches for HF transformers
│   └── integrations/
│       └── liger/                   # Liger kernel integration (plugin)
│
├── examples/
│   ├── llama-3/
│   │   ├── lora-1b.yml              # LoRA 1B example
│   │   ├── qlora-1b.yml             # QLoRA 1B example
│   │   ├── fft-8b.yaml              # Full fine-tune 8B example
│   │   ├── instruct-lora-8b.yml     # Instruct LoRA 8B (chat_template)
│   │   └── fft-8b-liger-fsdp.yaml   # Full FT + Liger + FSDP
│   └── ...                          # More model-specific examples
│
├── deepspeed_configs/               # DeepSpeed JSON configs (zero2, zero3)
└── docs/
    ├── getting-started.qmd          # Quickstart guide
    ├── config-reference.qmd         # Full config reference
    ├── dataset-formats/
    │   ├── conversation.qmd         # chat_template format docs
    │   ├── inst_tune.qmd            # Instruction tuning formats (alpaca, etc.)
    │   └── template_free.qmd        # input_output format docs
    └── multi-gpu.qmd               # Multi-GPU training guide
```
