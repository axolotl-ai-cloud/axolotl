# SFT — Agent Reference

Supervised fine-tuning pipeline reference. For config templates and dataset format examples, see [getting-started.qmd](../getting-started.qmd) and [dataset-formats/](../dataset-formats/).

## Architecture

```
YAML Config → axolotl train config.yaml

  1. Load base model (+ quantization if QLoRA/8-bit)
  2. Apply adapter layers (LoRA/QLoRA) if configured
  3. Load + tokenize dataset(s)
     - Apply prompt template (chat_template / alpaca / custom)
     - Mask inputs (train_on_inputs: false)
     - Pack samples into sequences (sample_packing: true)
  4. Training loop (HuggingFace Trainer)
     - forward → loss → backward → optimizer step → lr scheduler step
  5. Save model / adapter weights + tokenizer

Multi-GPU: FSDP or DeepSpeed shards model across GPUs automatically.
```

## Components Required

1. A YAML config — model, dataset(s), adapter settings, hyperparameters
2. A dataset — HuggingFace Hub, local JSONL/JSON/Parquet, or S3/GCS path
3. (Optional) A custom prompt strategy — for non-standard dataset formats

No external server processes needed (unlike GRPO which requires vLLM).

## Dataset Format Decision Tree

```
Is your data in chat/message format?
  ├─ YES: OpenAI message format (role/content)?
  │   ├─ YES ──────────────────────> type: chat_template  (recommended)
  │   └─ NO (custom field names) ──> type: chat_template + message_property_mappings
  └─ NO: Instruction/response pairs?
      ├─ YES ──> type: alpaca       (instruction, input, output)
      └─ NO: Raw text?
          ├─ YES with segments ─────> type: input_output  (template-free masking)
          └─ YES continuous ────────> type: completion     (pretraining-style)
```

Full format specs: [dataset-formats/](../dataset-formats/)

## Model Size to Adapter Choice

| Model Size | LoRA | QLoRA (4-bit) | Full Fine-Tune | VRAM (approx) |
|-----------|------|---------------|----------------|---------------|
| 1-3B | Preferred | Low-budget option | Single GPU OK | 8-16 GB (LoRA) |
| 7-8B | Preferred | Good balance | Needs multi-GPU | 16-24 GB (LoRA) |
| 13-14B | Preferred | Good balance | Multi-GPU required | 24-40 GB (LoRA) |
| 30-70B | LoRA or QLoRA | Preferred for single GPU | Multi-node | 40-80 GB (QLoRA) |

## Hyperparameter Ranges

| Parameter | LoRA | QLoRA | Full FT |
|-----------|------|-------|---------|
| `learning_rate` | 1e-4 to 3e-4 | 1e-4 to 3e-4 | 1e-5 to 5e-5 |
| `lora_r` | 16-64 | 16-64 | N/A |
| `lora_alpha` | 1-2x `lora_r` | 1-2x `lora_r` | N/A |
| `micro_batch_size` | 2-8 | 2-4 | 1-2 |
| `gradient_accumulation_steps` | 2-8 | 4-16 | 4-16 |
| `num_epochs` | 1-3 | 1-3 | 1-3 |
| `optimizer` | `adamw_8bit` | `adamw_bnb_8bit` | `adamw_torch_fused` |

Effective batch = micro_batch * grad_accum * num_gpus. Lower LR for larger models.

## Healthy Training Indicators

| Metric | Healthy | Problem |
|--------|---------|---------|
| `train_loss` | Decreasing, starting ~2-4 for chat models | Flat or increasing from step 1 — data or LR issue |
| `eval_loss` | Decreasing, tracks train_loss | Increasing while train_loss decreases — overfitting |
| `grad_norm` | 0.1-10, relatively stable | Spikes >100 — instability. 0.0 — frozen weights |
| `learning_rate` | Follows scheduler curve | Flat or NaN — config issue |

Watch for: loss never decreasing (check `train_on_inputs`, dataset, LR), loss goes to 0 quickly (overfitting), eval_loss diverging (reduce epochs, add regularization). See [training_stability.qmd](../training_stability.qmd).

## Known Issues

| Issue | Fix |
|-------|-----|
| OOM during training | Reduce `micro_batch_size`, enable `gradient_checkpointing`, reduce `sequence_len` |
| `sample_packing` + SDPA + bf16 = 0.0 loss | Use `attn_implementation: flash_attention_2` or disable `sample_packing` |
| Missing chat template error | Set `chat_template: chatml` explicitly |
| Label masking wrong | Run `axolotl preprocess config.yaml --debug` and inspect labels |
| Loss NaN | Use `bf16: auto`, lower LR, check data for empty samples |
| Tokenizer pad token / infinite loss | Set `special_tokens: pad_token: "<\|end_of_text\|>"` |
| FSDP save hangs | Use `fsdp_state_dict_type: FULL_STATE_DICT` |
| DeepSpeed CheckpointError | Set `use_reentrant: true` in `gradient_checkpointing_kwargs` |

## Profiling

To profile training and identify optimization opportunities:

```yaml
# Profile steps 3-7 (after warmup/autotuning settles)
profiler_steps_start: 3
profiler_steps: 5
```

This produces `profiler_trace.json` (Chrome trace) and `snapshot.pickle` (memory snapshot) in `output_dir`.
View the Chrome trace at `chrome://tracing`.

To programmatically inspect the trace:
```bash
python scripts/analyze_profile.py output_dir/
```

The trace shows per-kernel CUDA times, memory allocations, and operator-level breakdown. Look for:
- **Large matmul kernels**: candidates for fusion or quantization
- **Memory copies (H2D/D2H)**: unnecessary data movement
- **Small frequent kernels**: candidates for kernel fusion
- **Gaps between kernels**: pipeline bubbles from CPU overhead

Full troubleshooting: [training_stability.qmd](../training_stability.qmd), [debugging.qmd](../debugging.qmd)

## File Map

```
src/axolotl/
  cli/train.py                     # Entry point for `axolotl train`
  cli/preprocess.py                # Entry point for `axolotl preprocess`
  core/builders/causal.py          # HFCausalTrainerBuilder — wires config → SFT trainer
  core/trainers/base.py            # AxolotlTrainer — base trainer class
  core/trainers/mixins/            # Packing, optimizer, scheduler, checkpoints
  prompt_strategies/               # Format handlers: chat_template, alpaca, completion, input_output
  utils/schemas/config.py          # AxolotlInputConfig — main config schema
  utils/schemas/datasets.py        # SFTDataset, DatasetConfig
  utils/schemas/peft.py            # LoraConfig — LoRA parameters
  integrations/liger/              # Liger kernel plugin

examples/llama-3/                  # LoRA, QLoRA, full FT example configs
docs/getting-started.qmd           # Quickstart with config templates
docs/optimizations.qmd             # Flash attention, gradient checkpointing, sample packing
docs/multi-gpu.qmd                 # FSDP and DeepSpeed setup
```
