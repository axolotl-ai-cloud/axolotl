# Dynamic Fine-Tuning (DFT)

> Adaptive loss weighting for improved training quality in supervised fine-tuning

## Overview

**Dynamic Fine-Tuning (DFT)** is a training optimization that applies adaptive per-token weighting to the cross-entropy loss. Instead of treating all tokens equally, DFT gives higher weight to tokens that are neither too easy nor too hard to predict, focusing training on the "learning frontier."

**Key Formula**:
```
L_DFT = L_CE * exp(-L_CE.detach())
```

Where:
- `L_CE`: Standard cross-entropy loss per token
- `exp(-L_CE)`: Adaptive weight (higher for moderate losses, lower for extremes)
- `.detach()`: Prevents double backprop through the weighting term

**Benefits**:
- **Better sample efficiency**: Focus on informative tokens
- **Improved convergence**: Automatic curriculum learning effect
- **Robust to data quality**: Down-weights noisy/mislabeled tokens
- **No hyperparameters**: Weighting is dynamic and automatic

**When to Use**:
- Supervised fine-tuning (SFT) tasks
- Datasets with varying difficulty or noise
- Want better sample efficiency without manual tuning

**When NOT to Use**:
- RLHF / GRPO / ORPO training (incompatible)
- Using label smoothing (raises error)
- Prefer maximum speed over quality (Liger FLCE may be faster)

---

## Quick Start

### Basic Configuration

Add to your axolotl config YAML:

```yaml
# Enable DFT
enable_dft_loss: true

# That's it! DFT has no hyperparameters to tune
```

### With Sequence Packing (Recommended)

```yaml
# Typical SFT config with DFT
model_type: llama
base_model: meta-llama/Llama-2-7b-hf

# Data
datasets:
  - path: your/dataset
    type: sharegpt

# Packing for efficiency
sample_packing: true
pad_to_sequence_len: true

# Enable DFT
enable_dft_loss: true

# Mixed precision
bf16: true

# Standard optimizations (all compatible with DFT)
flash_attention: true
gradient_checkpointing: true
```

### With Large Vocabulary (Qwen, etc.)

For models with >50K vocab, use chunked cross-entropy to reduce memory:

```yaml
model_type: qwen2
base_model: Qwen/Qwen2.5-72B

enable_dft_loss: true
dft_chunk_size: 8192  # Memory optimization for 152K vocab

sample_packing: true
bf16: true
```

---

## Configuration Options

### Core DFT Settings

```yaml
# Enable DFT (default: false)
enable_dft_loss: true

# Chunked CE for large vocab models (default: None)
# Set to 4096-8192 for vocab >50K tokens
dft_chunk_size: 8192

# Channel Loss integration (default: false)
# Enables per-token loss intermediates for Channel Loss plugin
enable_dft_channel_loss: false
```

### Token Metrics (Optional)

```yaml
# Track token counts during training (default: false)
include_tkps: true
# Logs: trainer.state.num_tokens, trainer.state.total_tokens
```

---

## Compatibility

DFT is compatible with most axolotl features. See **[DFT_COMPATIBILITY.md](./DFT_COMPATIBILITY.md)** for full details.

### ✅ Compatible (Works Together)

- **Sequence Packing** - Verified
- **Data Parallelism (DDP)** - Verified
- **FSDP / DeepSpeed ZeRO** - Compatible
- **Tensor Parallelism** - Verified
- **Context Parallelism (SFT mode)** - Verified
- **Gradient Accumulation** - Verified
- **Mixed Precision (FP16/BF16)** - Verified
- **Flash Attention** - Transparent
- **Gradient Checkpointing** - Transparent
- **Channel Loss Integration** - Opt-in

### ❌ Incompatible (Don't Use Together)

- **Label Smoothing** - Raises ValueError
- **ORPO** - Silent fallback (DFT disabled)
- **Liger FLCE** - Conflicts with DFT chunked CE (choose one)
- **Cut Cross Entropy** - Incompatible approaches

See **[Decision Tree: When to Use DFT](./DFT_COMPATIBILITY.md#decision-tree-when-to-use-dft)** for detailed guidance.

---

## Example Configurations

### Small Model (7B, Single GPU)

```yaml
model_type: llama
base_model: meta-llama/Llama-2-7b-hf

datasets:
  - path: your/dataset
    type: sharegpt

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

# DFT for better sample efficiency
enable_dft_loss: true

# Standard optimizations
bf16: true
flash_attention: true
gradient_checkpointing: true

# Training hyperparams
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2e-5

optimizer: adamw_torch
lr_scheduler: cosine
warmup_steps: 100
```

### Large Model (70B, Multi-GPU FSDP)

```yaml
model_type: llama
base_model: meta-llama/Llama-2-70b-hf

# FSDP for model sharding
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_offload_params: false
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

# DFT is compatible with FSDP
enable_dft_loss: true

sample_packing: true
bf16: true

micro_batch_size: 1
gradient_accumulation_steps: 8

flash_attention: true
gradient_checkpointing: true
```

### Huge Vocab Model (Qwen 152K tokens)

```yaml
model_type: qwen2
base_model: Qwen/Qwen2.5-72B

# Tensor Parallelism for large model
tensor_parallel_size: 4

# DFT with chunked CE for memory efficiency
enable_dft_loss: true
dft_chunk_size: 8192  # Critical for 152K vocab

sample_packing: true
bf16: true

micro_batch_size: 1
gradient_accumulation_steps: 4

flash_attention: true
gradient_checkpointing: true
```

### Long Context (32K+ tokens with CP)

```yaml
model_type: llama
base_model: meta-llama/Llama-2-7b-hf

sequence_len: 32768

# Context Parallelism for long sequences
context_parallel_size: 4

# DFT has CP-aware implementation
enable_dft_loss: true

sample_packing: true
bf16: true

flash_attention: true
gradient_checkpointing: true
```

---

## How DFT Works

### Adaptive Weighting Intuition

Standard cross-entropy treats all tokens equally:
```
L_CE = mean(per_token_losses)
```

DFT applies adaptive weighting:
```
weighted_loss = per_token_loss * exp(-per_token_loss)
L_DFT = mean(weighted_loss)
```

**Effect**:
- **Low loss (0.1)**: `0.1 * exp(-0.1) ≈ 0.09` - Easy tokens get lower weight
- **Medium loss (2.0)**: `2.0 * exp(-2.0) ≈ 0.27` - Learning frontier gets highest weight
- **High loss (6.0)**: `6.0 * exp(-6.0) ≈ 0.01` - Outliers/noise get down-weighted

This creates an automatic curriculum that focuses on the "Goldilocks zone" of difficulty.

### Implementation Details

**Loss Computation Flow**:
```
1. Model forward → logits [batch, seq, vocab]
2. Compute per-token CE loss (with optional chunking for large vocab)
3. Apply DFT weighting: loss * exp(-loss.detach())
4. Reduce to scalar (masked by ignore_index=-100)
5. Backward pass (gradients flow through weighted loss)
```

**Key Files**:
- `patch.py` - Monkey patches `trainer.compute_loss()`
- `dft_utils.py` - Core loss computation with CP awareness
- `chunked_ce.py` - Memory-efficient CE for large vocab
- `args.py` - Configuration schema

**Special Handling**:
- **Context Parallelism**: CP-aware label slicing for sharded sequences
- **Mixed Precision**: Automatic `.float()` upcast for numerical stability
- **Packing**: Respects `-100` ignore_index for sequence boundaries
- **Gradient Accumulation**: Correct normalization via `num_items_in_batch`

---

## Advanced Features

### Channel Loss Integration

DFT can provide per-token losses and valid masks for integration with Channel Loss plugin:

```yaml
enable_dft_loss: true
enable_dft_channel_loss: true  # Expose intermediates
```

This attaches to model outputs:
- `outputs.per_token_loss`: `[batch*seq]` tensor with DFT-weighted losses
- `outputs.valid_mask`: `[batch*seq]` bool tensor (True where label != -100)
- `outputs.loss`: Scalar loss for backward

See `CHANNEL_LOSS_INTEGRATION.md` for details.

### Token Metrics Tracking

Track token counts during training:

```yaml
enable_dft_loss: true
include_tkps: true
```

Logs to `trainer.state`:
- `num_tokens`: Token count for current rank
- `total_tokens`: Token count across all ranks (if distributed)

Useful for:
- Tokens per second (TPS) calculations
- Distributed training verification
- Training progress monitoring

---

## Troubleshooting

### Error: "DFT loss is currently incompatible with label smoothing"

**Cause**: You have `label_smoothing_factor > 0` in config

**Solution**: Remove or set to 0
```yaml
enable_dft_loss: true
label_smoothing_factor: 0  # or remove this line
```

### OOM (Out of Memory) with large vocabulary

**Cause**: Large vocab (>50K tokens) materializes huge tensors

**Solution 1**: Enable chunked CE
```yaml
enable_dft_loss: true
dft_chunk_size: 8192  # Adjust based on vocab size and VRAM
```

**Solution 2**: Use Liger FLCE instead (faster, more memory-efficient)
```yaml
enable_dft_loss: false
# Configure Liger FLCE via plugins
```

### DFT silently disabled (using ORPO)

**Cause**: Both `enable_dft_loss` and `orpo_alpha` in config

**Explanation**: ORPO uses a different loss function. DFT falls back to ORPO automatically.

**Solution**: Choose one
```yaml
# Option 1: Use DFT
enable_dft_loss: true
# Remove: orpo_alpha

# Option 2: Use ORPO
orpo_alpha: 0.1
enable_dft_loss: false  # or remove
```

### Loss seems different with packing vs without

**Expected**: DFT's non-linear weighting (`loss * exp(-loss)`) causes packed and unpacked losses to differ

**Explanation**: This is correct behavior, not a bug. Both produce valid training signals, just with different token-level weights.

---

## Performance Considerations

### Memory Usage

**Standard DFT**: Same memory as vanilla CE

**DFT with large vocab (>50K)**:
- Without chunking: Can OOM due to `[batch*seq, vocab]` tensor
- With `dft_chunk_size=8192`: Reduces memory by `vocab / 8192`

**DFT vs Liger FLCE**:
- DFT: Pure PyTorch, moderate memory
- Liger FLCE: Triton kernel, fused ops, more memory-efficient

### Speed

**DFT overhead**: ~5-10% slower than vanilla CE
- Per-token loss computation: minimal overhead
- `exp(-loss)` weighting: very fast
- Main cost: Cannot fuse with linear layer (unlike Liger FLCE)

**Optimization tips**:
- Use `bf16` for faster computation
- Enable `flash_attention` for attention speedup (orthogonal to DFT)
- For huge vocab (>100K): Consider Liger FLCE if speed is critical

### Training Quality

**DFT benefits**:
- Better sample efficiency (fewer steps to converge)
- More robust to noisy data
- Automatic curriculum learning

**Trade-offs**:
- Slight computational overhead
- Cannot use with label smoothing
- Different loss values (not directly comparable to vanilla CE)

---

## Testing

DFT has comprehensive test coverage:

**Run all DFT tests**:
```bash
pytest tests/integrations/test_dft_*.py -v
```

**Test suites**:
- `test_dft_packing.py` - Sequence packing compatibility (7 tests)
- `test_dft_ddp.py` - Data parallelism (5 tests)
- `test_dft_cp_compatibility.py` - Context parallelism (5 tests)
- `test_dft_tensor_parallel.py` - Tensor parallelism (4 tests)
- `test_dft_incompatibilities.py` - Incompatibility detection (6 tests)
- `test_dft_compatibility.py` - Feature compatibility (9 tests)
- `test_dft_multi_feature.py` - Multi-feature combinations (9 tests)

**Total**: 45+ tests covering all compatibility claims

---

## References

- **Full Compatibility Matrix**: [DFT_COMPATIBILITY.md](./DFT_COMPATIBILITY.md)
- **Channel Loss Integration**: [CHANNEL_LOSS_INTEGRATION.md](./CHANNEL_LOSS_INTEGRATION.md)
- **Compatibility Spec**: `specs/001-dft-compatibility-matrix/README.md`
- **Implementation**: `src/axolotl/integrations/dft/`

---

## FAQ

**Q: Does DFT have hyperparameters to tune?**

A: No! DFT's weighting is fully automatic via `exp(-loss)`. The only optional parameter is `dft_chunk_size` for memory optimization with large vocabs.

**Q: Can I use DFT with my existing config?**

A: Probably yes! DFT is compatible with most features. Just add `enable_dft_loss: true` and check [DFT_COMPATIBILITY.md](./DFT_COMPATIBILITY.md) for incompatibilities.

**Q: How much does DFT slow down training?**

A: ~5-10% overhead compared to vanilla CE. The sample efficiency improvement often compensates for this.

**Q: Should I use DFT or Liger FLCE for large vocab models?**

A: See the [Decision Tree](./DFT_COMPATIBILITY.md#decision-tree-when-to-use-dft):
- **Training quality priority**: DFT + chunked CE
- **Speed/memory priority**: Liger FLCE
- **Both**: Start with DFT, switch to Liger if speed is bottleneck

**Q: Does DFT work with multi-GPU training?**

A: Yes! DFT is verified with DDP, FSDP, Tensor Parallelism, and Context Parallelism.

**Q: Can I use DFT for RLHF/GRPO?**

A: Not tested. DFT is designed for SFT (supervised fine-tuning) with CE loss. ORPO has automatic fallback.

---

**Last Updated**: 2026-01-06 (Phase 6: Documentation)
