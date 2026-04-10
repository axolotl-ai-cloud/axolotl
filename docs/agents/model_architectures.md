# Model Architectures — Agent Reference

Model-specific quirks, required settings, and known issues. Check this before debugging training failures on specific model families.

## Gemma 4

**Models**: `google/gemma-4-26B-A4B` (MoE), `google/gemma-4-31B` (dense), `google/gemma-4-E2B`, `google/gemma-4-E4B`

**Architecture**: Multimodal wrapper (`Gemma4ForConditionalGeneration`) over a text backbone (`Gemma4TextModel`), with optional vision/audio encoders. All Gemma4 HF repos have `model_type: "gemma4"` — even text-only variants load as multimodal with a vision tower.

### Required settings

```yaml
# Always needed for Gemma4:
freeze_mm_modules: true          # Freeze vision/audio encoders for text-only training
gradient_checkpointing_kwargs:
  use_reentrant: false           # Shared per-layer norms cause "marked ready twice" with reentrant

# LoRA target — restrict to language model only (DO NOT use lora_target_linear: true):
lora_target_modules: 'model.language_model.layers.[\d]+.(_checkpoint_wrapped_module.)?(mlp|self_attn).(up|down|gate|q|k|v|o)_proj'
```

### Auto-detection

Axolotl auto-detects Gemma4 and applies:
- `use_reentrant: false` for gradient checkpointing
- `ddp_find_unused_parameters: true` for DDP (skipped when `activation_offloading: true`)

### Multi-GPU

| Strategy | Works? | Notes |
|----------|--------|-------|
| DDP | Yes | Auto-sets `ddp_find_unused_parameters=True` |
| DDP + activation_offloading | Yes | `find_unused_parameters` is skipped (conflicts with checkpoint wrappers) |
| FSDP1 | No | OOM during dequantization/sharding with QLoRA |
| FSDP2 | Yes | Use `Gemma4TextDecoderLayer` (not `Gemma4DecoderLayer`) as wrap class |
| FSDP2 + activation_offloading | Yes | Lowest VRAM (~26 GiB/GPU for 26B-A4B) |

FSDP2 config:
```yaml
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Gemma4TextDecoderLayer
```

### MoE (26B-A4B)

- `enable_moe_block: true`, 256 experts, top-k routing
- No separate `SparseMoeBlock` — MoE is embedded in each decoder layer
- Expert LoRA targets 3D parameter tensors:
  ```yaml
  lora_target_parameters:
    - experts.gate_up_proj
    - experts.down_proj
  ```
- ScatterMoE kernel acceleration:
  ```yaml
  plugins:
    - axolotl.integrations.kernels.KernelsPlugin
  use_kernels: true
  use_scattermoe: true
  experts_implementation: scattermoe
  ```

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `mm_token_type_ids is required` in DDP | `model.config` not accessible through DDP wrapper | Already fixed — `unwrap_model()` in `compute_loss` and `prediction_step` |
| `marked a variable ready twice` in DDP | `ddp_find_unused_parameters=True` + activation_offloading checkpoint wrappers | Auto-handled — `find_unused_parameters` is skipped when `activation_offloading: true` |
| Loss ~12 instead of ~0.5 | Using `lora_target_linear: true` (applies LoRA to vision/audio modules) | Use the regex `lora_target_modules` pattern instead |
| FSDP2 `Could not find Gemma4AudioLayer` | Auto-wrap detects `_no_split_modules` including audio layers that don't exist | Explicitly set `fsdp_transformer_layer_cls_to_wrap: Gemma4TextDecoderLayer` |
| `Gemma4ClippableLinear not supported` by PEFT | Vision tower uses a non-standard linear wrapper | Axolotl patches this automatically via `_patch_peft_clippable_linear()` |

### E2B/E4B dense models

These have `hidden_size_per_layer_input: 256` (per-layer input embeddings) and `attention_k_eq_v: False`. Known issue: loss starts higher than expected (~12 vs ~0.5 for 26B). Root cause under investigation — may be related to the per-layer input mechanism or the `Gemma4ForConditionalGeneration` loss computation.

## Gemma 3

**Models**: `google/gemma-3-*`

- `ddp_find_unused_parameters: true` needed (multimodal unused params)
- `use_reentrant: false` recommended
- Attention mask must be dropped for sample packing (handled automatically)
- Multi-GPU test currently skipped (`tests/e2e/multigpu/test_gemma3.py`)

## Qwen 3.5 MoE

**Models**: `Qwen/Qwen3.5-35B-A3B`

- Hybrid architecture: DeltaNet linear attention (30 layers) + full attention (10 layers)
- 256 experts, 8 active per token
- Known weight scale drift in late DeltaNet layers (36-38) due to AdamW + rare expert interaction
- Fix: `normalize_weight_scales` config to detect and rescale outliers:
  ```yaml
  normalize_weight_scales:
    - name_pattern: 'linear_attn\.conv1d\.weight'
      threshold: 1.3
  ```

## General MoE Notes

- `lora_target_linear: true` with multimodal MoE models will apply LoRA to ALL linear modules including vision/audio encoders — use regex `lora_target_modules` to restrict to language model only
- Rare experts get larger effective learning rate from AdamW (small second-moment estimates) — can cause weight drift in recurrent/SSM components. Use `normalize_weight_scales` with `dry_run: true` to detect.
- For ScatterMoE kernel support, set `experts_implementation: scattermoe` and add the KernelsPlugin
