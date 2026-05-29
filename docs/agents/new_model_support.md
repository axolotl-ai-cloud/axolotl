# New Model Support — Agent Reference

Guide for debugging and adding support for new model architectures in axolotl. Based on lessons learned from Gemma4, Gemma3, Qwen2-VL, and other multimodal/MoE models.

## Quick Validation Checklist

When testing a new model, run through these checks in order:

1. **Does the model load?** `axolotl preprocess config.yaml` — catches config schema errors
2. **Does LoRA apply?** Check for "Unsupported layer type" warnings from PEFT
3. **Is the initial loss sane?** First-step loss for a pretrained model should be 0.5–2.0 for SFT
4. **Does sample packing work?** Compare loss with `sample_packing: true` vs `false` — should be similar
5. **Is CCE active?** Check for "Applying Cut Cross Entropy" log and verify peak VRAM is lower

## Loss Debugging

### Expected initial loss
A pretrained model doing SFT should start with loss roughly in the 0.5–2.0 range. If loss starts above 3.0, something is wrong. If it's near `log(vocab_size)` (≈ 12 for 262K vocab), the model is predicting at random — attention masking or model weights are broken.

### Direct comparison technique
The fastest way to isolate a loss issue — bypass the trainer entirely:

```python
# Load model via axolotl's pipeline (applies all patches)
from axolotl.cli.config import load_cfg
from axolotl.utils.config import normalize_config, prepare_plugins
from axolotl.loaders.tokenizer import load_tokenizer
from axolotl.loaders.model import ModelLoader

cfg = load_cfg("your_config.yaml")
normalize_config(cfg)
prepare_plugins(cfg)
tokenizer = load_tokenizer(cfg)
model, _ = ModelLoader(cfg, tokenizer).load()

# Forward pass on preprocessed data
model.train()
out = model(input_ids, labels=labels)
print(f"Direct loss: {out.loss.item()}")  # Compare to trainer's reported loss
```

If direct loss is correct (~1.0) but trainer reports 3–4x higher, check `model_accepts_loss_kwargs` (see below).

### `model_accepts_loss_kwargs` inflation
HF Trainer checks if the model's `forward()` has `**kwargs` and sets `model_accepts_loss_kwargs=True`. This changes loss normalization: the trainer does NOT divide loss by `gradient_accumulation_steps` before logging. The gradient is correct — only the logged loss is inflated.

**Symptom**: Logged loss ≈ actual_loss × gradient_accumulation_steps.

**Which models are affected**: Any model with `**kwargs` in forward (common in multimodal models for extra inputs like `mm_token_type_ids`, `pixel_values`, etc.).

**Fix location**: `src/axolotl/core/trainers/base.py` `__init__()` — after `super().__init__()`, check if the unwrapped model actually has `num_items_in_batch` in its forward signature. If not, set `self.model_accepts_loss_kwargs = False`.

## Multimodal Models (ForConditionalGeneration)

Many recent models use `ForConditionalGeneration` as the top-level class, not `ForCausalLM`:
- Gemma3 → `Gemma3ForConditionalGeneration`
- Gemma4 → `Gemma4ForConditionalGeneration`
- Qwen2-VL → `Qwen2VLForConditionalGeneration`
- LLaVA → `LlavaForConditionalGeneration`

### Why this matters

| Component | Targets `ForCausalLM` | Needs `ForConditionalGeneration` |
|-----------|----------------------|--------------------------------|
| CCE patches | ✅ (default) | ❌ silently inactive if not patched |
| PEFT LoRA | ✅ | May fail on custom layer types |
| HF Trainer label handling | ✅ | May need extra inputs |

### Required extra inputs
Multimodal models require special inputs during training even for text-only data:

| Model | Required Input | Value for Text-Only |
|-------|---------------|-------------------|
| Gemma4 | `mm_token_type_ids` | `torch.zeros_like(input_ids)` |
| Gemma3 | `token_type_ids` | `torch.zeros_like(input_ids)` |

Auto-inject in `compute_loss()` when not provided by the data collator. See `core/trainers/base.py`.

### Custom layer types and PEFT
Vision towers often use custom module wrappers that PEFT doesn't support:

| Model | Custom Layer | Wraps | Fix |
|-------|-------------|-------|-----|
| Gemma4 | `Gemma4ClippableLinear` | `nn.Linear` | Redirect to `.linear` child |

Fix location: `src/axolotl/loaders/adapter.py` `_patch_peft_clippable_linear()`.

## Sample Packing

### How packed sequence detection works (transformers ≥ 5.x)
`transformers.masking_utils._preprocess_mask_arguments()` detects packed sequences from `position_ids` resets. But **only when `attention_mask is None`**:

```python
# From masking_utils.py:
if position_ids is not None and attention_mask is None and past_key_values is None:
    packed_sequence_mask = find_packed_sequence_indices(position_ids)
```

If the collator provides an all-ones `attention_mask`, packing detection is **skipped** and the model builds a single causal mask spanning all packed sequences → cross-sequence attention leakage → very high loss.

### Fix for models using `create_causal_mask_mapping`
For Gemma3, Gemma4, and similar models that use the new transformers masking system, remove `attention_mask` from inputs when sample packing is active:

```python
# In compute_loss():
if (
    self.args.sample_packing
    and model_type in ("gemma4", "gemma3")
    and "attention_mask" in inputs
    and "position_ids" in inputs
):
    del inputs["attention_mask"]
```

Fix location: `src/axolotl/core/trainers/base.py` `compute_loss()`.

### Models that DON'T need this fix
Older models that use `_prepare_4d_causal_attention_mask` (Llama, Mistral, Qwen2, etc.) handle sample packing via axolotl's multipack attention monkeypatch instead. Only models using the new `create_causal_mask_mapping` / `create_causal_mask` masking system need the `attention_mask` removal.

## Attention Backend Selection

| Backend | Config | head_dim limit | torch_compile | Notes |
|---------|--------|---------------|---------------|-------|
| FA2 | `attn_implementation: flash_attention_2` | 256 | ✅ | Fastest when supported |
| FA4 | auto with `attn_implementation: flash_attention_2` | 256 (SM90+) | ✅ | Auto-detected on H100+ |
| SDPA | `attn_implementation: sdpa` | None | ✅ | Universal fallback |
| flex | `attn_implementation: flex_attention` | None | ⚠️ Triton OOM for large head_dim | Good for variable head dims |
| eager | `attn_implementation: eager` | None | ✅ | Slowest, always works |

**Check model support**: Look at `_supports_flash_attn_2`, `_supports_flex_attn`, `_supports_sdpa` attributes on the model class.

**head_dim gotcha**: The 256 limit is specific to flash-attn CUDA kernels, NOT PyTorch-level. SDPA and flex_attention both handle arbitrary head_dim. Models with `global_head_dim > 256` (Gemma4: 512) must use SDPA or flex.

**flex + compile gotcha**: `torch_compile` with flex_attention can hit Triton shared memory OOM for large head_dim. Falls back to eager per-function (not a crash, but slower). Unsloth disables flex for Gemma4 for this reason.

## Cut Cross Entropy (CCE)

### How CCE patches work
CCE replaces the model's `forward()` with a fused version that computes loss from hidden states + lm_head weight without materializing the full logits tensor. This saves ~`batch × seq_len × vocab_size × dtype_bytes` of VRAM.

### Adding CCE for a new model
1. Check if the model type is in `cut_cross_entropy.transformers.patch.PATCH_FNS`
2. If not, axolotl's generic fallback (`integrations/cut_cross_entropy/__init__.py` `patch_llama_like()`) patches `{Prefix}ForCausalLM.forward` with `cce_forward`
3. For multimodal models (`ForConditionalGeneration`), a model-specific patch is needed in `ml-cross-entropy` repo
4. The multimodal `cce_forward` must accept all extra kwargs (pixel_values, mm_token_type_ids, etc.) and pop any that would conflict before calling `self.model()`

### Common CCE pitfall
If CCE appears active (log says "Applying Cut Cross Entropy") but peak VRAM doesn't decrease, check which class was patched. If the model loads as `ForConditionalGeneration` but CCE patched `ForCausalLM`, the patch is silently inactive.

## MoE Models

### Dense MLP vs MoE experts
Some MoE models (e.g., Gemma4) have BOTH dense MLP layers and MoE expert layers at every decoder layer:
- `gate_proj/up_proj/down_proj` → targets the **dense MLP** (`Gemma4TextMLP`)
- `experts.gate_up_proj/experts.down_proj` → targets the **MoE experts** (`Gemma4TextExperts`)

LoRA on the dense MLP works normally. Expert LoRA via `lora_target_parameters` requires PEFT support for the specific expert module type (may warn "Unsupported layer type").

### ScatterMoE kernels
`use_scattermoe: true` with `experts_implementation: scattermoe` registers fused expert kernels via transformers' `ExpertsInterface`. Significant speedup for MoE models. Requires the kernels plugin:
```yaml
plugins:
  - axolotl.integrations.kernels.KernelsPlugin
use_kernels: true
use_scattermoe: true
experts_implementation: scattermoe
```

## Where to Add Model-Specific Fixes

| What | Where | Example |
|------|-------|---------|
| Missing forward inputs | `core/trainers/base.py` `compute_loss()` | mm_token_type_ids injection |
| Attention mask fixes | `core/trainers/base.py` `compute_loss()` | Sample packing mask removal |
| Loss logging fixes | `core/trainers/base.py` `__init__()` | model_accepts_loss_kwargs override |
| PEFT/LoRA patches | `loaders/adapter.py` | ClippableLinear redirect |
| Attention patches | `monkeypatch/attention/` | FA4 tuple fix |
| Model-specific patches | `loaders/patch_manager.py` `_apply_model_specific_patches()` | Llama4, Kimi, NemotronH |
| CCE patches | `ml-cross-entropy` repo `transformers/` | Per-model cce_forward |
| Example configs | `examples/<model>/` | Validated YAML |
| Config validation | `utils/schemas/validation.py` | Compatibility checks |
