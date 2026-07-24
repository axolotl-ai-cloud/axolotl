# New Model Support — Agent Reference

Guide for debugging and adding support for new model architectures in axolotl. Based on lessons learned from Gemma4, Gemma3, Qwen2-VL, and other multimodal/MoE models.

## Model Support Registry

New architectures are described by a `ModelSupport` descriptor in `src/axolotl/model_support/<model_type>/`. Its declarative `ModelProfile` starts from a reusable family template, then supplies only the capabilities, component strategies, discovery matchers, and lifecycle hooks where that model differs from the family path. Features query the registry instead of adding new `model_type` branches throughout loaders and integrations.

Use `VANILLA_CAUSAL_LM` for a standard text causal-language-model path and `IMAGE_TEXT_TO_TEXT` for a standard multimodal conditional-generation path. Strategy values are lazy zero-argument providers so optional or heavyweight classes are imported only when the pipeline needs them.

```python
# src/axolotl/model_support/my_model/__init__.py
from axolotl.model_support import (
    IMAGE_TEXT_TO_TEXT,
    Experimental,
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelStrategyOverrides,
    ModelSupport,
    Unsupported,
    register_model_support,
)


def _processing_strategy_cls():
    from .processing import MyModelProcessingStrategy

    return MyModelProcessingStrategy


def _matches_processor(processor):
    from transformers import MyModelProcessor

    return isinstance(processor, MyModelProcessor)


def _before_model_build(context: ModelHookContext):
    from .patches import patch_my_model

    patch_my_model(context.cfg)


@register_model_support
class MyModelSupport(ModelSupport):
    model_types = ("my_model",)
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            "cut_cross_entropy": Unsupported("No conditional-generation CCE forward."),
            "sample_packing": Experimental("Verify loss parity against unpacked batches."),
        },
        strategies=ModelStrategyOverrides(
            processing_strategy_cls=_processing_strategy_cls,
        ),
        matchers=ModelMatchers(processor=_matches_processor),
        hooks=ModelHooks(
            {
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
            }
        ),
    )
```

A missing capability means unknown and preserves the feature's generic handling. `Unsupported(reason)` raises when the feature is enabled, `Experimental(note)` warns, and `Supported(note)` records verified coverage. A profile capability value of `None` removes an inherited family declaration and restores unknown handling.

For an in-tree descriptor, add its module path to `_BUILTIN_MODULES` in `src/axolotl/model_support/registry.py`; the registry imports those modules lazily, which executes the decorator. An out-of-tree plugin must import the descriptor module while the plugin is registered and must not edit `_BUILTIN_MODULES`. Verify activation without importing the descriptor directly, so the test proves registry discovery rather than triggering registration itself:

```python
from axolotl.model_support import get_model_support

support = get_model_support("my_model")
assert support is not None
assert type(support).__name__ == "MyModelSupport"
```

### Resolution and precedence

`resolve_model_support()` produces an immutable effective profile. Values resolve in this order:

1. Family-template defaults.
2. Per-model profile overrides.
3. Attributes and methods explicitly declared by a legacy `ModelSupport` subclass.

Capabilities merge by key, strategies and matchers override field by field, and hooks run in family → profile → legacy order. Inherited defaults on `ModelSupport` do not erase profile values. This keeps existing descriptors compatible while allowing new descriptors to remain declarative.

`ModelStrategyOverrides` distinguishes omission from removal: an omitted field inherits its family provider, while explicit `None` removes that provider and restores the downstream generic fallback. Profile hooks append by default; include a phase in `ModelHooks.replace_phases` to replace its inherited family hooks, using an empty tuple to suppress them entirely. Legacy method hooks remain additive after the declarative result.

Use `ModelFamilyTemplate` when several architectures share more than the built-in vanilla paths. Keep the template limited to genuinely shared behavior; model folders should add only their own matcher, strategy, capability differences, and localized patches.

### Migrating a legacy descriptor

Move one concern at a time using this mapping:

| Legacy declaration | Profile equivalent |
|--------------------|--------------------|
| `is_multimodal = True` | `ModelProfile(family=IMAGE_TEXT_TO_TEXT)`, or `ModelProfile(family=custom_family, is_multimodal=True)` |
| `capabilities = {...}` | `ModelProfile(capabilities={...})` |
| `get_auto_model_cls()` | `ModelStrategyOverrides(auto_model_cls=provider)` |
| `get_processing_strategy_cls()` | `ModelStrategyOverrides(processing_strategy_cls=provider)` |
| `matches_cfg()` / `matches_processor()` | `ModelMatchers(cfg=...)` / `ModelMatchers(processor=...)` |
| `validate_cfg()` | `CONFIGURE_RUN` hook |
| `pre_config_load()` / `pre_tokenizer_load()` / `pre_model_load()` | `BEFORE_CONFIG_LOAD` / `BEFORE_TOKENIZER_LOAD` / `BEFORE_MODEL_BUILD` hook |
| `post_model_load()` | `AFTER_ADAPTER_LOAD` hook |

Unless a profile phase explicitly replaces its family hooks, hooks from the family, profile, and legacy method are additive. Remove a legacy method once its behavior moves into a profile hook, or the old and new implementations both run. The compatibility guard prevents a legacy method's `super()` call from re-entering the declarative hook, but it cannot identify duplicate logic implemented in both places. `AFTER_BASE_MODEL_BUILD` has no legacy-method equivalent.

### Hook phases

Hooks receive an immutable `ModelHookContext` containing the run config and the objects available at that phase. The `tokenizer` and `processor` fields are optional because early phases do not construct them and direct `ModelLoader` callers can omit a processor; hooks must handle `None` unless their phase and caller guarantee the object. Register hooks under `ModelHookPhase` rather than relying on an implied ordering.

| Phase | Meaning |
|-------|---------|
| `BEFORE_CONFIG_LOAD` | Before `AutoConfig.from_pretrained`; discovery uses the config matcher because the Hugging Face `model_type` may not be known yet. |
| `CONFIGURE_RUN` | After the model config and exact registry descriptor are known; use for model-specific validation or derived config values. |
| `BEFORE_TOKENIZER_LOAD` | Before tokenizer construction; discovery uses the config matcher or the already-resolved exact model type. |
| `BEFORE_MODEL_BUILD` | At the established pre-load patch slot before checkpoint construction; use only for patches that must exist while the model class is imported or instantiated. |
| `AFTER_BASE_MODEL_BUILD` | Immediately after the raw base model is built and before adapters are applied. |
| `AFTER_ADAPTER_LOAD` | After adapters and final load setup, before the remaining generic and plugin post-load hooks. |

The context fields available at each phase are:

| Phase | `model_config` | `tokenizer` | `processor` | `model` | `inference` / `reference_model` |
|-------|----------------|-------------|-------------|---------|---------------------------------|
| `BEFORE_CONFIG_LOAD` | — | — | — | — | `inference` from `cfg` when set; `reference_model` unavailable |
| `CONFIGURE_RUN` | Available | — | — | — | `inference` from `cfg` when set; `reference_model` unavailable |
| `BEFORE_TOKENIZER_LOAD` | — | — | — | — | `inference` from `cfg` when set; `reference_model` unavailable |
| `BEFORE_MODEL_BUILD` | Available | Available | Optional | — | Available |
| `AFTER_BASE_MODEL_BUILD` | Available | Available | Optional | Raw base model | Available |
| `AFTER_ADAPTER_LOAD` | Available | Available | Optional | Final loaded model | Available |

`cfg` is always available. Although the context dataclass prevents assigning different field values, the contained config and model objects remain mutable so hooks can apply their intended configuration or patch. At `BEFORE_TOKENIZER_LOAD`, the exact type may already be available as `cfg.model_config_type`, but the `model_config` object itself is not included in the context. The standard loading helper supplies a processor for multimodal runs; direct `ModelLoader` callers must pass one explicitly if their hooks require it.

The hook runner deliberately does not suppress repeated calls. A long-lived process may load multiple models or need per-run reconfiguration, so module-level monkeypatch functions must be idempotent themselves: guard against re-wrapping, preserve the original callable when practical, and leave model-instance hooks safe to run for every model.

### Registration and fallback

Register out-of-tree descriptors by calling `register_model_support()` from an imported plugin module. Built-ins are loaded before an external registration is installed, so a plugin can intentionally override a built-in `model_type`. Config and processor matchers must identify at most one descriptor; ambiguous matches raise instead of depending on import order. Once `cfg.model_config_type` is known, exact registry lookup takes precedence over heuristic matchers.

A matcher is a *global* predicate: `matches_cfg`/`matches_processor` run against every registered descriptor for every run, so scope them tightly (e.g. a specific model-name substring) and keep them side-effect-free. A matcher that returns `True` for a model it does not own will either shadow another descriptor or raise an ambiguity error and abort an otherwise-unrelated run. Because exact `model_config_type` lookup wins once the type is resolved, a matcher can only shadow a built-in during the pre-config/pre-tokenizer window before the config is loaded; to override a built-in for the whole run, register a descriptor under the same `model_type` instead of relying on a matcher.

Descriptors without a profile retain neutral legacy behavior: their resolved family is `None`, no component provider is injected, and explicitly declared legacy attributes and methods are adapted into the resolved result. This preserves the existing generic and hardcoded loader and processing fallbacks, including for legacy multimodal descriptors. Unregistered models, missing capabilities, and unset strategy fields also continue through those fallback paths. Profile resolution does not yet replace trainer selection, optimizer construction, adapter loading, loss adaptation, batch preparation, or every legacy `model_type` branch; add those as typed strategies only when the vanilla staged pipeline has a stable replacement seam.

See `model_support/paddleocr_vl/` for a multimodal profile and `model_support/kimi_linear/` for phase-specific idempotent patch hooks.

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
from axolotl.cli.utils.load import load_model_and_tokenizer

cfg = load_cfg("your_config.yaml")
model, tokenizer, processor = load_model_and_tokenizer(cfg=cfg)

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

For a new architecture, start with a `ModelSupport` descriptor in `model_support/<model_type>/` (capabilities, processing strategy, load hooks, config validation). The locations below are for fixes the registry does not cover yet, and for architectures not yet ported to it:

| What | Where | Example |
|------|-------|---------|
| Missing forward inputs | `core/trainers/base.py` `compute_loss()` | mm_token_type_ids injection |
| Attention mask fixes | `core/trainers/base.py` `compute_loss()` | Sample packing mask removal |
| Loss logging fixes | `core/trainers/base.py` `__init__()` | model_accepts_loss_kwargs override |
| PEFT/LoRA patches | `loaders/adapter.py` | ClippableLinear redirect |
| Attention patches | `monkeypatch/attention/` | FA4 tuple fix |
| Legacy model-specific patches | `loaders/patch_manager.py` `_apply_model_specific_patches()` | Llama4, Kimi, NemotronH |
| CCE patches | `ml-cross-entropy` repo `transformers/` | Per-model cce_forward |
| Example configs | `examples/<model>/` | Validated YAML |
| Config validation (generic) | `utils/schemas/validation.py` | Compatibility checks |
