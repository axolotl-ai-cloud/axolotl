# Gemma3 Text-from-Multimodal Plugin

Load a Gemma3 multimodal checkpoint (e.g. `google/gemma-3-4b-it`) directly into `Gemma3ForCausalLM` for text-only training. This bypasses the multimodal trainer path and enables sample packing and other text-specific optimizations.

## How it works

The plugin uses transformers v5's `key_mapping` parameter on `from_pretrained` to remap `model.language_model.*` checkpoint keys to `model.*`, matching what `Gemma3ForCausalLM` expects. Vision tower and projector weights are automatically discarded. On save, transformers reverses the mapping so checkpoints retain the original `model.language_model.*` prefix.

## Usage

Add the plugin to your YAML config:

```yaml
base_model: google/gemma-3-4b-it

plugins:
  - axolotl.integrations.gemma3.Gemma3TextFromMultimodalPlugin
```

See `examples/gemma3/gemma-3-4b-qlora.yml` for a complete example.

## Merging weights back into a multimodal checkpoint

After training, the saved checkpoint contains only the language model weights. To reconstruct a full `Gemma3ForConditionalGeneration` checkpoint (with the original vision tower and projector), use the merge script:

```bash
python scripts/merge_gemma3_multimodal_weights.py \
    --original-model google/gemma-3-4b-it \
    --trained-model /path/to/trained/output \
    --output-dir /path/to/merged
```

This combines:
- **Trained language model weights** from your output checkpoint
- **Original vision tower + projector weights** from the base multimodal model

The merged checkpoint can be loaded as `Gemma3ForConditionalGeneration` for multimodal inference or further training.
