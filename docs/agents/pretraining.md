# Pretraining / Continual Pretraining — Agent Reference

Train on raw text with no input masking. For general config guidance, see [getting-started.qmd](../getting-started.qmd).

## When to Use

- Continual pretraining on domain-specific corpora
- Adapting a base model to a new language or domain before fine-tuning
- Pretraining-style data where the entire text is the training signal

## Config

Use `type: completion` — trains on the entire text with no masking.

```yaml
base_model: meta-llama/Llama-3.1-8B
datasets:
  - path: my_corpus
    type: completion
    # field: text                # Column name (default: "text")
```

Dataset format:
```json
{"text": "The complete document text goes here."}
```

Key settings for pretraining:
- `train_on_inputs: true` (default for completion — all tokens are trained on)
- `sample_packing: true` + `pad_to_sequence_len: true` (pack documents into fixed-length sequences)
- `flash_attention: true` (required for sample packing)
- No adapter — typically full fine-tune for pretraining

## File Map

```
src/axolotl/
  prompt_strategies/completion.py    # Completion prompt strategy (no masking)
  utils/data/sft.py                  # Dataset loading and processing
```
