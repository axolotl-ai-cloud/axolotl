# Pretraining / Continual Pretraining — Agent Reference

Train on raw text with no input masking. Two approaches depending on dataset size.

## When to Use

- Continual pretraining on domain-specific corpora
- Adapting a base model to a new language or domain before fine-tuning
- Pretraining-style data where the entire text is the training signal

## Choosing an Approach

| | Non-streaming (`type: completion`) | Streaming (`pretraining_dataset`) |
|---|---|---|
| **Dataset size** | Fits in memory | Too large to fit in memory |
| **Tokenization** | Pre-tokenized before training | On-demand during training |
| **Config key** | `datasets:` | `pretraining_dataset:` |
| **Long text handling** | Splits texts exceeding `sequence_len` | Concatenates into fixed-length sequences |
| **Benefit** | Can preprocess on CPU, transfer to GPU | Start training immediately, no preprocessing |

## Non-Streaming: `type: completion`

For smaller datasets that fit in memory. Pre-tokenizes the entire dataset.

```yaml
datasets:
  - path: my_corpus
    type: completion
    # field: text              # Column name (default: "text")
```

## Streaming: `pretraining_dataset`

For large corpora. Streams data on-demand without loading everything into memory.

```yaml
pretraining_dataset:
  - path: HuggingFaceFW/fineweb-edu
    type: pretrain
    text_column: text
    split: train

max_steps: 1000                          # Required — axolotl can't infer dataset size
streaming_multipack_buffer_size: 10000   # Buffer for sample packing
pretrain_multipack_attn: true            # Prevent cross-attention between packed samples
```

`max_steps` is required for streaming — one step = `sequence_len * micro_batch_size * gradient_accumulation_steps * num_gpus` tokens.

Full streaming docs: [streaming.qmd](../streaming.qmd)

## Dataset Format

```json
{"text": "The complete document text goes here."}
```

## Key Settings

- `sample_packing: true` + `pad_to_sequence_len: true` — pack documents into fixed-length sequences
- `flash_attention: true` — required for sample packing
- No adapter — typically full fine-tune for pretraining
- `train_on_inputs: true` — default for completion (all tokens trained on)

## File Map

```
src/axolotl/
  prompt_strategies/completion.py    # Non-streaming: completion prompt strategy (no masking)
  utils/data/sft.py                  # Non-streaming: dataset loading and processing
  utils/data/streaming.py            # Streaming: encode_streaming(), wrap_streaming_dataset()
  utils/schemas/config.py            # Config fields: pretraining_dataset, pretrain_multipack_attn, etc.

examples/streaming/pretrain.yaml     # Full streaming pretraining example config
```
