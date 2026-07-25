# Gigatoken

Fast CPU tokenization for the pretraining/completion path, backed by
[gigatoken](https://github.com/marcelroed/gigatoken) — a SIMD BPE tokenizer that
is a drop-in for HuggingFace tokenizers.

The plugin attaches a gigatoken-accelerated encoder to the loaded tokenizer via
the `post_tokenizer_load` hook. Only the raw-text pretraining/completion encode
paths use it — `encode_streaming` (unpacked) and `PretrainTokenizationStrategy`
(`sample_packing: true`). The tokenizer itself is left untouched, so
chat-template and other prompt strategies keep the full HuggingFace API.

## Installation

```bash
pip install gigatoken
```

## Usage

Adding the plugin enables gigatoken by default:

```yaml
plugins:
  - axolotl.integrations.gigatoken.GigatokenPlugin

pretraining_dataset:
  - path: allenai/c4
    name: en
    type: pretrain
    split: train
```

Set `gigatoken: false` to disable without removing the plugin.

## Notes

- gigatoken accelerates **tokenization only** (the `axolotl preprocess` / dataset
  encoding phase); it does not affect the training step.
- Only raw-text encoding is accelerated. Chat-template and other prompt strategies
  keep using the HuggingFace tokenizer.
- Documents long enough to need HuggingFace's strided overflow chunking
  (`sequence_len * 64` tokens) fall back to the HuggingFace tokenizer for that
  batch, since gigatoken ignores `return_overflowing_tokens` rather than rejecting
  it.
- Not every tokenizer is supported by gigatoken's byte remapping (SmolLM2, for
  instance, is not). Tokenizers that can't be wrapped, or that disagree with the
  HuggingFace tokenizer on a parity check, error out at tokenizer load rather than
  silently falling back.
- Benchmark against the `dataset_num_proc` baseline for your corpus with
  `scripts/bench_gigatoken.py`.
