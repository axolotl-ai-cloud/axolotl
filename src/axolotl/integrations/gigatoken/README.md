# Gigatoken

Fast CPU tokenization for the pretraining/completion path, backed by
[gigatoken](https://github.com/marcelroed/gigatoken) — a SIMD BPE tokenizer that
is a drop-in for HuggingFace tokenizers.

The plugin attaches a gigatoken-accelerated encoder to the loaded tokenizer via
the `post_tokenizer_load` hook. Only the streaming pretraining/completion encode
path (`encode_streaming`) uses it; the tokenizer itself is left untouched, so
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
- Not every tokenizer is supported by gigatoken's byte remapping. If construction
  fails for your model, the run errors out at tokenizer load rather than silently
  falling back — verify token parity before relying on it.
- Benchmark against the `dataset_num_proc` baseline for your corpus with
  `scripts/bench_gigatoken.py`.
