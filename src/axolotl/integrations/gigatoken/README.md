# Gigatoken

Fast CPU tokenization for the pretraining/completion path, backed by
[gigatoken](https://github.com/marcelroed/gigatoken) — a SIMD BPE tokenizer that
is a drop-in for HuggingFace tokenizers.

The plugin attaches a gigatoken encoder to the loaded tokenizer via the
`post_tokenizer_load` hook. Only raw-text encoding uses it; the tokenizer itself
is left untouched, so chat-template and other prompt strategies are unaffected.

## Installation

```bash
pip install axolotl[gigatoken]
```

## Usage

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

- Only the tokenization phase is accelerated; training is unaffected.
- Documents long enough to need HuggingFace's strided overflow chunking fall back
  to the HuggingFace tokenizer for that batch.
- Not every vocabulary is supported by gigatoken's byte remapping (SmolLM2, for
  instance, is not). Unsupported tokenizers error out at load rather than
  silently falling back.
