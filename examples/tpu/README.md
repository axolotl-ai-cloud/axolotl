# TPU examples

Single-chip fine-tuning on Google Cloud TPUs with [TorchTPU](https://github.com/google-pytorch/torch_tpu).

## Setup

On a TPU VM (v5e, v5p, v6e or v7x) with Python 3.12+:

```bash
pip install "axolotl[tpu]" --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

```bash
axolotl fetch examples
axolotl train examples/tpu/qwen3-0.6b-lora.yml --launcher python
axolotl train examples/tpu/qwen3-0.6b-fft.yml --launcher python
```

Multi-chip training is not supported yet. See [docs/tpu.qmd](../../docs/tpu.qmd) for details.

| File | Description |
|---|---|
| `qwen3-0.6b-lora.yml` | LoRA on Qwen3-0.6B |
| `qwen3-0.6b-fft.yml` | Full fine-tune of Qwen3-0.6B |
