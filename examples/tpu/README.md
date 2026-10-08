# TPU examples

Single-host data-parallel training on Google Cloud TPUs.

## Setup

```bash
pip install "axolotl[tpu]" -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Usage

```bash
PJRT_DEVICE=TPU axolotl train examples/tpu/llama-3-1b-lora.yml
PJRT_DEVICE=TPU axolotl train examples/tpu/llama-3-1b-fft.yml
```

See [docs/tpu.qmd](../../docs/tpu.qmd) for the full guide.

## Files

| File | Description |
|---|---|
| `llama-3-1b-lora.yml` | LoRA on Llama-3.2-1B-Instruct |
| `llama-3-1b-fft.yml` | Full fine-tune on Llama-3.2-1B-Instruct |
