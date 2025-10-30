# Mistral Small 3.1/3.2 Fine-tuning

This guide covers fine-tuning [Mistral Small 3.1](mistralai/Mistral-Small-3.1-24B-Instruct-2503) and [Mistral Small 3.2](mistralai/Mistral-Small-3.2-24B-Instruct-2506) with vision capabilities using Axolotl.

## Prerequisites

Before starting, ensure you have:
- Installed Axolotl (see [Installation docs](https://docs.axolotl.ai/docs/installation.html))

## Getting Started

1. Install the required vision lib:
    ```bash
    pip install 'mistral-common[opencv]==1.8.5'
    ```

2. Download the example dataset image:
   ```bash
   wget https://huggingface.co/datasets/Nanobit/text-vision-2k-test/resolve/main/African_elephant.jpg
   ```

3. Run the fine-tuning:
   ```bash
   axolotl train examples/mistral/mistral-small/mistral-small-3.1-24B-lora.yml
   ```

This config uses about 29.4 GiB VRAM.

## Dataset Format

The vision model requires multi-modal dataset format as documented [here](https://docs.axolotl.ai/docs/multimodal.html#dataset-format).

One exception is that, passing `"image": PIL.Image` is not supported. MistralTokenizer only supports `path`, `url`, and `base64` for now.

Example:
```json
{
    "messages": [
        {"role": "system", "content": [{ "type": "text", "text": "{SYSTEM_PROMPT}"}]},
        {"role": "user", "content": [
            { "type": "text", "text": "What's in this image?"},
            {"type": "image", "path": "path/to/image.jpg" }
        ]},
        {"role": "assistant", "content": [{ "type": "text", "text": "..." }]},
    ],
}
```

## Limitations

- Sample Packing is not supported for multi-modality training currently.
