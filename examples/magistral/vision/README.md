# Magistral Small Vision Fine-tuning

This guide covers fine-tuning [Magistral Small 2509](https://huggingface.co/mistralai/Magistral-Small-2509) with vision capabilities using Axolotl.

## Prerequisites

Before starting, ensure you have:
- Installed Axolotl from source (see [main README](../README.md#getting-started))

## Getting started

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
   axolotl train magistral-small-vision-24B-qlora.yml
   ```

This config uses about 17GiB VRAM.

WARNING: The loss and grad norm will be much higher than normal at first. We suspect this to be inherent to the model as of the moment. If anyone would like to submit a fix for this, we are happy to take a look.

### Tips

Key differences from text-only model:
- `max_tokens: 131072` for inference
- Multi-modal dataset format required
- Sample packing not supported

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
