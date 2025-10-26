# Magistral Small Thinking Fine-tuning

This guide covers fine-tuning [Magistral Small 2507](https://huggingface.co/mistralai/Magistral-Small-2507) with thinking capabilities using Axolotl. The thinking model enables explicit Chain-of-Thought reasoning with separate thinking and response sections.

## Prerequisites

Before starting, ensure you have:
- Installed Axolotl (see [main README](../README.md))

## Getting Started

Run the thinking model fine-tuning:

```bash
axolotl train examples/magistral/think/magistral-small-think-qlora.yaml
```

This config uses about 19.1 GiB VRAM.

### Tips

- Dataset uses multi-content format with `type: thinking` support. See [Dataset Format](#dataset-format) below.
- You cannot mix `content: str` and `content: list[dict]`, otherwise, dataset loading will fail. Keep it consistent.

## Dataset Format

The thinking model requires the multi-content dataset format with support for an extra `role: thinking` within system and assistant messages.

Example format:

```json
{
    "messages": [
        {
            "role": "system",
            "content": [
                { "type": "text", "text": "{SYSTEM_PROMPT}"}
            ]
        },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "Solve this step by step: What is 15% of 240?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I need to calculate 15% of 240. First, I'll convert 15% to decimal: 0.15. Then multiply: 0.15 × 240 = 36."
                },
                {
                    "type": "text",
                    "text": "To find 15% of 240, I'll multiply 240 by 0.15:\n\n240 × 0.15 = 36\n\nTherefore, 15% of 240 is 36."
                }
            ]
        }
    ]
}
```

### Advanced Options

The `thinking` section supports an optional `closed` parameter:

```json
{
    "type": "thinking",
    "thinking": "Internal reasoning here...",
    "closed": true  // Default: true, controls adding the closing [/THINK] tag
}
```
