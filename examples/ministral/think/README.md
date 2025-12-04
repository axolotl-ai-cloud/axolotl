# Ministral3 2512 Thinking Fine-tuning

This guide covers fine-tuning [Ministral3 2512](https://huggingface.co/collections/mistralai/ministral-3) with thinking capabilities using Axolotl. The thinking model enables explicit Chain-of-Thought reasoning with separate thinking and response sections.

Thanks to the team at MistralAI for giving us early access to prepare for these releases.

## Prerequisites

Before starting, ensure you have:
- Installed Axolotl (see [main README](../README.md))

## Getting Started

1. Install transformers v5

    ```bash
    pip install transformers==5.0.0rc0
    ```

    Note: This is still experimental in Axolotl. Other stuff may break.

2. Upgrade `mistral-common`

    ```bash
    pip install mistral-common==1.8.6
    ```

3. Swap to the Axolotl transformers v5 branch

    ```bash
    # copy examples/ministral/think/ministral3-small-think-qlora.yaml somewhere
    cp examples/ministral/think/ministral3-small-think-qlora.yaml ministral3-small-think-qlora.yaml

    git fetch
    git checkout transformers-v5
    ```

4. Run the thinking model fine-tuning:

    ```bash
    axolotl train ministral3-small-think-qlora.yaml
    ```

This config uses about 4.76 GiB VRAM.

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
