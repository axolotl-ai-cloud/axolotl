# Finetune Muse Glimmer with Axolotl

Muse Glimmer is a 30B open agentic model from Meta Superintelligence Labs, found on [HuggingFace](https://huggingface.co/meta-models/Muse-Glimmer-30B). It pairs a dense text decoder with a frozen ViT-G/14 perception encoder, and is built to run agent workloads locally on a single consumer GPU.

The architecture is unusual in a few ways that matter when you fine-tune it: attention repeats a `[local, local, local, global]` pattern with a 2048 sliding window, RoPE is applied to the local layers only (the global layers are NoPE), attention output is sigmoid-gated, and the logits are scaled and `tanh`-softcapped after the LM head.

## Getting started

1. Install Axolotl from source following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage. Both configs enable it.

3. Run the fine-tuning:

    ```bash
    # QLoRA, language model only (~27 GiB VRAM)
    axolotl train examples/muse-glimmer/qlora.yaml

    # QLoRA, language model + vision tower (~23 GiB VRAM)
    axolotl train examples/muse-glimmer/qlora-vision.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### Tips

- Fused LoRA kernels are not available for this architecture.
- We added custom Liger Kernel support for RMSNorm, the SwiGLU MLP, RoPE, and the vision tower's LayerNorms.
- Reasoning strength is set in the system prompt (`low` / `medium` / `high` / `xhigh`, defaulting to `high`). Keep it consistent between training and inference.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Dataset Format

The configs use the OpenAI Messages format as described [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template). Image parts use `{"type": "image"}`, not `image_url`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "What is in this image?"}
      ]
    },
    {
      "role": "assistant",
      "content": "A cat sitting on a windowsill."
    }
  ]
}
```

To train reasoning traces, add `reasoning_content` alongside `content` on the assistant message:

```json
{
  "role": "assistant",
  "reasoning_content": "The animal has pointed ears and whiskers.",
  "content": "A cat sitting on a windowsill."
}
```

## Tool calling

Tool definitions live in their own dataset column, not in the messages. The default column
name is `tools` (override with `field_tools`), and it holds a list of
[JSON schema](https://json-schema.org/learn/getting-started-step-by-step) function definitions:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "City name"}},
          "required": ["city"]
        }
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
      "role": "assistant",
      "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
      ]
    },
    {"role": "tool", "tool_call_id": "c1", "content": "18C, cloudy"},
    {"role": "assistant", "content": "It's 18C and cloudy in Paris."}
  ]
}
```

The template turns that into Harmony turns with an ATEM XML call block, and adds each tool
namespace to the system block's valid-recipient list:

```
# Valid recipients: "self", "get_weather.*", "user".<|eot|>
<|start|>user<|message|>What's the weather in Paris?<|eot|>
<|start|>assistant to=get_weather<|message|><atem:function_calls>
<atem:invoke name="get_weather">
<atem:parameter name="city">Paris</atem:parameter>
</atem:invoke>
</atem:function_calls><|eot|>
<|start|>tool get_weather<|message|><tool_output name="get_weather">
18C, cloudy
</tool_output><|eot|>
<|start|>assistant to=user<|message|>It's 18C and cloudy in Paris.<|eot|>
```

Two things to watch:

- **`arguments` must be a dict, not a JSON string.** The template calls `raise_exception` on a
  string, because the Jinja sandbox cannot parse one. `{"city": "Paris"}`, never `"{\"city\": \"Paris\"}"`.
- **Use a text-only config for tool data.** The multimodal collator calls `apply_chat_template`
  without passing `tools`, so the tool definitions never reach the system block. Drop
  `processor_type` and `skip_prepare_dataset` so the `chat_template` strategy handles it
  and `field_tools` is read. That path needs `eot_tokens: ["<|eot|>", "<|eom|>"]`, since
  neither terminator is the tokenizer's `eos_token`.

The assistant tool-call turn is trained like any other assistant turn; only the `tool` role's
`<tool_output>` is masked, since that is input the model receives rather than text it writes.

## Related Resources

- [Muse Glimmer on HuggingFace](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [Meta AI Research announcement](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model)
- [Multimodal docs](https://docs.axolotl.ai/docs/multimodal.html)
