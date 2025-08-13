# Finetune GLM4.5 with Axolotl

[UNSTABLE]

```bash
# LoRA SFT (4xH200 @ 84GB/GPU)
axolotl train examples/glm45/glm4.5-lora-fsdp2.yaml

# FFT SFT (4xH200)
# Checkpointing error on backward pass
# Without checkpointing => OOM
axolotl train examples/glm45/glm4.5-fft-fsdp2.yaml
```

## Dataset

In addition to normal OpenAI Messages format, GLM4.5 support an extra parameter for thinking in assistant section.

```json
{
    "role": "assistant",
    "reasoning_content": "...",  // or have </think>...</think> in `content`
    "content": "...",
}
```

Note:
- The role name for tools in this template is `tool`.
- You will see this Axolotl WARNING. This is to be as expected as the template does not use EOS.
```bash
EOS token '<|endoftext|>' not found in chat_template. Please check if your template/EOS token is correct.
```
- Make sure you set the below extra attributes if needed
```yaml
datasets:
  - path: ...
    type: chat_template
    message_property_mappings:
      role: role
      content: content

    #   tool_calls: tool_calls  # uncomment if using tools
    #   reasoning_content: reasoning_content  # uncomment if have reasoning

# Uncomment if training on tool role (you would rarely if ever need this)
# eot_tokens:
#   - <|observation|>
```
