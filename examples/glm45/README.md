# Finetune Z.ai's GLM-4.5-Air with Axolotl

[GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) is a MoE model by Z.ai.

This guide shows how to fine-tune it with Axolotl.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

```bash
# QLoRA (1x80GB @ ~63.4GiB/GPU)
axolotl train examples/glm45/glm-45-air-qlora.yaml
```

### Dataset

In addition to the standard OpenAI Messages format, GLM-4.5 supports an extra parameter for thinking in the assistant section.

```json
{
    "role": "assistant",
    "reasoning_content": "...",  // or have </think>...</think> in `content`
    "content": "..."
}
```

Make sure you set the below extra attributes if needed:

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

### Tips

- The role name for tools in this template is `tool`.
- You will see this Axolotl WARNING — this is expected as the template does not use EOS:
  ```
  EOS token '<|endoftext|>' not found in chat_template. Please check if your template/EOS token is correct.
  ```
- You can run a full finetuning by removing `adapter: qlora`, `load_in_4bit: true`, and `quantize_moe_experts: true` from the config.
- **LoRA kernels**: Incompatible with this model. Must be explicitly disabled (`lora_*_kernel: false`).
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

### ⚠️ GGUF / llama.cpp loading error (missing tensors)

If you encounter an error like:

```
llama_model_load: error loading model: missing tensor 'blk.X.attn_norm.weight'
```

when converting or loading a GLM-4 / GLM4-MoE model in `llama.cpp`, this is likely caused by a mismatch between the model config and exported weights.

Some GLM-4 configs include:

```json
"num_nextn_predict_layers": = 1
```

This enables MTP (multi-token prediction) layers, but these layers are not always included during PEFT/QLoRA training or GGUF conversion.

#### ✅ Fix

Before converting to GGUF, set:

```json
"num_nextn_predict_layers": 0
```

in your `config.json`, then reconvert the model.

#### Why this happens

* The config enables MTP layers
* The actual weights do not include them
* This causes tensor misalignment during loading in `llama.cpp`

This is an upstream configuration mismatch rather than an Axolotl-specific issue.


## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [GLM-4.5-Air on HuggingFace](https://huggingface.co/zai-org/GLM-4.5-Air)
- [GLM-4.5 Blog](https://z.ai/blog/glm-4.5)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
