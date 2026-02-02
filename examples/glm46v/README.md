# Finetune GLM-4.6V with Axolotl

GLM-4.6V is a family of vision-language models from ZhipuAI found on [HuggingFace](https://huggingface.co/zai-org/GLM-4.6V). This guide shows how to fine-tune it with Axolotl for vision-language tasks.



## Getting started

1. Install Axolotl from source following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.


3. Run the fine-tuning:

    glm-4-6v-flash(9B)
    ```bash
    axolotl train examples/glm46v/glm-4-6v-flash-qlora.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

## Tips

- The configs use **`processor_type: AutoProcessor`** to automatically handle vision inputs.
- Vision datasets should follow the **vision data format** described in the [multimodal docs (Vision section)](https://docs.axolotl.ai/docs/multimodal.html#vision-data-format), similar to the `ministral3/vision` examples.
- The dataset format is based on the **OpenAI Messages** schema with image content – see the [conversation format docs](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template) for details.
- You can run a **full finetuning** by removing the `adapter: qlora` and `load_in_4bit: true` from the config.
- Read more on how to load your own dataset in the [dataset loading docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Supported Models

- **GLM-4.6V**: Full vision-language model (`zai-org/GLM-4.6V`)
- **GLM-4.6V-Flash**: Faster variant (`zai-org/GLM-4.6V-Flash`)

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [ZhipuAI GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
