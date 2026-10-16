# Finetune H Company's Holo4 with Axolotl

[Holo4-35B-A3B](https://huggingface.co/Hcompany/Holo4-35B-A3B) is a vision-language model by H Company for computer-use agents, built on Qwen3.6-35B-A3B.

This guide shows how to fine-tune it with Axolotl on screenshot grounding data to build your own computer-use agent.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Run the finetuning example:

    ```bash
    axolotl train examples/holo4/35b-a3b-lora-vision.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- For inference hyp, please see the model card details.
- The example trains on one shard of [AgentNet clicks](https://huggingface.co/datasets/mlfoundations-cua-dev/agentnet-clicks) (screenshot + instruction -> `(x,y)` click point). Add more shards to `data_files`, or swap in your own agent trajectories.
- The dataset format follows the OpenAI Messages format with an `images` column, see [Multimodal Formats](https://docs.axolotl.ai/docs/multimodal.html#dataset-format). Axolotl inserts the screenshot into the first user turn.
- Read more on loading your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- To also train the routed experts, see the MoE notes in the [Qwen3.5 guide](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/qwen3.5#routed-experts-moe).

## Optimization Guides

Please check the [Optimizations doc](https://docs.axolotl.ai/docs/optimizations.html).

## Related Resources

- [Holo4 Blog](https://hcompany.ai/newsroom/holo4)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
