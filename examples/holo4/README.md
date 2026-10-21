# Finetune H Company's Holo4 with Axolotl

[Holo4-35B-A3B](https://huggingface.co/Hcompany/Holo4-35B-A3B) is a vision-language model by H Company for computer-use agents, built on Qwen3.6-35B-A3B.

This guide shows how to fine-tune it with Axolotl on screenshot grounding data: given a screenshot and an element description, the model predicts where to click.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage.

3. Convert one shard (500 rows) of [AgentNet clicks](https://huggingface.co/datasets/mlfoundations-cua-dev/agentnet-clicks) to Holo4's element-localization format. This writes `holo4-clicks.parquet` to the current directory:

    ```bash
    python examples/holo4/prepare_clicks.py
    ```

4. Run the finetuning example:

    ```bash
    axolotl train examples/holo4/35b-a3b-lora-vision.yaml
    ```

This is a bf16 LoRA on a 35B model, so the weights alone take about 70 GB. Plan on at least one 80 GB GPU. Peak VRAM has not been measured.

Let us know how it goes. Happy finetuning! 🚀

### TIPS

- `prepare_clicks.py` follows Holo4's [element localization](https://hub.hcompany.ai/models-api/element-localization) format. The user turn holds the screenshot, then the localization prompt and target. The answer is JSON `{"x": ..., "y": ...}`, with both values in `[0, 1000]` relative to the image size. At inference, use the same prompt with thinking off (`enable_thinking: False`) and `temperature=0.0`.
- The AgentNet clicks dataset card sets no license of its own and refers to the original [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) licenses. AgentNet is MIT licensed. Check those terms before you use the data.
- To train on more data, add more shards to `data_files` in `prepare_clicks.py`, or convert your own screenshots to the same format.
- Vision configs set `skip_prepare_dataset: true`, so samples are tokenized in the collator at train time. `axolotl preprocess` only validates the config and exits without tokenizing, so `--debug` shows no samples.
- The dataset uses the OpenAI Messages format with an `images` column, see [Multimodal Formats](https://docs.axolotl.ai/docs/multimodal.html#dataset-format). Each `{"type": "image"}` placeholder is filled with the image from that column.
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
