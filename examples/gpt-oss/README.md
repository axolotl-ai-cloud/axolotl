# Finetune OpenAI's GPT-OSS with Axolotl

[GPT-OSS](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) are a family of open-weight MoE models trained by OpenAI, released in August 2025. There are two variants: 20B and 120B.

This guide shows how to fine-tune it with Axolotl with multi-turn conversations and proper masking.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

    Here is an example of how to install from pip:

```bash
# Ensure you have Pytorch installed (Pytorch 2.6.0 min)
pip3 install packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation 'axolotl[flash-attn]>=0.12.0'
```

2. Choose one of the following configs below for training the 20B model. (for 120B, see [below](#training-120b))

```bash
# LoRA SFT linear layers (1x48GB @ ~44GiB)
axolotl train examples/gpt-oss/gpt-oss-20b-sft-lora-singlegpu.yaml

# FFT SFT with offloading (2x24GB @ ~21GiB/GPU)
axolotl train examples/gpt-oss/gpt-oss-20b-fft-fsdp2-offload.yaml

# FFT SFT (8x48GB @ ~36GiB/GPU or 4x80GB @ ~46GiB/GPU)
axolotl train examples/gpt-oss/gpt-oss-20b-fft-fsdp2.yaml
```

Note: Memory usage taken from `device_mem_reserved(gib)` from logs.

### Training 120B

On 8xH100s, make sure you have ~3TB of free disk space. With each checkpoint clocking in at ~720GB, along with the base
model, and final model output, you may need at least 3TB of free disk space to keep at least 2 checkpoints.

```bash
# FFT SFT with offloading (8x80GB @ ~49GiB/GPU)
axolotl train examples/gpt-oss/gpt-oss-120b-fft-fsdp2-offload.yaml
```

ERRATA: Transformers saves the model Architecture prefixed with `FSDP` which needs to be manually renamed in `config.json`.
See https://github.com/huggingface/transformers/pull/40207 for the status of this issue.

```bash
sed -i 's/FSDPGptOssForCausalLM/GptOssForCausalLM/g' ./outputs/gpt-oss-out/config.json
```

When using SHARDED_STATE_DICT with FSDP, the final checkpoint should automatically merge the sharded weights to your
configured `output_dir`. However, if that step fails due to a disk space error, you can take an additional step to
merge the sharded weights.  This step will automatically determine the last checkpoint directory and merge the sharded
weights to `{output_dir}/merged`.

```bash
axolotl merge-sharded-fsdp-weights examples/gpt-oss/gpt-oss-120b-fft-fsdp2-offload.yaml
mv ./outputs/gpt-oss-out/merged/* ./outputs/gpt-oss-out/
```


### Inferencing your fine-tuned model

GPT-OSS support in vLLM does not exist in a stable release yet. See https://x.com/MaziyarPanahi/status/1955741905515323425
for more information about using a special vllm-openai docker image for inferencing with vLLM.

SGLang has 0-day support in main, see https://github.com/sgl-project/sglang/issues/8833 for infomation on installing
SGLang from source. Once you've installed SGLang, run the following command to launch a SGLang server:

```bash
python3 -m sglang.launch_server --model ./outputs/gpt-oss-out/ --served-model-name axolotl/gpt-oss-120b --host 0.0.0.0 --port 8888 --tp 8
```

### Tool use

GPT-OSS has a comprehensive tool understanding. Axolotl supports tool calling datasets for Supervised Fine-tuning.

Here is an example dataset config:
```yaml
datasets:
  - path: Nanobit/text-tools-2k-test
    type: chat_template
```

See [Nanobit/text-tools-2k-test](https://huggingface.co/datasets/Nanobit/text-tools-2k-test) for the sample dataset.

Refer to [our docs](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#using-tool-use) for more info.

### TIPS

- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).
- The dataset format follows the OpenAI Messages format as seen [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template).

## Optimization Guides

- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)

## Related Resources

- [GPT-OSS Blog](https://openai.com/index/introducing-gpt-oss/)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Website](https://axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
