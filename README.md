<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<p align="center">
    <img src="https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue" alt="GitHub License">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg" alt="tests">
    <a href="https://github.com/axolotl-ai-cloud/axolotl/releases"><img src="https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg" alt="Releases"></a>
    <br/>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square" alt="contributors" style="height: 20px;"></a>
    <img src="https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl" alt="GitHub Repo stars">
    <br/>
    <a href="https://discord.com/invite/HhrNrHJPRb"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;"></a>
    <a href="https://twitter.com/axolotl_ai"><img src="https://img.shields.io/twitter/follow/axolotl_ai?style=social" alt="twitter" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>

Axolotl is a tool designed to streamline post-training for various AI models.
Post-training refers to any modifications or additional training performed on
pre-trained models - including full model fine-tuning, parameter-efficient tuning (like
LoRA and QLoRA), supervised fine-tuning (SFT), instruction tuning, and alignment
techniques. With support for multiple model architectures and training configurations,
Axolotl makes it easy to get started with these techniques.

Axolotl is designed to work with YAML config files that contain everything you need to
preprocess a dataset, train or fine-tune a model, run model inference or evaluation,
and much more.

Features:

- Train various Huggingface models such as llama, pythia, falcon, mpt
- Supports fullfinetune, lora, qlora, relora, and gptq
- Customize configurations using a simple yaml file or CLI overwrite
- Load different dataset formats, use custom formats, or bring your own tokenized datasets
- Integrated with [xformers](https://github.com/facebookresearch/xformers), flash attention, [liger kernel](https://github.com/linkedin/Liger-Kernel), rope scaling, and multipacking
- Works with single GPU or multiple GPUs via FSDP or Deepspeed
- Easily run with Docker locally or on the cloud
- Log results and optionally checkpoints to wandb, mlflow or Comet
- And more!

## 🚀 Quick Start

**Requirements**:

- NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
- Python 3.11
- PyTorch ≥2.4.1

### Installation

```bash
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

Other installation approaches are described [here](https://axolotl-ai-cloud.github.io/axolotl/docs/installation.html).

### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

That's it! Check out our [Getting Started Guide](https://axolotl-ai-cloud.github.io/axolotl/docs/getting-started.html) for a more detailed walkthrough.

## ✨ Key Features

- **Multiple Model Support**: Train various models like LLaMA, Mistral, Mixtral, Pythia, and more
- **Training Methods**: Full fine-tuning, LoRA, QLoRA, and more
- **Easy Configuration**: Simple YAML files to control your training setup
- **Performance Optimizations**: Flash Attention, xformers, multi-GPU training
- **Flexible Dataset Handling**: Use various formats and custom datasets
- **Cloud Ready**: Run on cloud platforms or local hardware

## 📚 Documentation

- [Installation Options](https://axolotl-ai-cloud.github.io/axolotl/docs/installation.html) - Detailed setup instructions for different environments
- [Configuration Guide](https://axolotl-ai-cloud.github.io/axolotl/docs/config.html) - Full configuration options and examples
- [Dataset Guide](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/) - Supported formats and how to use them
- [Multi-GPU Training](https://axolotl-ai-cloud.github.io/axolotl/docs/multi-gpu.html)
- [Multi-Node Training](https://axolotl-ai-cloud.github.io/axolotl/docs/multi-node.html)
- [Multipacking](https://axolotl-ai-cloud.github.io/axolotl/docs/multipack.html)
- [API Reference](https://axolotl-ai-cloud.github.io/axolotl/docs/api/) - Auto-generated code documentation
- [FAQ](https://axolotl-ai-cloud.github.io/axolotl/docs/faq.html) - Frequently asked questions

## 🤝 Getting Help

- Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
- Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
- Read our [Debugging Guide](https://axolotl-ai-cloud.github.io/axolotl/docs/debugging.html)
- Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) for options

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## Supported Models

|             | fp16/fp32 | lora | qlora | gptq | gptq w/flash attn | flash attn | xformers attn |
|-------------|:----------|:-----|-------|------|-------------------|------------|--------------|
| llama       | ✅         | ✅    | ✅     | ✅             | ✅                 | ✅          | ✅            |
| Mistral     | ✅         | ✅    | ✅     | ✅             | ✅                 | ✅          | ✅            |
| Mixtral-MoE | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Mixtral8X22 | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Pythia      | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| cerebras    | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| btlm        | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| mpt         | ✅         | ❌    | ❓     | ❌             | ❌                 | ❌          | ❓            |
| falcon      | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| gpt-j       | ✅         | ✅    | ✅     | ❌             | ❌                 | ❓          | ❓            |
| XGen        | ✅         | ❓    | ✅     | ❓             | ❓                 | ❓          | ✅            |
| phi         | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| RWKV        | ✅         | ❓    | ❓     | ❓             | ❓                 | ❓          | ❓            |
| Qwen        | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Gemma       | ✅         | ✅    | ✅     | ❓             | ❓                 | ✅          | ❓            |
| Jamba       | ✅         | ✅    | ✅     | ❓             | ❓                 | ✅          | ❓            |

✅: supported
❌: not supported
❓: untested

## ❤️ Sponsors

Thank you to our sponsors who help make Axolotl possible:

- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl) - Modal lets you run
jobs in the cloud, by just writing a few lines of Python. Customers use Modal to deploy Gen AI models at large scale,
fine-tune large language models, run protein folding simulations, and much more.

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
