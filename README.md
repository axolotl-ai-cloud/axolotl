<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>
  <p align="center">
      <strong>A Free and Open Source LLM Fine-tuning Framework</strong><br>
  </p>

<p align="center">
    <img src="https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue" alt="GitHub License">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg" alt="tests">
    <a href="https://codecov.io/gh/axolotl-ai-cloud/axolotl"><img src="https://codecov.io/gh/axolotl-ai-cloud/axolotl/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/releases"><img src="https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg" alt="Releases"></a>
    <br/>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square" alt="contributors" style="height: 20px;"></a>
    <img src="https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl" alt="GitHub Repo stars">
    <br/>
    <a href="https://discord.com/invite/HhrNrHJPRb"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;"></a>
    <a href="https://twitter.com/axolotl_ai"><img src="https://img.shields.io/twitter/follow/axolotl_ai?style=social" alt="twitter" style="height: 20px;"></a>
    <a href="https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google-colab" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>


## 🎉 Latest Updates

- 2025/12: Axolotl now includes support for [Kimi-Linear](https://docs.axolotl.ai/docs/models/kimi-linear.html), [Plano-Orchestrator](https://docs.axolotl.ai/docs/models/plano.html), [MiMo](https://docs.axolotl.ai/docs/models/mimo.html), [InternVL 3.5](https://docs.axolotl.ai/docs/models/internvl3_5.html), [Olmo3](https://docs.axolotl.ai/docs/models/olmo3.html), [Trinity](https://docs.axolotl.ai/docs/models/trinity.html), and [Ministral3](https://docs.axolotl.ai/docs/models/ministral3.html).
- 2025/10: New model support has been added in Axolotl for: [Qwen3 Next](https://docs.axolotl.ai/docs/models/qwen3-next.html), [Qwen2.5-vl, Qwen3-vl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/qwen2_5-vl), [Qwen3, Qwen3MoE](https://docs.axolotl.ai/docs/models/qwen3.html), [Granite 4](https://docs.axolotl.ai/docs/models/granite4.html), [HunYuan](https://docs.axolotl.ai/docs/models/hunyuan.html), [Magistral 2509](https://docs.axolotl.ai/docs/models/magistral/vision.html), [Apertus](https://docs.axolotl.ai/docs/models/apertus.html), and [Seed-OSS](https://docs.axolotl.ai/docs/models/seed-oss.html).
- 2025/09: Axolotl now has text diffusion training. Read more [here](https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/integrations/diffusion).
- 2025/08: QAT has been updated to include NVFP4 support. See [PR](https://github.com/axolotl-ai-cloud/axolotl/pull/3107).
- 2025/07:
  - ND Parallelism support has been added into Axolotl. Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. Check out the [blog post](https://huggingface.co/blog/accelerate-nd-parallel) for more info.
  - Axolotl adds more models: [GPT-OSS](https://docs.axolotl.ai/docs/models/gpt-oss.html), [Gemma 3n](https://docs.axolotl.ai/docs/models/gemma3n.html), [Liquid Foundation Model 2 (LFM2)](https://docs.axolotl.ai/docs/models/LiquidAI.html), and [Arcee Foundation Models (AFM)](https://docs.axolotl.ai/docs/models/arcee.html).
  - FP8 finetuning with fp8 gather op is now possible in Axolotl via `torchao`. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
  - [Voxtral](https://docs.axolotl.ai/docs/models/voxtral.html), [Magistral 1.1](https://docs.axolotl.ai/docs/models/magistral.html), and [Devstral](https://docs.axolotl.ai/docs/models/devstral.html) with mistral-common tokenizer support has been integrated in Axolotl!
  - TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP support has been added to support Arctic Long Sequence Training. (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for using ALST with Axolotl!
- 2025/05: Quantization Aware Training (QAT) support has been added to Axolotl. Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more!

<details>

<summary>Expand older updates</summary>

- 2025/03: Axolotl has implemented Sequence Parallelism (SP) support. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length when fine-tuning.
- 2025/06: Magistral with mistral-common tokenizer support has been added to Axolotl. See [docs](https://docs.axolotl.ai/docs/models/magistral.html) to start training your own Magistral models with Axolotl!
- 2025/04: Llama 4 support has been added in Axolotl. See [docs](https://docs.axolotl.ai/docs/models/llama-4.html) to start training your own Llama 4 models with Axolotl's linearized version!
- 2025/03: (Beta) Fine-tuning Multimodal models is now supported in Axolotl. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html) to fine-tune your own!
- 2025/02: Axolotl has added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html) to give it a try.
- 2025/02: Axolotl has added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code) and have some fun!
- 2025/01: Axolotl has added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

</details>

## ✨ Overview

Axolotl is a free and open-source tool designed to streamline post-training and fine-tuning for the latest large language models (LLMs).

Features:

- **Multiple Model Support**: Train various models like GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and many more models available on the Hugging Face Hub.
- **Multimodal Training**: Fine-tune vision-language models (VLMs) including LLaMA-Vision, Qwen2-VL, Pixtral, LLaVA, SmolVLM2, and audio models like Voxtral with image, video, and audio support.
- **Training Methods**: Full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
- **Easy Configuration**: Re-use a single YAML configuration file across the full fine-tuning pipeline: dataset preprocessing, training, evaluation, quantization, and inference.
- **Performance Optimizations**: [Multipacking](https://docs.axolotl.ai/docs/multipack.html), [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Xformers](https://github.com/facebookresearch/xformers), [Flex Attention](https://pytorch.org/blog/flexattention/), [Liger Kernel](https://github.com/linkedin/Liger-Kernel), [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy/tree/main), [Sequence Parallelism (SP)](https://docs.axolotl.ai/docs/sequence_parallelism.html), [LoRA optimizations](https://docs.axolotl.ai/docs/lora_optims.html), [Multi-GPU training (FSDP1, FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html), [Multi-node training (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html), and many more!
- **Flexible Dataset Handling**: Load from local, HuggingFace, and cloud (S3, Azure, GCP, OCI) datasets.
- **Cloud Ready**: We ship [Docker images](https://hub.docker.com/u/axolotlai) and also [PyPI packages](https://pypi.org/project/axolotl/) for use on cloud platforms and local hardware.



## 🚀 Quick Start - LLM Fine-tuning in Minutes

**Requirements**:

- NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
- Python 3.11
- PyTorch ≥2.8.0

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb#scrollTo=msOCO4NRmRLa)

### Installation

#### Using pip

```bash
pip3 install -U packaging==26.0 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

#### Using Docker

Installing with Docker can be less error prone than installing in your own environment.
```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Other installation approaches are described [here](https://docs.axolotl.ai/docs/installation.html).

#### Cloud Providers

<details>

- [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
- [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
- [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
- [Novita](https://novita.ai/gpus-console?templateId=311)
- [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
- [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

That's it! Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.


## 📚 Documentation

- [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions for different environments
- [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options and examples
- [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
- [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them
- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
- [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
- [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

## 🤝 Getting Help

- Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
- Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
- Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
- Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) for options

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## 📈 Telemetry

Axolotl has opt-out telemetry that helps us understand how the project is being used
and prioritize improvements. We collect basic system information, model types, and
error rates—never personal data or file paths. Telemetry is enabled by default. To
disable it, set AXOLOTL_DO_NOT_TRACK=1. For more details, see our [telemetry documentation](https://docs.axolotl.ai/docs/telemetry.html).

## ❤️ Sponsors

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

If you use Axolotl in your research or projects, please cite it as follows:

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
