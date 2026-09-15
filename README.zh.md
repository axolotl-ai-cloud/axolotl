<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>
  <p align="center">
      <strong>免费开源的大语言模型 (LLM) 微调框架</strong><br>
  </p>

<p align="center">
  <a href="https://github.com/axolotl-ai-cloud/axolotl/blob/main/README.md">English</a> · <b>中文</b>
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
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/docker-e2e.yml/badge.svg" alt="docker-e2e-tests">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>


## 🎉 最新动态

- 2026/08:
  - Axolotl 新增了对 [Ling 3.0](https://docs.axolotl.ai/docs/models/ling3.html)、[Muse Glimmer](https://docs.axolotl.ai/docs/models/muse-glimmer.html)、[North Micro Vision Instruct](https://docs.axolotl.ai/docs/models/cohere-north-micro-vision-instruct.html) 与 [Shieldstral](https://docs.axolotl.ai/docs/models/shieldstral.html) 的模型支持。
- 2026/07:
  - 现已通过 ScatterMoE (W4A16) 与 SonicMoE (W4A4) 支持 [NVFP4 (4-bit) MoE LoRA 训练](https://docs.axolotl.ai/docs/nvfp4_lora.html)，包括将 Adapter 合并回普通 NVFP4 检查点。
- 2026/06:
  - 通过 DeepEP 实现用于分布式 MoE 训练的[专家并行 (Expert Parallelism, EP)](https://docs.axolotl.ai/docs/nd_parallelism.html)、通过[兼容 Tinker 的 API](https://github.com/axolotl-ai-cloud/axolotl/pull/3614) 进行远程训练、[面向混合 SSM 模型的上下文并行 (Context Parallelism)](https://github.com/axolotl-ai-cloud/axolotl/pull/3572)（Nemotron-H、Falcon-H1、Bamba）、[BitNet 1.58-bit](https://github.com/axolotl-ai-cloud/axolotl/pull/3634) 微调，以及[多模态仅对 Assistant 回复计算 Loss 的掩码修复](https://github.com/axolotl-ai-cloud/axolotl/pull/3625)。
- 2026/04:
  - Axolotl 新增了对 [Mistral Medium 3.5](https://docs.axolotl.ai/docs/models/mistral-medium-3_5.html) 与 [Gemma 4](https://docs.axolotl.ai/docs/models/gemma4.html) 的模型支持。
  - 新的强化学习与 Kernel：[异步 GRPO (Async GRPO)](https://github.com/axolotl-ai-cloud/axolotl/pull/3486)（每步提速最高达 58%）、[Flash Attention 4](https://docs.axolotl.ai/docs/attention.html#flash-attention)、[NeMo Gym](https://github.com/axolotl-ai-cloud/axolotl/pull/3516) 与 [EBFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3527)。
  - Axolotl 现已转为 [uv 优先 (uv-first)](https://github.com/axolotl-ai-cloud/axolotl/pull/3545)，并支持 [SonicMoE 融合 LoRA](https://github.com/axolotl-ai-cloud/axolotl/pull/3519)。
- 2026/03:
  - Axolotl 新增了对 [Mistral Small 4](https://docs.axolotl.ai/docs/models/mistral4.html)、[Qwen3.5、Qwen3.5 MoE](https://docs.axolotl.ai/docs/models/qwen3.5.html)、[GLM-4.7-Flash](https://docs.axolotl.ai/docs/models/glm47-flash.html)、[GLM-4.6V](https://docs.axolotl.ai/docs/models/glm46v.html) 与 [GLM-4.5-Air](https://docs.axolotl.ai/docs/models/glm45.html) 的模型支持。
  - 支持 [MoE 专家量化](https://docs.axolotl.ai/docs/expert_quantization.html)（通过 `quantize_moe_experts: true`），在训练 MoE 模型时大幅降低显存占用（兼容 FSDP2）。

<details>

<summary>展开查看历史更新</summary>

- 2026/02:
  - 支持 [ScatterMoE LoRA](https://github.com/axolotl-ai-cloud/axolotl/pull/3410)：使用自定义 Triton Kernel 直接在 MoE 专家权重上进行 LoRA 微调。
  - Axolotl 现已支持 [SageAttention](https://github.com/axolotl-ai-cloud/axolotl/pull/2823) 与 [GDPO](https://github.com/axolotl-ai-cloud/axolotl/pull/3353)（Generalized DPO）。
- 2026/01:
  - 新增集成 [EAFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3366)（Entropy-Aware Focal Training，按 Top-k Logit 分布的熵对 Loss 加权）与 [Scalable Softmax](https://github.com/axolotl-ai-cloud/axolotl/pull/3338)（改善注意力中的长上下文表现）。
- 2025/12:
  - Axolotl 现已支持 [Kimi-Linear](https://docs.axolotl.ai/docs/models/kimi-linear.html)、[Plano-Orchestrator](https://docs.axolotl.ai/docs/models/plano.html)、[MiMo](https://docs.axolotl.ai/docs/models/mimo.html)、[InternVL 3.5](https://docs.axolotl.ai/docs/models/internvl3_5.html)、[Olmo3](https://docs.axolotl.ai/docs/models/olmo3.html)、[Trinity](https://docs.axolotl.ai/docs/models/trinity.html) 与 [Ministral3](https://docs.axolotl.ai/docs/models/ministral3.html)。
  - 新增面向 FSDP2 预训练的[分布式 Muon 优化器 (Distributed Muon Optimizer)](https://github.com/axolotl-ai-cloud/axolotl/pull/3264) 支持。
- 2025/10: Axolotl 新增了对以下模型的支持：[Qwen3 Next](https://docs.axolotl.ai/docs/models/qwen3-next.html)、[Qwen2.5-vl、Qwen3-vl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/qwen2_5-vl)、[Qwen3、Qwen3MoE](https://docs.axolotl.ai/docs/models/qwen3.html)、[Granite 4](https://docs.axolotl.ai/docs/models/granite4.html)、[HunYuan](https://docs.axolotl.ai/docs/models/hunyuan.html)、[Magistral 2509](https://docs.axolotl.ai/docs/models/magistral/vision.html)、[Apertus](https://docs.axolotl.ai/docs/models/apertus.html) 与 [Seed-OSS](https://docs.axolotl.ai/docs/models/seed-oss.html)。
- 2025/09: Axolotl 现已支持文本扩散 (text diffusion) 训练，详见[此处](https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/integrations/diffusion)。
- 2025/08: QAT 已更新以支持 NVFP4，详见 [PR](https://github.com/axolotl-ai-cloud/axolotl/pull/3107)。
- 2025/07:
  - Axolotl 新增 ND 并行 (ND Parallelism) 支持：可在单节点内以及跨多节点组合上下文并行 (CP)、张量并行 (TP) 与全分片数据并行 (FSDP)。更多信息请查阅[博客文章](https://huggingface.co/blog/accelerate-nd-parallel)。
  - Axolotl 新增更多模型：[GPT-OSS](https://docs.axolotl.ai/docs/models/gpt-oss.html)、[Gemma 3n](https://docs.axolotl.ai/docs/models/gemma3n.html)、[Liquid Foundation Model 2 (LFM2)](https://docs.axolotl.ai/docs/models/LiquidAI.html) 与 [Arcee Foundation Models (AFM)](https://docs.axolotl.ai/docs/models/arcee.html)。
  - 现已可通过 `torchao` 在 Axolotl 中进行带 fp8 gather 算子的 FP8 微调，[点此开始](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)！
  - 支持 mistral-common 分词器的 [Voxtral](https://docs.axolotl.ai/docs/models/voxtral.html)、[Magistral 1.1](https://docs.axolotl.ai/docs/models/magistral.html) 与 [Devstral](https://docs.axolotl.ai/docs/models/devstral.html) 已集成到 Axolotl 中！
  - 新增 TiledMLP 支持，覆盖单卡到多卡训练（支持 DDP、DeepSpeed 与 FSDP），用于支持 Arctic 长序列训练 (ALST)。在 Axolotl 中使用 ALST 的方式参见[示例](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst)！
- 2025/06: 支持 mistral-common 分词器的 Magistral 已加入 Axolotl。查阅[文档](https://docs.axolotl.ai/docs/models/magistral.html)，开始用 Axolotl 训练你自己的 Magistral 模型！
- 2025/05: Axolotl 新增量化感知训练 (QAT) 支持。查阅[文档](https://docs.axolotl.ai/docs/qat.html)了解更多！
- 2025/04: Axolotl 新增 Llama 4 支持。查阅[文档](https://docs.axolotl.ai/docs/models/llama-4.html)，使用 Axolotl 的线性化版本开始训练你自己的 Llama 4 模型！
- 2025/03: Axolotl 已实现序列并行 (Sequence Parallelism, SP) 支持。阅读[博客](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)与[文档](https://docs.axolotl.ai/docs/sequence_parallelism.html)，了解如何在微调时扩展上下文长度。
- 2025/03: (Beta) Axolotl 现已支持多模态模型微调。查阅[文档](https://docs.axolotl.ai/docs/multimodal.html)，微调你自己的模型！
- 2025/02: Axolotl 新增 LoRA 优化，可在单 GPU 与多 GPU 训练（DDP 与 DeepSpeed）中降低 LoRA 与 QLoRA 的显存占用并提升训练速度。进入[文档](https://docs.axolotl.ai/docs/lora_optims.html)来试试看。
- 2025/02: Axolotl 新增 GRPO 支持。深入阅读我们的[博客](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm)与 [GRPO 示例](https://github.com/axolotl-ai-cloud/grpo_code)，尽情探索！
- 2025/01: Axolotl 新增奖励建模 / 过程奖励建模微调支持。详见[文档](https://docs.axolotl.ai/docs/reward_modelling.html)。

</details>

## ✨ 概览

Axolotl 是一款免费开源的工具，旨在简化最新大语言模型 (LLM) 的后训练与微调流程。

功能特性：

- **多模型支持**：训练 GPT-OSS、LLaMA、Mistral、Mixtral、Pythia 等多种模型，以及 Hugging Face Hub 上的众多其他模型。
- **多模态训练**：微调视觉语言模型 (VLM)，包括 LLaMA-Vision、Qwen2-VL、Pixtral、LLaVA、SmolVLM2、GLM-4.6V、InternVL 3.5、Gemma 3n、PaddleOCR-VL、Muse Glimmer，以及 Voxtral 等音频模型，支持图像、视频与音频。
- **训练方法**：全量微调、LoRA、QLoRA、GPTQ、QAT (int8/int4/FP8/NVFP4/MXFP4)、FP8 混合精度训练、NVFP4/MXFP4 MoE LoRA、偏好微调（DPO、IPO、KTO、ORPO）、强化学习（GRPO、GDPO），以及奖励建模 (RM) / 过程奖励建模 (PRM)。
- **配置简单**：在完整的微调流程中复用同一个 YAML 配置文件：数据集预处理、训练、评估、量化与推理。
- **性能优化**：[Multipacking](https://docs.axolotl.ai/docs/multipack.html)、[Flash Attention 2/3/4](https://docs.axolotl.ai/docs/attention.html#flash-attention)、[Xformers](https://docs.axolotl.ai/docs/attention.html#xformers)、[Flex Attention](https://docs.axolotl.ai/docs/attention.html#flex-attention)、[SageAttention](https://docs.axolotl.ai/docs/attention.html#sageattention)、[Liger Kernel](https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels)、[Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy)、[ScatterMoE](https://docs.axolotl.ai/docs/custom_integrations.html#kernels-integration)、[序列并行 (SP)](https://docs.axolotl.ai/docs/sequence_parallelism.html)、[LoRA 优化](https://docs.axolotl.ai/docs/lora_optims.html)、[多 GPU 训练 (FSDP1, FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html)、[多节点训练 (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html) 等等！
- **灵活的数据集处理**：支持从本地、HuggingFace 以及云端（S3、Azure、GCP、OCI）加载数据集。
- **云端就绪**：我们提供 [Docker 镜像](https://hub.docker.com/u/axolotlai)与 [PyPI 软件包](https://pypi.org/project/axolotl/)，可用于云平台与本地硬件。



## 🚀 快速上手 - 数分钟内完成 LLM 微调

**环境要求**：

- NVIDIA GPU（使用 `bf16` 与 Flash Attention 需 Ampere 或更新架构）或 AMD GPU
- Python >=3.11（推荐 3.12）
- PyTorch ≥2.11.0（推荐 2.12.1）

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb#scrollTo=msOCO4NRmRLa)

### 安装

```bash
# 若尚未安装 uv，请先安装（安装后重启 Shell）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 根据系统情况修改
export UV_TORCH_BACKEND=cu130

# 创建一个新的虚拟环境
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.12.1 torchvision
uv pip install --no-build-isolation axolotl[deepspeed]

# 下载 axolotl 示例配置与 deepspeed 配置
axolotl fetch examples
axolotl fetch deepspeed_configs  # 可选
```

#### 使用 Docker

使用 Docker 安装可能比在你自己的环境中安装更不容易出错。
```bash
docker run --gpus '"all"' --ipc=host --rm -it axolotlai/axolotl:main-latest
```

其他安装方式请参阅[此处](https://docs.axolotl.ai/docs/installation.html)。

#### 云服务商

<details>

- [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
- [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
- [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
- [Novita](https://novita.ai/gpus-console?templateId=311)
- [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
- [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### 你的第一次微调

```bash
# 获取 axolotl 示例
axolotl fetch examples

# 或者，指定自定义路径
axolotl fetch examples --dest path/to/folder

# 使用 LoRA 训练模型
axolotl train examples/llama-3/lora-1b.yml
```

就这么简单！查阅我们的[入门指南 (Getting Started Guide)](https://docs.axolotl.ai/docs/getting-started.html)，了解更详细的步骤讲解。


## 📚 文档

- [安装选项 (Installation Options)](https://docs.axolotl.ai/docs/installation.html) - 针对不同环境的详细安装说明
- [支持矩阵 (Support Matrix)](https://docs.axolotl.ai/docs/support-matrix.html) - 功能支持情况、兼容性与尚未支持的部分
- [配置指南 (Configuration Guide)](https://docs.axolotl.ai/docs/config-reference.html) - 完整的配置选项与示例
- [数据集加载 (Dataset Loading)](https://docs.axolotl.ai/docs/dataset_loading.html) - 从各类来源加载数据集
- [数据集指南 (Dataset Guide)](https://docs.axolotl.ai/docs/dataset-formats/) - 支持的格式及其使用方式
- [多 GPU 训练 (Multi-GPU Training)](https://docs.axolotl.ai/docs/multi-gpu.html)
- [多节点训练 (Multi-Node Training)](https://docs.axolotl.ai/docs/multi-node.html)
- [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
- [API 参考 (API Reference)](https://docs.axolotl.ai/docs/api/) - 自动生成的代码文档
- [FAQ](https://docs.axolotl.ai/docs/faq.html) - 常见问题解答

## AI 智能体支持

Axolotl 内置了为 AI 编码智能体（Claude Code、Cursor、Copilot 等）优化的文档。这些文档随 pip 包一同打包，无需克隆仓库。

```bash
# 显示概览与可用的训练方法
axolotl agent-docs

# 特定主题的参考文档
axolotl agent-docs sft                 # 监督微调
axolotl agent-docs grpo                # GRPO 在线强化学习
axolotl agent-docs preference_tuning   # DPO、KTO、ORPO、SimPO
axolotl agent-docs reward_modelling    # 结果奖励模型与过程奖励模型
axolotl agent-docs pretraining         # 持续预训练
axolotl agent-docs --list              # 列出所有主题

# 导出配置 Schema 以供程序化使用
axolotl config-schema
axolotl config-schema --field adapter
```

如果你在源码仓库中工作，智能体文档同样位于 `docs/agents/`，项目概览则在 `AGENTS.md` 中。

## 🤝 获取帮助

- 加入我们的 [Discord 社区](https://discord.gg/HhrNrHJPRb)获取支持
- 查看我们的 [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) 目录
- 阅读我们的[调试指南 (Debugging Guide)](https://docs.axolotl.ai/docs/debugging.html)
- 需要专属支持？请联系 [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) 了解可选方案

## 🌟 参与贡献

欢迎贡献！详情请参阅我们的[贡献指南 (Contributing Guide)](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md)。

## 📈 遥测

Axolotl 提供可退出 (opt-out) 的遥测，帮助我们了解项目的使用情况并确定改进的优先级。我们收集基础系统信息、模型类型与错误率，绝不收集个人数据或文件路径。遥测默认开启。如需关闭，请设置 AXOLOTL_DO_NOT_TRACK=1。更多细节请参阅我们的[遥测文档](https://docs.axolotl.ai/docs/telemetry.html)。

## ❤️ 赞助

有意赞助？请通过 [wing@axolotl.ai](mailto:wing@axolotl.ai) 联系我们

## 📝 引用 Axolotl

如果你在研究或项目中使用了 Axolotl，请按如下方式引用：

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 许可证

本项目基于 Apache 2.0 许可证授权，详见 [LICENSE](LICENSE) 文件。
