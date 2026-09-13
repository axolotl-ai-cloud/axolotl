<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>
  <p align="center">
      <strong>免费开源的大语言模型（LLM）后训练与微调框架</strong><br>
  </p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
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


## 🎉 最新动态 (Latest Updates)

- 2026/08:
  - Axolotl 新增对以下前沿模型的微调支持：[Ling 3.0](https://docs.axolotl.ai/docs/models/ling3.html)、[Muse Glimmer](https://docs.axolotl.ai/docs/models/muse-glimmer.html)、[North Micro Vision Instruct](https://docs.axolotl.ai/docs/models/cohere-north-micro-vision-instruct.html) 与 [Shieldstral](https://docs.axolotl.ai/docs/models/shieldstral.html)。
- 2026/07:
  - 现已支持 [NVFP4 (4-bit) MoE LoRA 训练](https://docs.axolotl.ai/docs/nvfp4_lora.html)：通过 ScatterMoE (W4A16) 与 SonicMoE (W4A4) 算子实现，支持将训练好的 Adapter 权重无损融合回原始 NVFP4 权重检查点中。
- 2026/06:
  - 通过 DeepEP 实现用于分布式 MoE 训练的[专家并行 (Expert Parallelism, EP)](https://docs.axolotl.ai/docs/nd_parallelism.html)、通过[兼容 Tinker 的 API](https://github.com/axolotl-ai-cloud/axolotl/pull/3614) 进行远程分布式训练、面向[混合架构 SSM 模型](https://github.com/axolotl-ai-cloud/axolotl/pull/3572)（如 Nemotron-H、Falcon-H1、Bamba）的上下文并行 (Context Parallelism)、[BitNet 1.58-bit](https://github.com/axolotl-ai-cloud/axolotl/pull/3634) 极低比特微调，以及[多模态仅针对 Assistant 回复计算 Loss 的掩码修复](https://github.com/axolotl-ai-cloud/axolotl/pull/3625)。
- 2026/04:
  - 新增模型支持：[Mistral Medium 3.5](https://docs.axolotl.ai/docs/models/mistral-medium-3_5.html) 与 [Gemma 4](https://docs.axolotl.ai/docs/models/gemma4.html)。
  - 全新强化学习与加速 Kernel：[异步 GRPO (Async GRPO)](https://github.com/axolotl-ai-cloud/axolotl/pull/3486)（每步提速最高达 58%）、[Flash Attention 4](https://docs.axolotl.ai/docs/attention.html#flash-attention)、[NeMo Gym](https://github.com/axolotl-ai-cloud/axolotl/pull/3516) 与 [EBFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3527)。
  - Axolotl 现已全面确立以 [uv 为第一优先级包管理器 (uv-first)](https://github.com/axolotl-ai-cloud/axolotl/pull/3545)，并深度支持 [SonicMoE 融合 LoRA 算子](https://github.com/axolotl-ai-cloud/axolotl/pull/3519)。
- 2026/03:
  - 新增模型支持：[Mistral Small 4](https://docs.axolotl.ai/docs/models/mistral4.html)、[Qwen3.5、Qwen3.5 MoE](https://docs.axolotl.ai/docs/models/qwen3.5.html)、[GLM-4.7-Flash](https://docs.axolotl.ai/docs/models/glm47-flash.html)、[GLM-4.6V](https://docs.axolotl.ai/docs/models/glm46v.html) 与 [GLM-4.5-Air](https://docs.axolotl.ai/docs/models/glm45.html)。
  - [MoE 专家量化](https://docs.axolotl.ai/docs/expert_quantization.html)特性上线（通过配置 `quantize_moe_experts: true`），在训练 MoE 架构模型时大幅节省显存占用（全面兼容 FSDP2）。

<details>

<summary>展开查看历史更新日志</summary>

- 2026/02:
  - 支持 [ScatterMoE LoRA](https://github.com/axolotl-ai-cloud/axolotl/pull/3410)：使用专属 Triton Kernel 直接在 MoE 专家权重上进行 LoRA 微调。
  - 支持 [SageAttention](https://github.com/axolotl-ai-cloud/axolotl/pull/2823) 与 [GDPO](https://github.com/axolotl-ai-cloud/axolotl/pull/3353)（广义直接偏好优化 Generalized DPO）。
- 2026/01:
  - 集成 [EAFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3366)（熵感知焦点训练 Entropy-Aware Focal Training，基于 Top-k Logit 分布的熵对 Loss 进行自适应加权）与 [Scalable Softmax](https://github.com/axolotl-ai-cloud/axolotl/pull/3338)（显著提升注意力机制的长上下文外推能力）。
- 2025/12:
  - Axolotl 新增模型生态支持：[Kimi-Linear](https://docs.axolotl.ai/docs/models/kimi-linear.html)、[Plano-Orchestrator](https://docs.axolotl.ai/docs/models/plano.html)、[MiMo](https://docs.axolotl.ai/docs/models/mimo.html)、[InternVL 3.5](https://docs.axolotl.ai/docs/models/internvl3_5.html)、[Olmo3](https://docs.axolotl.ai/docs/models/olmo3.html)、[Trinity](https://docs.axolotl.ai/docs/models/trinity.html) 以及 [Ministral3](https://docs.axolotl.ai/docs/models/ministral3.html)。
  - 新增面向 FSDP2 预训练的[分布式 Muon 优化器 (Distributed Muon Optimizer)](https://github.com/axolotl-ai-cloud/axolotl/pull/3264)。
- 2025/10: 新增模型支持：[Qwen3 Next](https://docs.axolotl.ai/docs/models/qwen3-next.html)、[Qwen2.5-VL、Qwen3-VL](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/qwen2_5-vl)、[Qwen3、Qwen3MoE](https://docs.axolotl.ai/docs/models/qwen3.html)、[Granite 4](https://docs.axolotl.ai/docs/models/granite4.html)、[HunYuan（腾讯混元）](https://docs.axolotl.ai/docs/models/hunyuan.html)、[Magistral 2509](https://docs.axolotl.ai/docs/models/magistral/vision.html)、[Apertus](https://docs.axolotl.ai/docs/models/apertus.html) 与 [Seed-OSS](https://docs.axolotl.ai/docs/models/seed-oss.html)。
- 2025/09: Axolotl 支持文本扩散模型（Text Diffusion）训练，详见[说明文档](https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/integrations/diffusion)。
- 2025/08: 量化感知训练（QAT）全面升级支持 NVFP4，详见 [PR #3107](https://github.com/axolotl-ai-cloud/axolotl/pull/3107)。
- 2025/07:
  - 新增 ND 并行（ND Parallelism）支持：支持在单节点及跨多节点内自由组合上下文并行 (CP)、张量并行 (TP) 与全分片数据并行 (FSDP)。详情请参阅官方[博客文章](https://huggingface.co/blog/accelerate-nd-parallel)。
  - 新增模型支持：[GPT-OSS](https://docs.axolotl.ai/docs/models/gpt-oss.html)、[Gemma 3n](https://docs.axolotl.ai/docs/models/gemma3n.html)、[Liquid Foundation Model 2 (LFM2)](https://docs.axolotl.ai/docs/models/LiquidAI.html) 以及 [Arcee Foundation Models (AFM)](https://docs.axolotl.ai/docs/models/arcee.html)。
  - 通过 `torchao` 实现支持 FP8 Gather 算子的 FP8 混合精度微调，快速入门参见[文档](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)。
  - 集成支持 mistral-common 分词器的 [Voxtral](https://docs.axolotl.ai/docs/models/voxtral.html)、[Magistral 1.1](https://docs.axolotl.ai/docs/models/magistral.html) 与 [Devstral](https://docs.axolotl.ai/docs/models/devstral.html)。
  - 新增 TiledMLP 支持，兼容单卡到多卡的 DDP、DeepSpeed 与 FSDP，赋能 Arctic 长序列训练 (ALST)，参考[范例目录](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst)。
- 2025/06: 引入支持 mistral-common 分词器的 Magistral 模型，详见[文档](https://docs.axolotl.ai/docs/models/magistral.html)。
- 2025/05: 新增量化感知训练 (QAT) 支持，详见 [QAT 文档](https://docs.axolotl.ai/docs/qat.html)。
- 2025/04: 新增 Llama 4 支持，参见[文档](https://docs.axolotl.ai/docs/models/llama-4.html)探索 Axolotl 线性化版本训练。
- 2025/03: 实现序列并行 (Sequence Parallelism, SP)，通过[博客](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)与[文档](https://docs.axolotl.ai/docs/sequence_parallelism.html)了解如何扩展长文本微调上下文。
- 2025/03: (测试版) 开启多模态模型微调支持，详见[多模态文档](https://docs.axolotl.ai/docs/multimodal.html)。
- 2025/02: 引入深度 LoRA 显存优化技术，降低单 GPU 与多 GPU（DDP / DeepSpeed）环境下的 LoRA 与 QLoRA 显存开销并提升训练速度，详见[文档](https://docs.axolotl.ai/docs/lora_optims.html)。
- 2025/02: 新增强化学习算法 GRPO（群相对策略优化）支持，参考[技术博客](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm)与 [GRPO 代码范例](https://github.com/axolotl-ai-cloud/grpo_code)。
- 2025/01: 支持奖励模型 (Reward Modelling, RM) 与过程奖励模型 (Process Reward Modelling, PRM) 的微调，详见[文档](https://docs.axolotl.ai/docs/reward_modelling.html)。

</details>

## ✨ 核心概览 (Overview)

Axolotl 是一款免费开源的现代化大语言模型（LLM）微调与后训练（Post-Training）工具链，旨在为开发者和研究团队提供最便捷、高效、标准化的训练体验。

核心特性：

- **广泛的模型生态覆盖**：原生支持 GPT-OSS、LLaMA、Mistral、Mixtral、Pythia 以及 Hugging Face Hub 上托管的海量开源模型。
- **全方位多模态训练**：支持对视觉语言模型（VLM）进行高效微调，涵盖 LLaMA-Vision、Qwen2-VL、Pixtral、LLaVA、SmolVLM2、GLM-4.6V、InternVL 3.5、Gemma 3n、PaddleOCR-VL、Muse Glimmer，以及支持语音多模态模型（如 Voxtral），覆盖图像、视频与音频模态。
- **前沿训练与对齐算法**：全面支持全量微调（Full Fine-Tuning）、LoRA、QLoRA、GPTQ、量化感知训练 QAT（int8/int4/FP8/NVFP4/MXFP4）、FP8 混合精度微调、NVFP4/MXFP4 MoE LoRA、偏好对齐算法（DPO、IPO、KTO、ORPO）、强化学习（GRPO、GDPO），以及奖励建模（RM 与 PRM 过程奖励建模）。
- **极简配置驱动 (YAML-First)**：复用单一 YAML 配置文件，即可贯穿从数据预处理、分布式训练、指标评估、模型量化到本地推理的全生命周期流水线。
- **极致性能加速算子**：集成 [Multipacking（样本高效打包）](https://docs.axolotl.ai/docs/multipack.html)、[Flash Attention 2/3/4](https://docs.axolotl.ai/docs/attention.html#flash-attention)、[Xformers](https://docs.axolotl.ai/docs/attention.html#xformers)、[Flex Attention](https://docs.axolotl.ai/docs/attention.html#flex-attention)、[SageAttention](https://docs.axolotl.ai/docs/attention.html#sageattention)、[Liger Kernel](https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels)、[Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy)、[ScatterMoE](https://docs.axolotl.ai/docs/custom_integrations.html#kernels-integration)、[序列并行 (Sequence Parallelism)](https://docs.axolotl.ai/docs/sequence_parallelism.html)、[LoRA 显存优化](https://docs.axolotl.ai/docs/lora_optims.html)、[多卡分布式训练 (FSDP1, FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html) 以及[多节点协同集群 (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html) 等等！
- **灵活的数据集加载机制**：支持无缝加载本地数据文件、Hugging Face 数据集仓库以及主流云对象存储（AWS S3、Azure Blob、Google Cloud Storage、Oracle OCI）。
- **开箱即用的云就绪生态**：官方维护发布 [Docker 容器镜像](https://hub.docker.com/u/axolotlai) 与标准 [PyPI 软件包](https://pypi.org/project/axolotl/)，适配各大云平台与本地算力集群。



## 🚀 快速上手 - 数分钟内开启 LLM 微调

**环境与硬件要求**：

- NVIDIA GPU（推荐 Ampere 或更新架构以启用 `bf16` 精度和 Flash Attention 加速）或 AMD GPU
- Python >= 3.11（强烈推荐 Python 3.12）
- PyTorch >= 2.11.0（推荐 PyTorch 2.12.1）

### Google Colab 在线体验

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb#scrollTo=msOCO4NRmRLa)

### 本地安装指南

```bash
# 若尚未安装 uv，请先执行安装（安装完成后建议重启终端 Shell）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 根据你的系统环境与 CUDA 版本进行配置
export UV_TORCH_BACKEND=cu130

# 创建全新的 Python 3.12 虚拟环境
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.12.1 torchvision
uv pip install --no-build-isolation axolotl[deepspeed]

# 一键拉取 Axolotl 官方配置范例与 DeepSpeed 配置文件
axolotl fetch examples
axolotl fetch deepspeed_configs  # 可选项
```

#### 使用 Docker 容器镜像

在容器环境中运行可以大幅减少底层驱动与依赖项冲突：
```bash
docker run --gpus '"all"' --ipc=host --rm -it axolotlai/axolotl:main-latest
```

更多安装选项请参阅[官方安装指南](https://docs.axolotl.ai/docs/installation.html)。

#### 云端算力托管方案 (Cloud Providers)

<details>

- [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
- [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
- [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
- [Novita](https://novita.ai/gpus-console?templateId=311)
- [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
- [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### 运行你的首个模型微调

```bash
# 获取 Axolotl 官方预置训练配置示例
axolotl fetch examples

# 或者指定保存配置的自定义本地路径
axolotl fetch examples --dest path/to/folder

# 使用 LoRA 启动模型微调训练
axolotl train examples/llama-3/lora-1b.yml
```

大功告成！如需了解更详尽的步骤说明，请参阅[新手入门指南 (Getting Started Guide)](https://docs.axolotl.ai/docs/getting-started.html)。


## 📚 官方文档索引

- [安装选项 (Installation Options)](https://docs.axolotl.ai/docs/installation.html) - 不同硬件环境下的详细安装说明
- [特性兼容矩阵 (Support Matrix)](https://docs.axolotl.ai/docs/support-matrix.html) - 特性支持情况、版本兼容性与已知说明
- [配置参考指南 (Configuration Guide)](https://docs.axolotl.ai/docs/config-reference.html) - 完整的 YAML 配置选项说明与详尽范例
- [数据集加载指南 (Dataset Loading)](https://docs.axolotl.ai/docs/dataset_loading.html) - 从各类本地与云端源加载数据
- [数据集格式手册 (Dataset Guide)](https://docs.axolotl.ai/docs/dataset-formats/) - 支持的数据格式及对应解析处理规范
- [多卡并行训练 (Multi-GPU Training)](https://docs.axolotl.ai/docs/multi-gpu.html)
- [多节点集群训练 (Multi-Node Training)](https://docs.axolotl.ai/docs/multi-node.html)
- [Multipacking 样本打包优化](https://docs.axolotl.ai/docs/multipack.html)
- [API 参考手册 (API Reference)](https://docs.axolotl.ai/docs/api/) - 自动生成的代码级别接口文档
- [常见问题解答 (FAQ)](https://docs.axolotl.ai/docs/faq.html) - 社区高频疑难解答

## AI 智能体开发支持 (AI Agent Support)

Axolotl 内置了专为 AI 编码智能体（如 Claude Code、Cursor、Copilot 等）深度优化的结构化文档支持。这些文档直接随 pip 软件包分发打包，无需克隆完整仓库即可即时唤起。

```bash
# 查看全局概览与支持的训练方法
axolotl agent-docs

# 查看特定主题的快速参考手册
axolotl agent-docs sft                 # 监督式微调 (Supervised Fine-Tuning)
axolotl agent-docs grpo                # GRPO 在线强化学习算法
axolotl agent-docs preference_tuning   # DPO、KTO、ORPO、SimPO 偏好微调
axolotl agent-docs reward_modelling    # 结果奖励模型与过程奖励模型 (PRM)
axolotl agent-docs pretraining         # 持续预训练 (Continual Pretraining)
axolotl agent-docs --list              # 列出所有可查询的文档主题

# 以编程方式导出 YAML 配置 Schema 校验模型
axolotl config-schema
axolotl config-schema --field adapter
```

如果你在源码仓库中工作，Agent 文档同样存放在源码目录 `docs/agents/` 下，项目架构总览记录于 `AGENTS.md`。

## 🤝 社区与支持

- 加入官方 [Discord 社区频道](https://discord.gg/HhrNrHJPRb) 寻求技术支持与交流
- 探索官方 [Examples 范例目录](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) 获取实战灵感
- 查阅[调试与排错指南 (Debugging Guide)](https://docs.axolotl.ai/docs/debugging.html)
- 如需企业专属支持，请发送邮件至 [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) 咨询合作

## 🌟 参与贡献

我们非常欢迎来自社区的贡献！请阅读[贡献指南 (Contributing Guide)](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) 了解详细规范与提交流程。

## 📈 遥测与数据隐私

Axolotl 内置了可退出的遥测机制（Opt-out Telemetry），用于帮助维护团队了解项目在社区中的使用模式并优先安排功能改进。我们仅收集基础系统信息、模型架构类型与异常报错率，**绝不会收集任何用户隐私数据、密钥或文件路径**。
遥测默认处于开启状态；如需关闭，仅需在环境中设置环境变量 `AXOLOTL_DO_NOT_TRACK=1`。更多细节请参阅[遥测说明文档](https://docs.axolotl.ai/docs/telemetry.html)。

## ❤️ 赞助支持

有意向赞助 Axolotl 项目发展？欢迎随时联系 [wing@axolotl.ai](mailto:wing@axolotl.ai)。

## 📝 引用 Axolotl

如果您在学术研究或开源项目中使用了 Axolotl，请按照如下格式引用：

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 开源许可证

本项目采用 Apache-2.0 开源许可证授权 - 详见 [LICENSE](LICENSE) 文件。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
