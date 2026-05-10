# MixLoRA Integration

MixLoRA is an integration that enables MoE-style (Mixture of Experts) LoRA fine-tuning of dense language models.
It works by inserting multiple LoRA-based experts into FFN (Feed-Forward Network) layers alongside a trainable router, while keeping the base model frozen. Independent LoRA adapters can also be added to the attention layers via standard PEFT.

See [MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts](https://arxiv.org/abs/2404.15159)

## Overview

MixLoRA dynamically routes tokens to specific LoRA experts within the FFN blocks during training and inference. This achieves parameter-efficient fine-tuning with higher capacity, decoupling the expert parameters from the core dense network operations while maintaining efficient computation.

## Usage

To enable MixLoRA, set the adapter type to `mixlora` and include the `MixLoraPlugin` in your configuration file. MixLoRA uses the base LoRA config for generating standard LoRA modules (like attention) but overrides the FFN targets to replace them with MixLoRA Modules.

```yaml
plugins:
  - axolotl.integrations.mixlora.MixLoraPlugin

adapter: mixlora

# Base LoRA parameters (applied to non-FFN targets like attention layers)
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

# MixLoRA specific configuration
mixlora_num_experts: 8           # Number of LoRA experts per FFN layer (default: 8)
mixlora_top_k: 2                 # Number of experts to route each token to (default: 2)
mixlora_router_aux_loss_coef: 0.01 # Coefficient for the auxiliary load balance loss (default: 0.01)
mixlora_router_init_range: 0.02  # Initialization range for router weights (default: 0.02)
mixlora_jitter_noise: 0.0        # Noise added to router inputs during training for exploration (default: 0.0)

# Optional: Override base LoRA config specifically for the experts
mixlora_expert_lora_r: 16        # Defaults to lora_r if not set
mixlora_expert_lora_alpha: 32    # Defaults to lora_alpha if not set
mixlora_expert_lora_dropout: 0.1 # Defaults to lora_dropout if not set
```

## Limitations

- MixLoRA patching currently supports SwiGLU FFN architectures with `gate_proj`, `up_proj`, and `down_proj` linear modules.
- `lora_target_modules` must not include `gate_proj`, `up_proj`, or `down_proj`, and `lora_target_linear` is not supported with `adapter: mixlora`.
- The expert dispatch loop is a straightforward per-expert implementation intended for correctness and integration simplicity over maximum throughput.

## Citation

```bib
@misc{li2024mixloraenhancinglarge,
      title={MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts},
      author={Jiahui Li and Qianguzhi Chen and Xiangyu Dong and Zhenwei Qiao and Hang Qi and Jiankai Sun and Jianing Lu and Junjie Zhao and Qingqiu Li and Zhenguo Li},
      year={2024},
      eprint={2404.15159},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.15159},
}
```
