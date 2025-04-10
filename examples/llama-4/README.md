# Llama 4 by Meta AI

## Flash Attention vs Flex Attention

While Flash Attention to support is "enabled" for Llama-4, the upstream implementation is not correct and usage of Flex Attention is recommended.

## Available Examples

### Llama 4 Scout 17Bx16Experts (109B)

Flex Attention
- [Text Single GPU (H100) QLoRA](./scout-qlora-single-h100-flex.yaml)
- [Text Multi GPU QLoRA w/ FSDP2](./scout-qlora-flexattn-fsdp2.yaml)

[//]: # (Flash Attention &#40;Do not use&#41;)

[//]: # (- [Multi-Modal/Vision QLoRA w/ FSDP1]&#40;./scout-vision-qlora-fsdp.yaml&#41;)

[//]: # (- [Text Single GPU &#40;H100&#41; QLoRA]&#40;./scout-qlora-single-h100.yaml&#41;)

[//]: # (- [Text Multi GPU QLoRA w/ FSDP1]&#40;./scout-qlora-fsdp1.yaml&#41;)

Our Single H100 implementation for Llama 4 Scout uses only 68.5GB VRAM for post-training with 4k context length @ 546 tokens/second. [WandB logs here](https://wandb.ai/axolotl-ai/llama4-sft/runs/zic56rhd)

### Llama 4 Maverick 17Bx128Experts (400B)

- [Text Multi GPU QLoRA w/FSDP1](./maverick-qlora-fsdp1.yaml)

Our 4xH100 implementation for Llama 4 Maverick uses 79.5GB VRAM/GPU for post-training with 4k context length @ 206 tokens/second. [WandB logs here.](https://wandb.ai/axolotl-ai/llama-sft/runs/siyvwuxc?nw=nwuserwinglian)
