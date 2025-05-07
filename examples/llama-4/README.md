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

Our Single H100 implementation for Llama 4 Scout uses only 64.5GB VRAM for post-training with 4k context length @ 519 tokens/second. [WandB logs here](https://wandb.ai/axolotl-ai/llama4-flexattn-qlora/runs/wpie7dkj)
Multi-GPU (4xH100) for Llama 4 Scout uses 62.8GB VRAM/GPU @ 4k contenxt length @ 280tps/gpu, [WandB logs here](https://wandb.ai/axolotl-ai/llama4-flexattn-qlora/runs/2lkezdj8)

### Llama 4 Maverick 17Bx128Experts (400B)

Coming Soon

## Delinearized Llama 4 Models

We provide a script to delinearize Llama 4 linearized models into regular HuggingFace Llama 4 models.

```bash
axolotl delinearize-llama4 --model path/to/model_dir --output path/to/output_dir
```

Note: This only works with the non-quantized linearized model. If you have an adapter, merge it with the *non-quantized linearized* model before delinearizing.
