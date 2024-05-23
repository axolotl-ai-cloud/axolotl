# Llama-3

https://llama.meta.com/llama3/

[8B Base Model](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
 - [Full Fine Tune](./fft-8b.yaml)
   - Single GPU @ 48GB VRAM
 - [LoRA](./lora-8b.yml)
   - Single GPU @ 11GB VRAM

[70B Base Model](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
 - [QLORA+FSDP](./qlora-fsdp-70b.yaml)
   - Dual GPU @ 21GB VRAM
