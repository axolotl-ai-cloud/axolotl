# Jamba

- ✅ qlora w/ deepspeed Zero-2 needs at least 2x GPUs and
  - 35GiB VRAM per GPU w minimal context length
  - 56GiB VRAM per GPU (w multipack enabled)
- ✅ qlora w/ deepspeed Zero-3 needs at least 2x GPUs and 67GiB VRAM (wtf?)
- ✅ qlora single-gpu, ~51GiB VRAM
- ✅ multipack
- ✅ FSDP
- ❓ 8-bit LoRA
