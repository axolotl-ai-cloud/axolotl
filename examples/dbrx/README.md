# DBRX MoE

Currently, for LoRA, only the `q_proj`, `k_proj`, `v_proj` `out_proj` and `layer` Linear layers are trainable.

We are using the "converted" base models based on [this issue](https://huggingface.co/databricks/dbrx-instruct/discussions/10)
where the Experts are fused as an `nn.Parameter` rather than a `nn.Linear` layer. However, the implementation
is still a bit buggy and attempting to train a LoRA adapter over those `w1`, `w2` and `v1` layers
results in the trainer hanging.


### FSDP
We've tested using the [`LnL-AI/dbrx-base-converted-v2`](https://huggingface.co/LnL-AI/dbrx-base-converted-v2) model as the base model for FSDP.

The high memory usage seen w/ FSDP is due to FSDP not supporting 8bit optimizers.

- 16-bit LoRA w/ FSDP
  - ✅ w/o CPU Offload - 8x80GB uses ~80GiB/gpu
  - ❌ w/ CPU Offload - `paged_adamw_8bit` optimizer errors from being on cpu
- ✅ 8-bit LoRA w/ FSDP
- ❌ 4-bit QLoRA w/ FSDP - errors w/: `Error an illegal memory access was encountered at line 90 in file /src/csrc/ops.cu`
- ✅ bf16 full finetune w/ FSDP, freezing all but first 8 layers (8x80GB uses ~78GiB/gpu)


### Deepspeed

WIP
