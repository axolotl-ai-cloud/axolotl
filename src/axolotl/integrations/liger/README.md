# Liger Kernel Integration

Liger Kernel provides efficient Triton kernels for LLM training, offering:

- 20% increase in multi-GPU training throughput
- 60% reduction in memory usage
- Compatibility with both FSDP and DeepSpeed

See https://github.com/linkedin/Liger-Kernel

## Usage

```yaml
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rope: true
liger_rms_norm: true
liger_glu_activation: true
liger_layer_norm: true
liger_fused_linear_cross_entropy: true

# FLCE-specific
liger_use_token_scaling: true

# Optional: alternative kernel backend (liger-kernel >= 0.8.1, off by default)
# cutile (pip install cuda-tile) covers cross_entropy/geglu/layer_norm/rope and more;
# cutedsl (pip install nvidia-cutlass-dsl) covers cross_entropy/rms_norm, tuned for Blackwell.
# fused_linear_cross_entropy stays on Triton with either backend.
liger_kernel_impl: cutedsl
```

## Supported Models

Any model type in liger-kernel's native dispatch table (`liger_kernel.transformers.monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN`) is supported out of the box — llama, mistral, mixtral, qwen2/qwen3 families, gemma through gemma4, deepseek_v4, glm4, phi3, olmo2, paligemma, and many more (50+ types as of liger 0.8.1).

On top of the native table, axolotl hand-patches these types (not covered upstream, or extended with kernels the native path lacks):

- deepseek_v2
- gemma4_unified / gemma4_unified_text
- granitemoe
- jamba
- qwen3_5 / qwen3_5_moe (adds the fused gated-RMSNorm kernel for linear-attention layers)

## Citation

```bib
@article{hsu2024ligerkernelefficienttriton,
      title={Liger Kernel: Efficient Triton Kernels for LLM Training},
      author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen},
      year={2024},
      eprint={2410.10989},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.10989},
      journal={arXiv preprint arXiv:2410.10989},
}
```
