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
```

## Supported Models

- deepseek_v2
- gemma
- gemma2
- gemma3 (partial support, no support for FLCE yet)
- granite
- jamba
- llama
- mistral
- mixtral
- mllama
- mllama_text_model
- olmo2
- paligemma
- phi3
- qwen2
- qwen2_5_vl
- qwen2_vl

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
