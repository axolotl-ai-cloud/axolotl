# QuACK Kernels

[QuACK](https://github.com/Dao-AILab/quack) (Quirky Assortment of CuTe Kernels) is a
CuTe-DSL kernel library that reaches speed-of-light on Hopper/Blackwell/RTX50 in pure
Python. This plugin currently exposes one op:

- **Fused gated MLP (`quack_mlp_kernel`)** — fuses the up-projection GEMM with the gated
  activation (SwiGLU / GeGLU) in a single kernel, forward and backward. Unlike
  `liger_glu_activation`, which fuses only the elementwise activation and leaves the two
  projection matmuls to cuBLAS, quack fuses the matmul with the activation so the
  `2*intermediate`-wide pre-activation never fully lands in HBM.

## Usage

```yaml
plugins:
  - axolotl.integrations.quack_kernels.QuackKernelsPlugin

quack_mlp_kernel: true
```

## Requirements

- A Hopper/Blackwell/RTX50 GPU (SM90+). Patched MLPs fall back to the original forward
  on other devices, so the flag is safe to leave on for mixed fleets.
- `pip install quack-kernels` (see the `quack-cute` skill for the pinned toolchain; the
  same `nvidia-cutlass-dsl` install is shared with the sonic-moe NVFP4 path).

## Scope and limitations

- Applies to **dense** gated MLPs (the decoder layer's `.mlp` / `.feed_forward` with
  bias-free `gate_proj` / `up_proj` / `down_proj`). Llama, Mistral, Qwen2/3, Gemma-style
  MLPs qualify. Intermediate size must be divisible by 8, hidden size by 8.
- **Skips LoRA and quantized projections** — using the base `.weight` would drop the
  adapter delta. For fused LoRA MLP use `lora_mlp_kernel` instead.
- **Skips routed MoE experts and routers** — those are handled by the MoE kernels
  (`sonicmoe` / `scattermoe`).
- Mutually exclusive with `liger_glu_activation`, `lora_mlp_kernel`, and `tiled_mlp`
  (unless `tiled_mlp_use_original_mlp: true`).
- Weights must be fp16/bf16.
- Under `torch_compile: true` the wrapper graph-breaks (eager fallback still runs, so
  correctness is preserved) and the fusion speedup may be reduced. Benchmark before
  combining the two.

## Conflicts

The plugin refuses to start if `quack_mlp_kernel` is combined with another MLP-rewriting
path (`liger_glu_activation`, `lora_mlp_kernel`, or `tiled_mlp`).
