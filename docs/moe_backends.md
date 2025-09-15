MoE Backends in Axolotl

Axolotl supports selecting a Mixture-of-Experts (MoE) compute backend via the training config (YAML):

- Set `moe_backend: auto|hf_triton|torch_grouped|naive`

Behavior
- auto (default): prefers PyTorch 2.8+ grouped GEMM, then Hugging Face kernels hub, otherwise naive.
- hf_triton: uses the Hugging Face kernels hub (kernels-community/triton_kernels) when available.
- torch_grouped: targets PyTorch 2.8+ grouped GEMM.
- naive: keeps the reference per-expert loop.

Notes
- Current implementation wires the backend selector and routes Mixtral MoE through it. The hf_triton path is initially a stub: it uses kernels hub for routing but still falls back to per-expert computation until grouped GEMM is fully integrated.
- No changes to training scripts are required; selection happens inside the model forward. The `AXOLOTL_MOE_BACKEND` environment variable is no longer used.

Example
moe_backend: hf_triton
accelerate launch -m axolotl.cli.train path/to/config.yaml
