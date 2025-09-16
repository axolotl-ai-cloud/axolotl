MoE Backends in Axolotl

Axolotl supports selecting a Mixture-of-Experts (MoE) compute backend via the training config (YAML):

- Set `moe_backend: auto|torch_grouped|naive`

Behavior
- auto (default): prefers PyTorch 2.8+ grouped GEMM; otherwise naive.
- torch_grouped: targets PyTorch 2.8+ grouped GEMM (H100/SM90+ recommended).
- naive: keeps the reference per-expert loop.

Notes
- Current implementation wires the backend selector and routes Mixtral MoE through it. Torch grouped uses cuBLASLt grouped GEMM when available; otherwise, the code falls back to the naive per-expert loop.
- No changes to training scripts are required; selection happens inside the model forward.

Example
moe_backend: torch_grouped
accelerate launch -m axolotl.cli.train path/to/config.yaml
