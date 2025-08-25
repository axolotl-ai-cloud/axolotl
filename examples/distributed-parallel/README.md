# ND Parallelism Examples

This directory contains example configurations for training models using ND Parallelism in Axolotl. These examples demonstrate how to compose different parallelism strategies (FSDP, TP, CP, HSDP) for efficient multi-GPU training.

## Quick Start

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).

2. Run the command below:

```bash
# Train Qwen3 8B with FSDP + TP + CP on a single 8-GPU node
axolotl train examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml

# Train Llama 3.1 8B with HSDP + TP on 2 nodes (16 GPUs total)
axolotl train examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml
```

## Example Configurations

### Single Node (8 GPUs)

**Qwen3 8B with FSDP + TP + CP** ([qwen3-8b-fsdp-tp-cp.yaml](./qwen3-8b-fsdp-tp-cp.yaml))
- Uses all 3 parallelism dimensions on a single node
- Ideal for: when model weights, activations, and/or context are too large to fit on single GPU

```yaml
dp_shard_size: 2         # FSDP across 2 GPUs
tensor_parallel_size: 2  # TP across 2 GPUs
context_parallel_size: 2 # CP across 2 GPUs
# Total: 2 × 2 × 2 = 8 GPUs
```

### Multi-Node

**Llama 3.1 8B with HSDP + TP** ([llama-3_1-8b-hsdp-tp.yaml](./llama-3_1-8b-hsdp-tp.yaml))
- FSDP & TP within nodes, DDP across nodes to minimize inter-node communication
- Ideal for: Scaling to multiple nodes while maintaining training efficiency

```yaml
dp_shard_size: 4        # FSDP within each 4-GPU group
tensor_parallel_size: 2 # TP within each node
dp_replicate_size: 2    # DDP across 2 groups
# Total: (4 × 2) × 2 = 16 GPUs (2 nodes)
```

## Learn More

- [ND Parallelism Documentation](https://docs.axolotl.ai/docs/nd_parallelism.html)
- [Blog: Accelerate ND-Parallel Guide](https://huggingface.co/blog/accelerate-nd-parallel)
- [Multi-GPU Training Guide](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
