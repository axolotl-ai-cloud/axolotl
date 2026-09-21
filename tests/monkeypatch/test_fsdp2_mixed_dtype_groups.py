"""FSDP2 tolerates fp32 embeddings in the root group alongside bf16 layer groups.

The embedding upcast only has to agree *within* a `fully_shard` group: axolotl wraps per
decoder layer, so fp32 embeddings and lm_head stay in the root group while the trainable
bf16 adapter params live in the layer groups.
"""

import pytest
import torch
from torch import nn


class _Layer(nn.Module):
    def __init__(self, hidden, rank):
        super().__init__()
        self.base = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
        self.base.weight.requires_grad_(False)
        self.lora_a = nn.Linear(hidden, rank, bias=False, dtype=torch.bfloat16)
        self.lora_b = nn.Linear(rank, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states):
        return self.base(hidden_states) + self.lora_b(self.lora_a(hidden_states))


class _Model(nn.Module):
    def __init__(self, vocab=32, hidden=16, rank=2, layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden, dtype=torch.float32)
        self.layers = nn.ModuleList(_Layer(hidden, rank) for _ in range(layers))
        self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=torch.float32)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(hidden_states)


def _mixed_dtype_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    world_size = 2
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(0)
        model = _Model()
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)

        assert model.embed_tokens.weight.dtype == torch.float32
        assert model.layers[0].lora_a.weight.dtype == torch.bfloat16

        logits = model(torch.randint(0, 32, (2, 4)))
        logits.float().pow(2).mean().backward()

        for layer in model.layers:
            assert layer.lora_a.weight.grad is not None
        assert model.embed_tokens.weight.grad is not None
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed_cpu
def test_fsdp2_fp32_embeddings_with_bf16_layer_groups(tmp_path):
    torch.multiprocessing.spawn(
        _mixed_dtype_worker, args=(str(tmp_path / "mixed"),), nprocs=2
    )
