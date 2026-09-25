"""Four-rank Mamba2 mixer parity, including halo and prefix-state gradients."""

import copy
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from ringmaster.mamba import mamba2_mixers, wire_mamba2
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer


def main():
    dist.init_process_group("gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(42)
        config = Mamba2Config(
            hidden_size=16,
            expand=2,
            num_heads=4,
            head_dim=8,
            state_size=8,
            n_groups=2,
            num_hidden_layers=1,
            chunk_size=8,
        )
        model = Mamba2Mixer(config, 0).float()
        reference = copy.deepcopy(model)
        x = torch.randn(2, 64, 16)
        target_x = x.clone().requires_grad_()
        packed = os.environ.get("RM_PACKED") == "1"
        lengths = [5, 19, 1, 21, 18]
        expected = (
            torch.cat(
                [reference(chunk) for chunk in target_x.split(lengths, dim=1)], dim=1
            )
            if packed
            else reference(target_x)
        )
        if packed:
            from ringmaster.runtime import set_runtime
            from ringmaster.shard import varlen_meta

            positions = torch.cat([torch.arange(n) for n in lengths]).expand(2, -1)
            set_runtime(SimpleNamespace(varlen=varlen_meta(positions, 64)))
        grad = torch.randn_like(expected)
        (expected * grad).sum().backward()
        mixers = mamba2_mixers(model)
        assert mixers == [model]
        restore = wire_mamba2(mixers, dist.group.WORLD)
        local_x = x.chunk(world, dim=1)[rank].contiguous().requires_grad_()
        actual = model(local_x)
        torch.testing.assert_close(
            actual, expected.chunk(world, dim=1)[rank], atol=2e-5, rtol=2e-4
        )
        (actual * grad.chunk(world, dim=1)[rank]).sum().backward()
        torch.testing.assert_close(
            local_x.grad, target_x.grad.chunk(world, dim=1)[rank], atol=2e-5, rtol=2e-4
        )
        for (name, param), (_, ref) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            dist.all_reduce(param.grad)
            torch.testing.assert_close(
                param.grad, ref.grad, atol=2e-4, rtol=2e-3, msg=name
            )
        restore()
        assert "forward" not in vars(model)
        print(f"PASS Mamba CP={world} rank={rank} forward/backward", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
