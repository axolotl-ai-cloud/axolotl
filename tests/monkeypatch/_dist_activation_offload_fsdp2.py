import os

import torch
import torch.distributed as dist
from accelerate import PartialState
from torch import nn
from torch.distributed.fsdp import fully_shard
from trl.models.activation_offloading import get_act_offloading_ctx_manager

from axolotl.core.trainers.mixins.activation_checkpointing import (
    _install_live_fsdp_parameter_storage_hooks,
    _patch_trl_offload_compute_stream_clone,
    _patch_trl_offload_current_stream,
)

live_parameter_pointers = set()
observed_parameter_pointers = set()


class CapturingLinear(nn.Linear):
    def forward(self, inputs):
        pointer = self.weight.untyped_storage().data_ptr()
        live_parameter_pointers.add(pointer)
        observed_parameter_pointers.add(pointer)
        try:
            return super().forward(inputs)
        finally:
            live_parameter_pointers.discard(pointer)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = CapturingLinear(32, 32, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    PartialState()
    _patch_trl_offload_current_stream()
    _patch_trl_offload_compute_stream_clone()
    model = Tiny().cuda().bfloat16()
    context = get_act_offloading_ctx_manager(model, min_offload_size=1)
    if os.environ["OFFLOAD_FSDP_LAYOUT"] == "leaf":
        fully_shard(model.linear)
    fully_shard(model)
    _install_live_fsdp_parameter_storage_hooks(context, model)
    parameter_offloads, parameter_pack_attempts, activation_offloads = [], [], []
    original_pack = context.pack_hook

    def pack(tensor):
        parameter = tensor.untyped_storage().data_ptr() in live_parameter_pointers
        result = original_pack(tensor)
        if isinstance(result, int) and result in context.tracker:
            cpu = any(
                torch.is_tensor(value) and value.device.type == "cpu"
                for value in context.tracker[result]
            )
            if parameter:
                parameter_pack_attempts.append(tuple(tensor.shape))
            if parameter and cpu:
                parameter_offloads.append(tuple(tensor.shape))
            if cpu and not parameter:
                activation_offloads.append(True)
        return result

    context.pack_hook = pack
    for _ in range(2):
        live_parameter_pointers.clear()
        inputs = torch.randn(
            4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with context:
            model(inputs).float().square().mean().backward()
        assert not live_parameter_pointers
        assert not context.param_storages
        model.zero_grad(set_to_none=True)
    assert activation_offloads
    assert observed_parameter_pointers
    assert parameter_pack_attempts
    assert all(shape == (32, 32) for shape in parameter_pack_attempts)
    assert not parameter_offloads, parameter_offloads
    dist.barrier()
    print("FSDP2_LIVE_OFFLOAD_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
