"""Distributed packed-expert ownership regression worker."""

import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
from bitsandbytes.nn.parametrize import replace_parameter_4bit
from torch import nn

from axolotl.integrations.expert_parallel.shard import (
    _detect_experts_modules,
    shard_expert_weights,
)


def main():
    device_type = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if device_type == "cuda":
        import os

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("cpu:gloo,cuda:nccl" if device_type == "cuda" else "gloo")
    rank = dist.get_rank()
    device = (
        torch.device(device_type, rank)
        if device_type == "cuda"
        else torch.device("cpu")
    )
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    for label, rows, compressed, meta_peer, quant_type in (
        ("nested-aligned", 128, True, False, "nf4"),
        ("nested-scale-cut", 8, True, False, "nf4"),
        ("uncompressed", 8, False, False, "fp4"),
        ("rank-zero-materialized", 128, True, True, "nf4"),
    ):
        torch.manual_seed(100 + rank)
        model = nn.Module()
        model.experts = nn.Module()
        model.experts.num_experts = 4
        expected = {}
        for name in ("gate_up_proj", "down_proj"):
            parameter = torch.randn(4, rows, 256, device=device, dtype=dtype)
            if meta_peer and rank:
                parameter = torch.empty_like(parameter, device="meta")
            setattr(model.experts, name, nn.Parameter(parameter, requires_grad=False))
            if parameter.device.type != "meta":
                replace_parameter_4bit(
                    model.experts,
                    name,
                    compress_statistics=compressed,
                    quant_type=quant_type,
                )
            reference = [
                getattr(model.experts, name).detach().cpu().clone()
                if rank == 0
                else None
            ]
            dist.broadcast_object_list(reference, src=0)
            expected[name] = reference[0][rank * 2 : (rank + 1) * 2]
        with patch(
            "bitsandbytes.functional.quantize_4bit",
            side_effect=AssertionError("Weights were requantized"),
        ):
            assert shard_expert_weights(model, dist.group.WORLD) == 1
        for name, reference in expected.items():
            torch.testing.assert_close(
                getattr(model.experts, name).cpu(), reference, rtol=0, atol=0
            )
            stack = model.experts.parametrizations[name]
            assert stack.original.dtype == torch.uint8
            assert stack.original.device == device
            assert stack[0].quant_state.absmax.device == device
            assert stack[0].quant_state.nested == (compressed and rows == 128)
            assert (
                f"experts.parametrizations.{name}.original"
                in model._ddp_params_and_buffers_to_ignore
            )
            with patch.object(
                stack[0],
                "forward",
                side_effect=AssertionError("Detection dequantized weights"),
            ):
                assert len(list(_detect_experts_modules(model))) == 1
        assert model.experts.num_local_experts == 2
        assert model.experts.local_expert_offset == rank * 2
        if rank == 0:
            print(f"PASS {label}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
