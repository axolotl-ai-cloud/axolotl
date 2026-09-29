"""ZeRO FP32 accumulation, reduction, master updates, and Adam state parity."""

import copy
import os

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import (
    safe_get_full_fp32_param,
    safe_get_full_grad,
    safe_get_full_optimizer_state,
)
from test_lora_fp32_gradients import projection
from torch import nn

from axolotl.monkeypatch.deepspeed_utils import patch_zero_gradient_accumulation_dtype
from axolotl.utils.lora_precision import configure_deepspeed_lora_precision


class UnevenPartitionBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(3, device="cuda", dtype=torch.bfloat16))

    def forward(self, value):
        return value + torch.nn.functional.pad(self.bias, (0, value.shape[-1] - 3))


def main():
    patch_zero_gradient_accumulation_dtype()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(17)
    model = nn.Sequential(projection("cuda"))
    if os.environ.get("TEST_UNEVEN_PARTITION"):
        model.append(UnevenPartitionBias())
    reference = copy.deepcopy(model)
    reference_parameters = dict(reference.named_parameters())
    masters = {
        name: nn.Parameter(parameter.detach().float().clone())
        for name, parameter in reference_parameters.items()
        if parameter.requires_grad
    }
    reference_optimizer = torch.optim.AdamW(list(masters.values()), lr=0.01)
    stage = int(os.environ.get("TEST_ZERO_STAGE", "2"))
    config = configure_deepspeed_lora_precision(
        {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": stage,
                "reduce_bucket_size": int(os.environ.get("TEST_BUCKET_SIZE", "1024")),
                "contiguous_gradients": os.environ.get("TEST_CONTIGUOUS", "1") == "1",
                "overlap_comm": os.environ.get("TEST_OVERLAP", "0") == "1",
                "reduce_scatter": os.environ.get("TEST_REDUCE_SCATTER", "1") == "1",
            },
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 3,
            "gradient_clipping": 0.0,
            "steps_per_print": 1000,
        }
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=config
    )
    assert engine.get_data_types()[1] == torch.float32
    assert engine.communication_data_type == torch.float32
    for update in range(2):
        sums = {name: torch.zeros_like(master) for name, master in masters.items()}
        for micro in range(3):
            generator = torch.Generator(device="cuda").manual_seed(
                1000 + update * 100 + micro * world + rank
            )
            x = torch.randn(
                2, 5, 32, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            engine.backward(engine(x).float().square().mean())
            reference.zero_grad(set_to_none=True)
            ref_output = reference(x)
            # DeepSpeed divides the BF16 output gradient rather than the FP32 loss.
            ref_output.register_hook(lambda grad: grad / 3)
            ref_output.float().square().mean().backward()
            for name in masters:
                sums[name].add_(reference_parameters[name].grad.float())
            if micro == 2:
                for name, parameter in engine.module.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    dist.all_reduce(sums[name], op=dist.ReduceOp.SUM)
                    sums[name].div_(world)
                    # BF16Optimizer's full gradient list has partition-local accessor indices.
                    gradient = (
                        parameter._hp_grad
                        if stage == 1
                        else safe_get_full_grad(parameter)
                    )
                    if stage == 1:
                        assert gradient.ndim == 1
                        assert gradient.numel() == sums[name].numel()
                        gradient = gradient.view_as(sums[name])
                    assert gradient.shape == sums[name].shape
                    assert gradient.dtype == torch.float32
                    if stage == 1 and os.environ.get("TEST_AUDIT_ZERO1_ACCESSOR"):
                        accessor = parameter.get_full_hp_grad()
                        if not torch.equal(accessor, gradient):
                            aliases = [
                                n
                                for n, p in engine.module.named_parameters()
                                if p.requires_grad
                                and p._hp_grad.numel() == accessor.numel()
                                and torch.equal(
                                    p._hp_grad.flatten(), accessor.flatten()
                                )
                            ]
                            print("ZERO1_ACCESSOR_ALIAS", name, aliases, flush=True)
                    torch.testing.assert_close(
                        gradient,
                        sums[name],
                        rtol=1e-5,
                        atol=1e-7,
                        msg=lambda error, update=update, name=name: (
                            f"stage={stage} rank={rank} update={update} {name}: {error}"
                        ),
                    )
            engine.step()
        for name, master in masters.items():
            master.grad = sums[name]
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        for name, parameter in engine.module.named_parameters():
            if not parameter.requires_grad:
                continue
            actual_master = safe_get_full_fp32_param(parameter)
            assert actual_master.dtype == torch.float32
            torch.testing.assert_close(
                actual_master, masters[name], rtol=1e-5, atol=1e-7, msg=name
            )
            for state_key in ("exp_avg", "exp_avg_sq"):
                actual_state = safe_get_full_optimizer_state(parameter, state_key)
                assert actual_state.dtype == torch.float32
                torch.testing.assert_close(
                    actual_state,
                    reference_optimizer.state[masters[name]][state_key],
                    rtol=1e-5,
                    atol=1e-7,
                    msg=f"{name}/{state_key}",
                )
        with torch.no_grad():
            for name, master in masters.items():
                reference_parameters[name].copy_(master)
        with deepspeed.zero.GatheredParameters(
            list(engine.module.parameters()), enabled=stage == 3
        ):
            for name, parameter in engine.module.named_parameters():
                assert parameter.dtype == torch.bfloat16
                torch.testing.assert_close(
                    parameter, reference_parameters[name], rtol=0, atol=0, msg=name
                )
        assert engine.global_steps == update + 1
    print(
        f"LORA_FP32_ZERO_{stage}_OK rank={rank} world_size={world} "
        "updates=2 gradients=fp32 communication=fp32 masters=fp32 moments=fp32",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
