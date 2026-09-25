"""Two-rank FSDP lifecycle oracle for frozen dynamic NVFP4 input STEs."""

import datetime
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fsdp import (
    normalize_dense_nvfp4_scales,
    patch_nvfp4_fsdp,
)
from axolotl.monkeypatch.torchao_nvfp4_dynamic_ste import (
    install_fsdp_native_nvfp4_dynamic_input_stes,
)

WIDTH = 128
RANK = 4


class Toy(torch.nn.Module):
    def __init__(self, post):
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.randn(RANK, WIDTH) / 64)
        self.lora_b = torch.nn.Parameter(torch.randn(WIDTH, RANK) / 64)
        self.post = post

    def forward(self, inputs):
        hidden = inputs + (inputs @ self.lora_a.t()) @ self.lora_b.t()
        return self.post(hidden)


def _dynamic_weight():
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        QuantizeTensorToNVFP4Kwargs,
    )

    weight = NVFP4Tensor.to_nvfp4(
        torch.randn(WIDTH, WIDTH, device="cuda", dtype=torch.bfloat16) / 32,
        act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(use_dynamic_per_tensor_scale=True),
    )
    normalize_dense_nvfp4_scales(weight)
    return weight


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    patch_nvfp4_fsdp()
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    try:
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        post = torch.nn.Linear(
            WIDTH, WIDTH, bias=False, device="cuda", dtype=torch.bfloat16
        )
        post.weight = torch.nn.Parameter(_dynamic_weight(), requires_grad=False)
        reference_weight = post.weight.detach().clone()
        model = Toy(post).cuda().bfloat16()
        fully_shard(model.post)
        assert install_fsdp_native_nvfp4_dynamic_input_stes(model) == 1
        assert not hasattr(model.post, "_axolotl_materialize_orig_forward")

        inputs = torch.randn(2, WIDTH, device="cuda", dtype=torch.bfloat16)
        output = model(inputs)
        output.float().square().mean().backward()
        assert model.lora_a.grad is not None and model.lora_b.grad is not None

        reference_a = model.lora_a.detach().clone().requires_grad_()
        reference_b = model.lora_b.detach().clone().requires_grad_()
        hidden = inputs + (inputs @ reference_a.t()) @ reference_b.t()
        grad_output = 2 * output.detach().float() / output.numel()
        hidden.backward(grad_output.to(hidden.dtype) @ reference_weight.dequantize())
        torch.testing.assert_close(
            model.lora_a.grad, reference_a.grad, rtol=2e-3, atol=2e-4
        )
        torch.testing.assert_close(
            model.lora_b.grad, reference_b.grad, rtol=2e-3, atol=2e-4
        )
        if dist.get_rank() == 0:
            print("NATIVE_NVFP4_DYNAMIC_STE_PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
