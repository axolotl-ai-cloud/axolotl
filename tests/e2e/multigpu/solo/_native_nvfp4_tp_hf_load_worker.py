import os
import shutil
import sys

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from transformers import LlamaConfig, LlamaForCausalLM, TorchAoConfig

from axolotl.monkeypatch.torchao_tp import native_nvfp4_tp_checkpoint_loading


def main():
    rank = int(os.environ["RANK"])
    print(
        f"NVFP4_TP_GATE_ENV rank={rank} executable={sys.executable} torch={torch.__version__}",
        flush=True,
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        path = os.environ["NVFP4_TP_GATE_PATH"]
        if rank == 0:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)
            config = LlamaConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=64,
            )
            config.quantization_config = TorchAoConfig(
                NVFP4WeightOnlyConfig()
            ).to_dict()
            config.save_pretrained(path)
            torch.manual_seed(17)
            model = LlamaForCausalLM(config).eval()
            for module in model.modules():
                if (
                    isinstance(module, torch.nn.Linear)
                    and module.weight.shape[0] % 16 == 0
                    and module.weight.shape[1] % 16 == 0
                ):
                    weight = NVFP4Tensor.to_nvfp4(
                        module.weight.detach().float(),
                        per_tensor_scale=torch.tensor(1.0),
                        is_swizzled_scales=True,
                    )
                    module.weight = torch.nn.Parameter(weight, requires_grad=False)
            flattened, metadata = flatten_tensor_state_dict(model.state_dict())
            save_file(
                flattened,
                os.path.join(path, "model.safetensors"),
                metadata=metadata,
            )
        dist.barrier()
        serial = (
            LlamaForCausalLM.from_pretrained(path, dtype=torch.float32)
            .to("cuda")
            .eval()
            if rank == 0
            else None
        )
        with native_nvfp4_tp_checkpoint_loading(mesh):
            loaded = LlamaForCausalLM.from_pretrained(
                path,
                tp_plan="auto",
                tp_size=2,
                device_mesh=mesh,
                dtype=torch.float32,
            ).eval()
        q_weight = loaded.model.layers[0].self_attn.q_proj.weight
        o_weight = loaded.model.layers[0].self_attn.o_proj.weight
        assert type(q_weight).__name__ == "DTensor"
        assert type(o_weight).__name__ == "DTensor"
        assert q_weight.shape == (64, 64), q_weight.shape
        assert o_weight.shape == (64, 64), o_weight.shape
        q_local = q_weight.to_local()
        o_local = o_weight.to_local()
        assert type(q_local).__name__ == "NVFP4Tensor"
        assert type(o_local).__name__ == "NVFP4Tensor"
        assert q_local.shape == (32, 64), q_local.shape
        assert o_local.shape == (64, 32), o_local.shape
        assert q_local.qdata.untyped_storage().nbytes() == q_local.qdata.numel()
        assert o_local.qdata.untyped_storage().nbytes() == o_local.qdata.numel()
        input_ids = torch.tensor([[1, 2, 3]], device="cuda")
        with torch.no_grad():
            output = loaded(input_ids).logits
            if rank == 0:
                torch.testing.assert_close(
                    output, serial(input_ids).logits, rtol=1e-5, atol=1e-5
                )
        outputs = [torch.empty_like(output) for _ in range(2)]
        dist.all_gather(outputs, output)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        if rank == 0:
            print("HF_NVFP4_TP_GATE_PASS", q_weight.shape, o_weight.shape, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
