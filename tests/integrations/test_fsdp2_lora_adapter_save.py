"""CPU tests for FSDP2 adapter gathering with parameter offload."""

from types import SimpleNamespace

import torch

from axolotl.integrations.expert_parallel.shard import (
    _gather_adapter_tensor,
    save_fsdp2_lora_adapter,
)


def test_cpu_offloaded_adapter_dtensor_uses_temporary_mesh_copy(monkeypatch):
    local = torch.tensor([2.0, 3.0])
    original_storage = local.untyped_storage().data_ptr()
    calls = {}

    class Mesh:
        device_type = "cuda"

    class DTensor:
        device_mesh = Mesh()
        placements = ("shard",)
        shape = torch.Size((4,))

        def to_local(self):
            return local

        def stride(self):
            return (1,)

        def full_tensor(self):
            raise AssertionError("must gather the temporary CUDA DTensor")

    class Temporary:
        def full_tensor(self):
            calls["gathered"] = True
            return torch.tensor([2.0, 3.0, 5.0, 7.0])

    def fake_to(tensor, device, *args, **kwargs):
        assert tensor is not local or device == "cuda"
        calls["device"] = device
        return tensor.clone()

    def from_local(tensor, mesh, placements, **kwargs):
        calls["local"] = tensor
        calls["mesh"] = mesh
        calls["placements"] = placements
        calls["kwargs"] = kwargs
        return Temporary()

    from torch.distributed.tensor import DTensor as TorchDTensor

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(TorchDTensor, "from_local", from_local)

    full = _gather_adapter_tensor(DTensor())

    torch.testing.assert_close(full, torch.tensor([2.0, 3.0, 5.0, 7.0]))
    torch.testing.assert_close(local, torch.tensor([2.0, 3.0]))
    assert local.untyped_storage().data_ptr() == original_storage
    assert calls == {
        "device": "cuda",
        "local": calls["local"],
        "mesh": DTensor.device_mesh,
        "placements": DTensor.placements,
        "kwargs": {
            "run_check": False,
            "shape": torch.Size((4,)),
            "stride": (1,),
        },
        "gathered": True,
    }


def test_plain_adapter_tensor_is_detached_without_collective():
    adapter = torch.nn.Parameter(torch.tensor([1.0]))

    gathered = _gather_adapter_tensor(adapter)

    assert gathered.data_ptr() == adapter.data.data_ptr()
    assert not gathered.requires_grad


def test_fsdp2_adapter_save_ignores_frozen_base_parameters(monkeypatch, tmp_path):
    saved = {}

    class FrozenBase:
        def full_tensor(self):
            raise AssertionError("frozen base must not be gathered")

    adapter = torch.nn.Parameter(torch.tensor([4.0]))
    config = SimpleNamespace(save_pretrained=lambda output_dir: None)
    model = SimpleNamespace(
        active_adapter="default",
        peft_config={"default": config},
        named_parameters=lambda: iter(
            (("base.weight", FrozenBase()), ("lora_A.default.weight", adapter))
        ),
        named_modules=lambda: iter(()),
    )

    from peft.utils import save_and_load
    from safetensors import torch as safetensors_torch

    monkeypatch.setattr(
        save_and_load, "get_peft_model_state_dict", lambda model, state_dict: state_dict
    )
    monkeypatch.setattr(
        safetensors_torch,
        "save_file",
        lambda state_dict, output: saved.update(state_dict),
    )

    assert save_fsdp2_lora_adapter(model, str(tmp_path))
    assert saved.keys() == {"lora_A.default.weight"}
    torch.testing.assert_close(saved["lora_A.default.weight"], adapter.detach())


def test_fsdp2_adapter_save_strips_checkpoint_wrapper_names(monkeypatch, tmp_path):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors import torch as safetensors_torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.q(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(0)
    model = get_peft_model(Model(), LoraConfig(r=2, target_modules=["q"]))
    expected = get_peft_model_state_dict(model)
    for i, layer in enumerate(model.base_model.model.layers):
        model.base_model.model.layers[i] = checkpoint_wrapper(layer)

    saved = {}
    monkeypatch.setattr(
        safetensors_torch,
        "save_file",
        lambda state_dict, output: saved.update(state_dict),
    )
    assert save_fsdp2_lora_adapter(model, str(tmp_path))
    assert saved.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(saved[key], value)
