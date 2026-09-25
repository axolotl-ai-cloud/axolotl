import os
import shutil

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from transformers import LlamaConfig, LlamaForCausalLM, TorchAoConfig

from axolotl.monkeypatch.torchao_tp import native_nvfp4_tp_checkpoint_loading
from axolotl.monkeypatch.torchao_tp_lora import prepare_native_nvfp4_tp_lora


def _factors(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def _shard_axis(name):
    if "q_proj" in name and "lora_B" in name:
        return 0
    if "o_proj" in name and "lora_A" in name:
        return 1
    return None


def _tolerances(dtype):
    return (1e-5, 1e-6, 1e-5) if dtype == torch.float32 else (5e-2, 5e-2, 5e-2)


def _assert_close(actual, expected, run_dtype):
    rtol, atol, normalized_error = _tolerances(run_dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    expected_norm = torch.linalg.vector_norm(expected.float())
    assert expected_norm > 0
    error = torch.linalg.vector_norm(actual.float() - expected.float()) / expected_norm
    assert error <= normalized_error
    return error.item()


def _copy_serial_factors(factors, serial_factors, rank, tp_group):
    for name, parameter in factors.items():
        value = serial_factors[name].to(device="cuda", dtype=parameter.dtype)
        axis = _shard_axis(name)
        if axis is not None:
            value = value.chunk(dist.get_world_size(tp_group), dim=axis)[rank]
        parameter.data.copy_(value)


def _gather_factor_values(factors, *, gradients, tp_group):
    gathered_factors = {}
    for name, parameter in factors.items():
        value = parameter.grad if gradients else parameter.data
        assert value is not None
        axis = _shard_axis(name)
        gathered = [
            torch.empty_like(value) for _ in range(dist.get_world_size(tp_group))
        ]
        dist.all_gather(gathered, value, group=tp_group)
        gathered_factors[name] = (
            torch.cat(gathered, dim=axis) if axis is not None else gathered,
            axis,
            value.dtype,
        )
    return gathered_factors


def _assert_factor_parity(gathered_factors, expected, run_dtype):
    max_error = 0.0
    for name, (actual, axis, factor_dtype) in gathered_factors.items():
        reference = expected[name].to(device="cuda", dtype=factor_dtype)
        if axis is None:
            for replica in actual:
                max_error = max(max_error, _assert_close(replica, reference, run_dtype))
        else:
            max_error = max(max_error, _assert_close(actual, reference, run_dtype))
    return max_error


def _reference_payload(serial, inputs, labels):
    serial_inputs = inputs.detach().clone().requires_grad_(True)
    output = serial(inputs_embeds=serial_inputs, labels=labels)
    output.loss.backward()
    return {
        "logits": output.logits.detach().cpu(),
        "loss": output.loss.detach().cpu(),
        "input_grad": serial_inputs.grad.detach().cpu(),
        "grads": {
            name: parameter.grad.detach().cpu()
            for name, parameter in _factors(serial).items()
        },
    }


def _broadcast_object(value):
    payload = [value]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _sync_stage_error(stage, error):
    if error is not None:
        print(f"HF_NVFP4_TP_LORA_STAGE_ERROR {stage}: {error}", flush=True)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(
            f"{stage}: " + "; ".join(str(item) for item in errors if item)
        )


def _run_rank_zero_stage(stage, rank, action):
    error = None
    if rank == 0:
        try:
            action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    _sync_stage_error(stage, error)


def _run_stage(stage, action):
    error = None
    try:
        action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _sync_stage_error(stage, error)


def _make_checkpoint(path, dtype):
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
    config.quantization_config = TorchAoConfig(NVFP4WeightOnlyConfig()).to_dict()
    config.save_pretrained(path)
    torch.manual_seed(17)
    model = LlamaForCausalLM(config).eval()
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and module.weight.shape[0] % 16 == 0
            and module.weight.shape[1] % 16 == 0
        ):
            module.weight = torch.nn.Parameter(
                NVFP4Tensor.to_nvfp4(
                    module.weight.detach().to(dtype),
                    per_tensor_scale=torch.tensor(1.0),
                    is_swizzled_scales=True,
                ),
                requires_grad=False,
            )
    flattened, metadata = flatten_tensor_state_dict(model.state_dict())
    save_file(flattened, os.path.join(path, "model.safetensors"), metadata=metadata)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    failure = None
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        tp_group = mesh.get_group("tp")
        dtype = getattr(torch, os.environ["NVFP4_TP_LORA_DTYPE"])
        path = os.environ["NVFP4_TP_GATE_PATH"]

        def make_checkpoint():
            _make_checkpoint(path, dtype)

        _run_rank_zero_stage("checkpoint creation", rank, make_checkpoint)
        dist.barrier()

        config = LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj", "o_proj"],
            lora_dropout=0.0,
        )
        serial = None
        initial = None
        inputs = None

        def make_serial_reference():
            nonlocal serial, initial, inputs
            serial = get_peft_model(
                LlamaForCausalLM.from_pretrained(path, dtype=dtype).cuda(), config
            )
            torch.manual_seed(101)
            for parameter in _factors(serial).values():
                parameter.data.normal_(mean=0.0, std=0.1)
            initial = {
                name: parameter.detach().cpu()
                for name, parameter in _factors(serial).items()
            }
            inputs = torch.randn(1, 3, 64, device="cuda", dtype=dtype)

        _run_rank_zero_stage("serial model preparation", rank, make_serial_reference)
        initial = _broadcast_object(initial)
        inputs = _broadcast_object(None if inputs is None else inputs.cpu()).to("cuda")
        labels = torch.tensor([[1, 2, 3]], device=inputs.device)

        with native_nvfp4_tp_checkpoint_loading(mesh):
            model = LlamaForCausalLM.from_pretrained(
                path, tp_plan="auto", tp_size=2, device_mesh=mesh, dtype=dtype
            )
        model = get_peft_model(model, config)
        assert prepare_native_nvfp4_tp_lora(model)
        assert prepare_native_nvfp4_tp_lora(model)
        factors = _factors(model)
        _copy_serial_factors(factors, initial, rank, tp_group)

        reference = None

        def build_reference_payload():
            nonlocal reference
            reference = _reference_payload(serial, inputs, labels)

        _run_rank_zero_stage("serial reference backward", rank, build_reference_payload)
        reference = _broadcast_object(reference)

        tp_inputs = inputs.detach().clone().requires_grad_(True)
        output = model(inputs_embeds=tp_inputs, labels=labels)
        output.loss.backward()

        def verify_outputs():
            _assert_close(output.logits, reference["logits"].to("cuda"), dtype)
            _assert_close(output.loss, reference["loss"].to("cuda"), dtype)
            _assert_close(tp_inputs.grad, reference["input_grad"].to("cuda"), dtype)

        _run_stage("forward, loss, and input-gradient parity", verify_outputs)
        gathered_gradients = _gather_factor_values(
            factors, gradients=True, tp_group=tp_group
        )
        gradient_error = [0.0]

        def verify_gradients():
            gradient_error[0] = _assert_factor_parity(
                gathered_gradients, reference["grads"], dtype
            )

        _run_stage("LoRA gradient parity", verify_gradients)
        print("HF_NVFP4_TP_LORA_GRADIENT_ERROR", rank, gradient_error[0], flush=True)

        optimizer = torch.optim.SGD(factors.values(), lr=0.01)
        optimizer.step()
        final = None

        def step_serial_optimizer():
            nonlocal final
            torch.optim.SGD(_factors(serial).values(), lr=0.01).step()
            final = {
                name: parameter.detach().cpu()
                for name, parameter in _factors(serial).items()
            }

        _run_rank_zero_stage("serial optimizer step", rank, step_serial_optimizer)
        final = _broadcast_object(final)
        gathered_parameters = _gather_factor_values(
            factors, gradients=False, tp_group=tp_group
        )
        parameter_error = [0.0]

        def verify_parameters():
            parameter_error[0] = _assert_factor_parity(
                gathered_parameters, final, dtype
            )

        _run_stage("LoRA optimizer-step parity", verify_parameters)
        print("HF_NVFP4_TP_LORA_PARAMETER_ERROR", rank, parameter_error[0], flush=True)
        print("HF_NVFP4_TP_LORA_PARITY", rank, "ok", flush=True)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        if any(failures):
            print("HF_NVFP4_TP_LORA_PARITY", rank, failures, flush=True)
        dist.destroy_process_group()
    if any(failures):
        raise RuntimeError("; ".join(str(error) for error in failures if error))


if __name__ == "__main__":
    main()
