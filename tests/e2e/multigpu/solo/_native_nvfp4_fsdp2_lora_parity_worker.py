"""Worker for two-rank FSDP2 static native-NVFP4 LoRA parity."""

import datetime
import faulthandler
import os
import sys
import traceback
import types

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fsdp import (
    normalize_dense_nvfp4_scales,
    patch_nvfp4_fsdp,
)
from axolotl.monkeypatch.torchao_nvfp4_fsdp_lora import (
    install_fsdp_native_nvfp4_merge_aware_lora_linears,
)
from axolotl.monkeypatch.torchao_nvfp4_merge import (
    native_nvfp4_merge_aware_linear,
    quantize_native_effective_weight,
)

WORLD_SIZE = 2
WIDTH = 128 if os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1" else 16
RANK = 4
LOCAL_BATCH = 2
LEARNING_RATE = 0.01


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


def _fixture_tensor(shape, start, scale):
    return (
        torch.arange(
            start, start + torch.tensor(shape).prod().item(), dtype=torch.float32
        )
        .reshape(shape)
        .mul(scale)
    )


def _initial_fixture():
    if os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1":
        base = _fixture_tensor((WIDTH, WIDTH), -128, 1 / 4096)
        lora_a = _fixture_tensor((RANK, WIDTH), 1, 1 / 4096)
        lora_b = _fixture_tensor((WIDTH, RANK), -31, 1 / 4096)
        inputs = _fixture_tensor((WORLD_SIZE * LOCAL_BATCH, WIDTH), 11, 1 / 4096)
    else:
        base = _fixture_tensor((WIDTH, WIDTH), -128, 1 / 512)
        lora_a = _fixture_tensor((RANK, WIDTH), 1, 1 / 97)
        lora_b = _fixture_tensor((WIDTH, RANK), -31, 1 / 89)
        inputs = _fixture_tensor((WORLD_SIZE * LOCAL_BATCH, WIDTH), 11, 1 / 53)
    return base, lora_a, lora_b, inputs


def _native_base(dense):
    kwargs = {}
    if os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1":
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            QuantizeTensorToNVFP4Kwargs,
        )

        kwargs = {
            "per_tensor_scale": torch.tensor(1.125, device="cuda"),
            "act_per_tensor_scale": torch.tensor(1.25, device="cuda"),
            "act_quant_kwargs": QuantizeTensorToNVFP4Kwargs(
                use_dynamic_per_tensor_scale=True
            ),
        }
    base = NVFP4Tensor.to_nvfp4(dense.to("cuda"), **kwargs)
    normalize_dense_nvfp4_scales(base)
    return base


def _factor_values(lora, *, gradients):
    values = {}
    offload = os.environ["NVFP4_FSDP2_OFFLOAD"] == "1"
    for name, module in (
        ("A", lora.lora_A["default"]),
        ("B", lora.lora_B["default"]),
    ):
        value = module.weight.grad if gradients else module.weight
        if value is None:
            kind = "gradient" if gradients else "weight"
            raise AssertionError(f"missing LoRA {name} {kind}")
        if not hasattr(value, "full_tensor"):
            raise AssertionError(f"FSDP did not shard LoRA {name} as a DTensor")
        local_value = getattr(value, "_local_tensor", value)
        if offload:
            assert local_value.device.type == "cpu"
        value = value.detach().to(torch.device("cuda", torch.cuda.current_device()))
        if offload:
            assert local_value.device.type == "cpu"
        value = value.full_tensor()
        values[name] = value.detach().float().cpu().clone()
    return values


def _capture_live_base(base):
    with torch.no_grad():
        captured = base(_axolotl_materialize_weight=True)
    if type(captured).__name__ != "NVFP4Tensor":
        raise AssertionError(
            "FSDP base forward did not materialize a native NVFP4 weight"
        )
    return captured


def _assert_close(actual, expected, label):
    expected = expected.to(actual.device, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5, msg=label)


def _assert_nvfp4_bytes(actual, expected):
    for name in ("qdata", "scale"):
        actual_bytes = (
            getattr(actual, name).detach().contiguous().view(torch.uint8).cpu()
        )
        expected_bytes = (
            getattr(expected, name).detach().contiguous().view(torch.uint8).cpu()
        )
        assert actual_bytes.shape == expected_bytes.shape, name
        assert torch.equal(actual_bytes, expected_bytes), name


def _serial_reference():
    base_dense, initial_a, initial_b, inputs = _initial_fixture()
    base = _native_base(base_dense)
    lora_a = initial_a.cuda().requires_grad_(True)
    lora_b = initial_b.cuda().requires_grad_(True)
    dynamic = os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1"
    if dynamic:
        serial_inputs = [
            value.cuda().requires_grad_(True)
            for value in inputs.chunk(WORLD_SIZE, dim=0)
        ]
        outputs = torch.cat(
            [
                native_nvfp4_merge_aware_linear(value, None, base, lora_a, lora_b, 1.0)
                for value in serial_inputs
            ]
        )
    else:
        serial_inputs = inputs.cuda().requires_grad_(True)
        outputs = native_nvfp4_merge_aware_linear(
            serial_inputs, None, base, lora_a, lora_b, 1.0
        )
    outputs.square().mean().backward()
    with torch.no_grad():
        lora_a.add_(lora_a.grad, alpha=-LEARNING_RATE)
        lora_b.add_(lora_b.grad, alpha=-LEARNING_RATE)
        if dynamic:
            updated_outputs = torch.cat(
                [
                    native_nvfp4_merge_aware_linear(
                        value.detach(), None, base, lora_a, lora_b, 1.0
                    )
                    for value in serial_inputs
                ]
            )
            input_grads = torch.cat([value.grad for value in serial_inputs])
        else:
            updated_outputs = native_nvfp4_merge_aware_linear(
                serial_inputs.detach(), None, base, lora_a, lora_b, 1.0
            )
            input_grads = serial_inputs.grad
        effective = quantize_native_effective_weight(base, lora_a, lora_b, 1.0)
    return {
        "outputs": outputs.detach().cpu(),
        "input_grads": input_grads.detach().cpu(),
        "grads": {"A": lora_a.grad.detach().cpu(), "B": lora_b.grad.detach().cpu()},
        "updated_outputs": updated_outputs.detach().cpu(),
        "updated_factors": {"A": lora_a.detach().cpu(), "B": lora_b.detach().cpu()},
        "effective_qdata": effective.qdata.detach().cpu(),
        "effective_scale": effective.scale.detach().cpu(),
    }


def _sync_stage_error(stage, local_trace):
    traces = [None] * dist.get_world_size()
    dist.all_gather_object(traces, local_trace)
    if any(traces):
        details = "\n".join(
            f"rank {index}:\n{trace}"
            for index, trace in enumerate(traces)
            if trace is not None
        )
        raise RuntimeError(f"{stage} failed:\n{details}")


def _run_stage(stage, action):
    trace = None
    try:
        action()
    except BaseException:
        trace = traceback.format_exc()
        print(
            f"NATIVE_NVFP4_FSDP2_LORA_FAILURE rank={dist.get_rank()} stage={stage}",
            file=sys.stderr,
            flush=True,
        )
        print(trace, file=sys.stderr, flush=True)
    _sync_stage_error(stage, trace)


def _install_model():
    base_dense, initial_a, initial_b, _ = _initial_fixture()
    model = get_peft_model(
        Toy().cuda(),
        LoraConfig(
            r=RANK,
            lora_alpha=RANK,
            lora_dropout=0.0,
            target_modules=["proj"],
        ),
    )
    lora = model.base_model.model.proj
    base = lora.get_base_layer()
    base.weight = torch.nn.Parameter(_native_base(base_dense), requires_grad=False)
    lora.lora_A["default"].weight.data.copy_(initial_a.cuda())
    lora.lora_B["default"].weight.data.copy_(initial_b.cuda())
    assert torch.count_nonzero(lora.lora_B["default"].weight).item() > 0
    assert type(base.weight).__name__ == "NVFP4Tensor"

    offload_policy = (
        CPUOffloadPolicy() if os.environ["NVFP4_FSDP2_OFFLOAD"] == "1" else None
    )
    shard_kwargs = {"offload_policy": offload_policy} if offload_policy else {}
    fully_shard(lora.lora_A["default"], **shard_kwargs)
    fully_shard(lora.lora_B["default"], **shard_kwargs)
    fully_shard(base, **shard_kwargs)
    fully_shard(model, **shard_kwargs)

    installed = install_fsdp_native_nvfp4_merge_aware_lora_linears(model)
    assert installed == 1, f"expected one FSDP native wrapper, installed {installed}"
    assert hasattr(lora, "_axolotl_fsdp_native_orig_forward")
    assert lora.forward.__func__.__name__ == "_fsdp_native_forward"

    def ordinary_fallback(_self, *_args, **_kwargs):
        raise AssertionError("merge-aware FSDP wrapper used its ordinary LoRA fallback")

    lora._axolotl_fsdp_native_orig_forward = types.MethodType(ordinary_fallback, lora)
    return model, lora


def main():
    faulthandler.dump_traceback_later(120, repeat=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    patch_nvfp4_fsdp()
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    rank = dist.get_rank()
    failure = None
    try:
        if dist.get_world_size() != WORLD_SIZE:
            raise AssertionError(f"expected {WORLD_SIZE} ranks")

        reference = [None]

        def build_reference():
            reference[0] = _serial_reference()

        _run_stage("serial global-batch reference", build_reference)
        reference = reference[0]

        model_box = [None]
        lora_box = [None]

        def construct_fsdp_model():
            model_box[0], lora_box[0] = _install_model()

        _run_stage(
            "FSDP construction and production wrapper installation",
            construct_fsdp_model,
        )
        model, lora = model_box[0], lora_box[0]
        _, initial_a, initial_b, global_inputs = _initial_fixture()
        local_inputs = (
            global_inputs.chunk(WORLD_SIZE, dim=0)[rank].cuda().requires_grad_(True)
        )
        assert not torch.equal(global_inputs[:LOCAL_BATCH], global_inputs[LOCAL_BATCH:])

        initial_factors = [None]
        _run_stage(
            "canonical initial factor gathering",
            lambda: initial_factors.__setitem__(
                0, _factor_values(lora, gradients=False)
            ),
        )

        def verify_initial_factors():
            _assert_close(initial_factors[0]["A"].cuda(), initial_a.cuda(), "initial A")
            _assert_close(initial_factors[0]["B"].cuda(), initial_b.cuda(), "initial B")

        _run_stage("canonical initial factor parity", verify_initial_factors)

        outputs = [None]

        def forward_and_backward():
            outputs[0] = model(local_inputs)
            outputs[0].square().mean().backward()

        _run_stage("FSDP routed forward and backward", forward_and_backward)
        local_slice = slice(rank * LOCAL_BATCH, (rank + 1) * LOCAL_BATCH)

        def verify_outputs_and_input_grads():
            _assert_close(
                outputs[0], reference["outputs"][local_slice].cuda(), "routed output"
            )
            _assert_close(
                local_inputs.grad,
                reference["input_grads"][local_slice].cuda() * WORLD_SIZE,
                "routed input gradient",
            )

        _run_stage(
            "routed output and input-gradient parity", verify_outputs_and_input_grads
        )

        gathered_grads = [None]
        _run_stage(
            "canonical LoRA gradient gathering",
            lambda: gathered_grads.__setitem__(0, _factor_values(lora, gradients=True)),
        )

        def verify_factor_grads():
            for name in ("A", "B"):
                _assert_close(
                    gathered_grads[0][name].cuda(),
                    reference["grads"][name].cuda(),
                    f"distributed-average {name} gradient",
                )

        _run_stage("LoRA gradient parity", verify_factor_grads)

        optimizer = torch.optim.SGD(
            [lora.lora_A["default"].weight, lora.lora_B["default"].weight],
            lr=LEARNING_RATE,
        )
        _run_stage("FSDP SGD update", optimizer.step)

        updated_factors = [None]
        _run_stage(
            "canonical updated factor gathering",
            lambda: updated_factors.__setitem__(
                0, _factor_values(lora, gradients=False)
            ),
        )

        def verify_updated_factors():
            for name in ("A", "B"):
                _assert_close(
                    updated_factors[0][name].cuda(),
                    reference["updated_factors"][name].cuda(),
                    f"updated {name}",
                )

        _run_stage("LoRA SGD parity", verify_updated_factors)

        live_base = [None]
        _run_stage(
            "live FSDP native base capture",
            lambda: live_base.__setitem__(0, _capture_live_base(lora.get_base_layer())),
        )

        def verify_updated_output_and_encoding():
            if os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1":
                assert live_base[0].act_quant_kwargs.use_dynamic_per_tensor_scale
                assert live_base[0].per_tensor_scale is not None
                assert live_base[0].act_per_tensor_scale is not None
            updated_output = model(local_inputs.detach())
            _assert_close(
                updated_output,
                reference["updated_outputs"][local_slice].cuda(),
                "updated routed output",
            )
            actual = quantize_native_effective_weight(
                live_base[0],
                updated_factors[0]["A"].cuda(),
                updated_factors[0]["B"].cuda(),
                1.0,
            )
            expected = types.SimpleNamespace(
                qdata=reference["effective_qdata"].cuda(),
                scale=reference["effective_scale"].cuda(),
            )
            _assert_nvfp4_bytes(actual, expected)

        _run_stage(
            "updated routed output and NVFP4 byte parity",
            verify_updated_output_and_encoding,
        )
        _run_stage("completion barrier", dist.barrier)
        if rank == 0:
            print("NATIVE_NVFP4_FSDP2_LORA_PARITY_PASS", flush=True)
    except BaseException:
        failure = traceback.format_exc()
        print(
            f"NATIVE_NVFP4_FSDP2_LORA_FAILURE rank={rank}",
            file=sys.stderr,
            flush=True,
        )
        print(failure, file=sys.stderr, flush=True)
    finally:
        try:
            if dist.is_initialized():
                failures = [None] * dist.get_world_size()
                dist.all_gather_object(failures, failure)
                if any(failures):
                    for index, trace in enumerate(failures):
                        if trace is not None:
                            print(
                                "NATIVE_NVFP4_FSDP2_LORA_RANK_FAILURE "
                                f"rank={index}\n{trace}",
                                file=sys.stderr,
                                flush=True,
                            )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            faulthandler.cancel_dump_traceback_later()
    if failure is not None:
        raise RuntimeError(f"rank {rank} failed; original traceback was logged")


if __name__ == "__main__":
    main()
