"""DeepSpeed native-NVFP4 LoRA parity against a global native reference."""

import os

import deepspeed
import torch
import torch.distributed as dist
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from torch import nn
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
)

IN, OUT, RANK = 128, 128, 8


def _model(device, dynamic=False):
    torch.manual_seed(913)
    base = nn.Linear(IN, OUT, bias=False, dtype=torch.bfloat16, device=device)
    native = NVFP4Tensor.to_nvfp4(
        torch.randn(OUT, IN, dtype=torch.bfloat16, device=device),
        per_tensor_scale=torch.tensor(1.125, device=device),
        is_swizzled_scales=False,
        act_quant_kwargs=(
            QuantizeTensorToNVFP4Kwargs(use_dynamic_per_tensor_scale=True)
            if dynamic
            else None
        ),
    )
    base.weight = nn.Parameter(native, requires_grad=False)
    lora = LoraLinear(
        base,
        adapter_name="default",
        config=LoraConfig(r=RANK, lora_alpha=2 * RANK),
        r=RANK,
        lora_alpha=2 * RANK,
        lora_dropout=0.0,
    )
    lora.lora_A["default"].to(torch.float32)
    lora.lora_B["default"].to(torch.float32)
    with torch.no_grad():
        lora.lora_B["default"].weight.normal_(std=0.1)
    post = nn.Linear(IN, OUT, bias=False, dtype=torch.bfloat16, device=device)
    post.weight = nn.Parameter(
        NVFP4Tensor.to_nvfp4(
            torch.randn(OUT, IN, dtype=torch.bfloat16, device=device),
            per_tensor_scale=torch.tensor(1.25, device=device),
            is_swizzled_scales=False,
            act_quant_kwargs=(
                QuantizeTensorToNVFP4Kwargs(use_dynamic_per_tensor_scale=True)
                if dynamic
                else None
            ),
        ),
        requires_grad=False,
    )
    return nn.Sequential(lora, post)


def _factors(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    }


def _native_bytes(model):
    components = {}
    for name, parameter in model.named_parameters():
        if name.endswith(("_axolotl_nvfp4_qdata", "_axolotl_nvfp4_scale_bytes")):
            components[name] = (
                _compute_parameter(parameter)
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .clone()
            )
        elif type(parameter).__name__ == "NVFP4Tensor":
            components[f"{name}.qdata"] = parameter.qdata.detach().cpu().clone()
            components[f"{name}.scale"] = (
                parameter.scale.detach().contiguous().view(torch.uint8).cpu().clone()
            )
    return components


def _full_gradient(parameter):
    from deepspeed.utils import safe_get_full_grad

    value = safe_get_full_grad(parameter)
    assert value is not None
    return value.detach()


def _full_parameter(parameter):
    from deepspeed.utils import safe_get_full_fp32_param

    value = safe_get_full_fp32_param(parameter)
    return (parameter.detach() if value is None else value).detach()


def _compute_parameter(parameter):
    if not hasattr(parameter, "ds_id"):
        return parameter.detach().clone()
    with deepspeed.zero.GatheredParameters(parameter):
        return parameter.detach().clone()


def _assert_close(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-5, atol=1e-6)


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    stage = int(os.environ["ZERO_STAGE"])
    dynamic = os.environ.get("NATIVE_NVFP4_DEEPSPEED_DYNAMIC") == "1"
    opt_out = os.environ.get("NATIVE_NVFP4_DEEPSPEED_OPT_OUT") == "1"
    model = _model(device, dynamic)
    from axolotl.integrations.kernels.merge_aware_setup import (
        configure_native_merge_aware,
    )
    from axolotl.monkeypatch.torchao_deepspeed import prepare_native_nvfp4_deepspeed
    from axolotl.utils.dict import DictDefault

    configure_native_merge_aware(
        DictDefault(adapter="lora", nvfp4_merge_aware=not opt_out),
        model,
        sharded_backend="DeepSpeed",
    )
    assert prepare_native_nvfp4_deepspeed(model, device, stage)
    canonical_factors = {
        name: parameter.detach().cpu().clone()
        for name, parameter in _factors(model).items()
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=torch.optim.SGD(_factors(model).values(), lr=1e-2),
        config={
            "train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 0.0,
            "bf16": {"enabled": True},
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {"stage": stage},
        },
    )
    import axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora as bridge
    from axolotl.monkeypatch.torchao_nvfp4_merge import (
        quantize_native_effective_weight,
    )

    calls = []
    original = bridge.native_nvfp4_merge_aware_linear

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    bridge.native_nvfp4_merge_aware_linear = counted
    try:
        from types import SimpleNamespace

        bridge.DeepSpeedNativeNVFP4MergeAwareCallback(
            SimpleNamespace(model_wrapped=engine)
        ).on_train_begin(None, None, None)
        if dynamic:
            assert engine.module._axolotl_native_nvfp4_dynamic_input_gradients
            if stage in (1, 2):
                assert hasattr(
                    engine.module[1], "_axolotl_dynamic_nvfp4_ste_orig_forward"
                )
        if opt_out:
            assert getattr(engine.module, "_axolotl_merge_aware_unsupported", False)
            assert not hasattr(
                engine.module[0], "_axolotl_deepspeed_native_orig_forward"
            )
        else:
            assert (
                engine.module._axolotl_native_nvfp4_deepspeed_merge_aware_installed == 1
            )
        frozen_before = _native_bytes(engine.module)
        factors = _factors(engine.module)
        initial_factors = {
            name: _full_parameter(parameter).cpu().clone()
            for name, parameter in factors.items()
        }
        for name, value in initial_factors.items():
            _assert_close(
                value,
                canonical_factors[name].to(factors[name].dtype).float(),
            )
        torch.manual_seed(1000 + rank)
        inputs = torch.randn(
            2, IN, dtype=torch.bfloat16, device=device
        ).requires_grad_()

        reference = _model(device, dynamic)
        reference_factors = _factors(reference)
        with torch.no_grad():
            for name, parameter in reference_factors.items():
                parameter.data = parameter.data.to(factors[name].dtype)
                parameter.copy_(initial_factors[name].to(device, parameter.dtype))
        from axolotl.monkeypatch.torchao_nvfp4_merge import (
            install_native_nvfp4_merge_aware_lora_linears,
        )

        if not opt_out:
            assert install_native_nvfp4_merge_aware_lora_linears(reference) == 1
        if dynamic:
            from axolotl.monkeypatch.torchao_nvfp4_dynamic_ste import (
                install_native_nvfp4_dynamic_input_stes,
            )

            assert install_native_nvfp4_dynamic_input_stes(reference) == 2
            assert hasattr(reference[1], "_axolotl_dynamic_nvfp4_ste_orig_forward")
        reference_inputs = inputs.detach().clone().requires_grad_()
        reference_output = reference(reference_inputs)
        reference_output.float().square().mean().backward()
        for parameter in reference_factors.values():
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())

        output = engine(inputs)
        engine.backward(output.float().square().mean())
        assert bool(calls) is (not opt_out)
        _assert_close(output, reference_output.detach())
        _assert_close(inputs.grad, reference_inputs.grad)
        for name, parameter in factors.items():
            _assert_close(_full_gradient(parameter), reference_factors[name].grad)
            assert _full_gradient(parameter).abs().max() > 0

        master_factors = {
            name: torch.nn.Parameter(value.to(device))
            for name, value in initial_factors.items()
        }
        for name, parameter in master_factors.items():
            parameter.grad = reference_factors[name].grad.detach().float().clone()
        engine.step()
        torch.optim.SGD(master_factors.values(), lr=1e-2).step()
        for name, parameter in factors.items():
            _assert_close(_full_parameter(parameter), master_factors[name])
        with torch.no_grad():
            for name, parameter in reference_factors.items():
                parameter.copy_(master_factors[name].to(parameter.dtype))
        for name, parameter in factors.items():
            _assert_close(_compute_parameter(parameter), reference_factors[name])

        if not opt_out:
            engine_lora = next(
                module
                for module in engine.module.modules()
                if hasattr(module, "lora_A")
            )
            reference_lora = next(
                module for module in reference.modules() if hasattr(module, "lora_A")
            )
            adapter = engine_lora.active_adapters[0]
            actual_merged = quantize_native_effective_weight(
                bridge._materialize_weight(engine_lora.get_base_layer()),
                _compute_parameter(engine_lora.lora_A[adapter].weight),
                _compute_parameter(engine_lora.lora_B[adapter].weight),
                engine_lora.scaling[adapter],
            )
            expected_merged = quantize_native_effective_weight(
                reference_lora.get_base_layer().weight,
                reference_lora.lora_A[adapter].weight,
                reference_lora.lora_B[adapter].weight,
                reference_lora.scaling[adapter],
            )
            assert torch.equal(actual_merged.qdata, expected_merged.qdata)
            assert torch.equal(
                actual_merged.scale.contiguous().view(torch.uint8),
                expected_merged.scale.contiguous().view(torch.uint8),
            )
        assert all(
            torch.equal(value, _native_bytes(engine.module)[name])
            for name, value in frozen_before.items()
        )
        print(f"NATIVE_NVFP4_DEEPSPEED_LORA_MERGE_AWARE_PASS rank={rank}", flush=True)
    finally:
        bridge.native_nvfp4_merge_aware_linear = original
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
