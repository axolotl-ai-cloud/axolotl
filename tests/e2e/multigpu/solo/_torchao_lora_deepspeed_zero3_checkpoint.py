"""Two-rank native TorchAO NVFP4 + ordinary LoRA DeepSpeed checkpoint smoke."""

import importlib
import os
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parents[3] / "monkeypatch"))
_ddp = importlib.import_module("_torchao_lora_ddp")
barrier = _ddp.barrier
model_with_lora = _ddp.model_with_lora


def _local(value):
    return value.to_local() if type(value).__name__ == "DTensor" else value


def adapters(model):
    import deepspeed

    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if hasattr(parameter, "ds_id"):
            with deepspeed.zero.GatheredParameters(parameter):
                assert parameter.numel() == parameter.ds_numel
                result[name] = parameter.detach().float().cpu().clone()
        else:
            result[name] = _local(parameter).detach().float().cpu().clone()
    assert result and any(value.count_nonzero() for value in result.values())
    return result


def optimizer_snapshot(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__module__ + "." + type(value).__name__,
            "fields": {
                field.name: optimizer_snapshot(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": optimizer_snapshot(value.value)}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: optimizer_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(optimizer_snapshot(item) for item in value)
    if type(value).__module__ == "deepspeed.runtime.fp16.loss_scaler":
        return {"class": type(value).__name__, "state": optimizer_snapshot(vars(value))}
    return value


def assert_optimizer_snapshot(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_optimizer_snapshot(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_optimizer_snapshot(left, right)
    else:
        assert actual == expected


def zero3_components(model):
    import deepspeed

    names = getattr(model, "_axolotl_native_nvfp4_zero3_components", ())
    assert names
    parameters = dict(model.named_parameters(remove_duplicate=False))
    result = {}
    for name in sorted(names):
        parameter = parameters[name]
        assert parameter.dtype == torch.uint8 and not parameter.requires_grad
        assert parameter.ds_tensor.numel() < parameter.ds_numel
        assert parameter.numel() == 0
        with deepspeed.zero.GatheredParameters(parameter):
            result[name] = parameter.detach().cpu().clone()
    return result


def native_component_snapshot(model):
    result = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if type(parameter).__name__ != "NVFP4Tensor":
            continue
        prefix = name.removesuffix(".weight")
        result[prefix + "._axolotl_nvfp4_qdata"] = (
            parameter.qdata.detach().cpu().clone()
        )
        result[prefix + "._axolotl_nvfp4_scale_bytes"] = (
            parameter.scale.detach().view(torch.uint8).cpu().clone()
        )
    return result


def native_per_tensor_scale_snapshot(model):
    result = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if type(parameter).__name__ != "NVFP4Tensor":
            continue
        value = parameter.per_tensor_scale
        if value is not None:
            result[name.removesuffix(".weight")] = (
                value.detach().reshape(-1).view(torch.uint8).cpu().clone()
            )
    return result


def zero3_per_tensor_scale_snapshot(model):
    result = {}
    for name, module in model.named_modules(remove_duplicate=False):
        value = getattr(module, "_axolotl_nvfp4_per_tensor_scale_bytes", None)
        if value is not None:
            result[name] = value.detach().cpu().clone()
    return result


def assert_snapshot_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def export_logits(model, base, export, device, zero3_load_config):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers.integrations.deepspeed import (
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )

    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=attention_mask).logits
    with torch.random.fork_rng(devices=[device]):
        unset_hf_deepspeed_config()
        try:
            reloaded = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16
            ).to(device)
            reloaded = PeftModel.from_pretrained(reloaded, export).to(device).eval()
            with torch.no_grad():
                actual = reloaded(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
        finally:
            set_hf_deepspeed_config(zero3_load_config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def train(model, output, resume=None):
    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.training_args import AxolotlTrainingArguments
    from axolotl.utils.dict import DictDefault

    rows = torch.arange(1, 25).reshape(6, 4).remainder(31).tolist()
    dataset = Dataset.from_dict(
        {"input_ids": rows, "labels": rows, "attention_mask": [[1] * 4] * 6}
    )
    args = AxolotlTrainingArguments(
        output_dir=str(output),
        max_steps=2,
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        bf16=True,
        report_to=[],
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        disable_tqdm=True,
        deepspeed={
            "train_batch_size": "auto",
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": int(os.environ["ZERO_STAGE"]),
                "stage3_param_persistence_threshold": 0,
            },
        },
    )
    trainer = AxolotlTrainer(model=model, args=args, train_dataset=dataset)
    trainer.axolotl_cfg = DictDefault(deepspeed=args.deepspeed)
    trainer.train(resume_from_checkpoint=resume)
    return trainer


def configure_zero3_model_loading():
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    return HfTrainerDeepSpeedConfig(
        {
            "train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 0,
            },
        }
    )


def make_base_with_non_bf16_per_tensor_scale(base):
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.utils.quantization import quantize_model, save_quantized_model
    from axolotl.utils.schemas.enums import TorchAOQuantDType

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
        )
    ).bfloat16()
    quantize_model(model, TorchAOQuantDType.nvfp4)
    for parameter in model.parameters():
        if type(parameter).__name__ == "NVFP4Tensor":
            parameter.per_tensor_scale.fill_(1.0012345)
    save_quantized_model(model, base)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    root = Path(os.environ["TORCHAO_LORA_DEEPSPEED_CHECKPOINT_TMP"])
    base = root / "base"
    if dist.get_rank() == 0 and not base.exists():
        make_base_with_non_bf16_per_tensor_scale(base)
    barrier()
    zero3_load_config = configure_zero3_model_loading()
    phase = os.environ["TORCHAO_LORA_DEEPSPEED_PHASE"]
    model = model_with_lora(base, device)
    baseline_components = native_component_snapshot(model)
    baseline_per_tensor_scales = native_per_tensor_scale_snapshot(model)
    expected_scale_bytes = (
        torch.tensor(1.0012345, dtype=torch.float32).reshape(-1).view(torch.uint8)
    )
    assert baseline_per_tensor_scales and all(
        torch.equal(value, expected_scale_bytes)
        for value in baseline_per_tensor_scales.values()
    )
    if phase == "reference":
        before = adapters(model)
        trainer = train(model, root / "run")
        after = adapters(model)
        assert any(not torch.equal(before[name], after[name]) for name in before)
        components = zero3_components(model)
        assert components.keys() == baseline_components.keys()
        for name, value in components.items():
            torch.testing.assert_close(value, baseline_components[name], rtol=0, atol=0)
        assert_snapshot_equal(
            zero3_per_tensor_scale_snapshot(model), baseline_per_tensor_scales
        )
        expected = {
            "adapters": after,
            "optimizer": optimizer_snapshot(trainer.optimizer.state_dict()),
            "scheduler": optimizer_snapshot(trainer.lr_scheduler.state_dict()),
            "rng_cpu": torch.get_rng_state().cpu().clone(),
            "rng_cuda": torch.cuda.get_rng_state(device).cpu().clone(),
        }
        expected_path = root / f"expected-{dist.get_rank()}.pt"
        torch.save(expected, expected_path)
        assert_optimizer_snapshot(
            torch.load(expected_path, map_location="cpu", weights_only=True), expected
        )
        export = root / "export"
        trainer.save_model(export)
        if dist.get_rank() == 0:
            from safetensors.torch import load_file

            exported = load_file(export / "adapter_model.safetensors")
            assert exported and all(value.numel() for value in exported.values())
            assert (export / "adapter_model.safetensors").is_file()
            assert not (export / "model.safetensors").exists()
        export_logits(trainer.model_wrapped, base, export, device, zero3_load_config)
        checkpoint = root / "run" / "checkpoint-1"
        assert checkpoint.is_dir()
        engine_state_path = Path(
            trainer.model_wrapped._get_ckpt_name(checkpoint, "global_step1")
        )
        assert engine_state_path.is_file()
        engine_state = torch.load(
            engine_state_path, map_location="cpu", weights_only=True
        )
        frozen = {
            name for name, param in model.named_parameters() if not param.requires_grad
        }
        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        saved = set(engine_state["module"])
        assert frozen and not (saved & frozen)
        assert trainable <= saved
        assert not (saved & set(components))
    else:
        trainer = train(model, root / "resume", str(root / "run" / "checkpoint-1"))
        expected = torch.load(
            root / f"expected-{dist.get_rank()}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for name, value in adapters(model).items():
            torch.testing.assert_close(
                value, expected["adapters"][name], rtol=0, atol=0
            )
        assert_optimizer_snapshot(
            optimizer_snapshot(trainer.optimizer.state_dict()), expected["optimizer"]
        )
        assert_optimizer_snapshot(
            optimizer_snapshot(trainer.lr_scheduler.state_dict()), expected["scheduler"]
        )
        torch.testing.assert_close(
            torch.get_rng_state(), expected["rng_cpu"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.cuda.get_rng_state(device).cpu(), expected["rng_cuda"], rtol=0, atol=0
        )
        components = zero3_components(model)
        assert components.keys() == baseline_components.keys()
        for name, value in components.items():
            torch.testing.assert_close(value, baseline_components[name], rtol=0, atol=0)
        assert_snapshot_equal(
            zero3_per_tensor_scale_snapshot(model), baseline_per_tensor_scales
        )
        export_logits(
            trainer.model_wrapped, base, root / "export", device, zero3_load_config
        )
    print(f"TORCHAO_LORA_DEEPSPEED_CHECKPOINT_OK rank={dist.get_rank()}", flush=True)
    barrier()
    dist.destroy_process_group()
    del zero3_load_config


if __name__ == "__main__":
    main()
