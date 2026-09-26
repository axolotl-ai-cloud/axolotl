"""Actual two-rank native-NVFP4 merge-aware FSDP2 trainer resume worker."""

import datetime
import faulthandler
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from axolotl.integrations.kernels.merge_aware_setup import configure_native_merge_aware
from axolotl.monkeypatch.torchao_nvfp4_merge import (
    quantize_native_effective_weight,
)
from axolotl.monkeypatch.torchao_nvfp4_merge_metadata import (
    validate_native_merge_aware_header,
    validate_native_merge_aware_target,
)
from axolotl.utils.dict import DictDefault

WORLD_SIZE = 2
TARGET = "model.layers.0.self_attn.q_proj.weight"


def _local(value):
    return getattr(value, "_local_tensor", value)


def _cuda_full(value):
    local = _local(value)
    offload = os.environ["NVFP4_FSDP2_CPU_OFFLOAD"] == "1"
    if offload:
        assert local.device.type == "cpu"
    value = value.detach().to(torch.device("cuda", torch.cuda.current_device()))
    if offload:
        assert local.device.type == "cpu"
    return value.full_tensor() if hasattr(value, "full_tensor") else value


def _canonical_lora_name(name):
    for kind in ("A", "B"):
        name = name.replace(f".lora_{kind}.default.weight", f".lora_{kind}.weight")
    return name


def _canonical_lora_values(values):
    canonical = {_canonical_lora_name(name): value for name, value in values.items()}
    assert len(canonical) == len(values)
    return canonical


def _lora_values(model):
    values = {}
    for name, parameter in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        values[name] = _cuda_full(parameter).cpu().clone()
    assert values and all("lora_" in name for name in values)
    return _canonical_lora_values(values)


def _optimizer_moments(trainer, model):
    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    moments = {}
    for name, parameter in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        state = optimizer.state.get(parameter)
        if state is None:
            raise AssertionError(f"missing optimizer state for {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in state:
                raise AssertionError(f"missing {key} for {name}")
            moments[f"{name}:{key}"] = _cuda_full(state[key]).float().cpu().clone()
    return moments


def _scheduler_state(trainer):
    state = trainer.lr_scheduler.state_dict()
    return {
        "last_epoch": state["last_epoch"],
        "_step_count": state["_step_count"],
        "_last_lr": tuple(state["_last_lr"]),
    }


def _make_base(path):
    from axolotl.utils.quantization import quantize_model, save_quantized_model
    from axolotl.utils.schemas.enums import TorchAOQuantDType

    torch.manual_seed(7)
    dynamic = os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1"
    hidden_size = 128 if dynamic else 32
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=1,
            num_attention_heads=hidden_size // 16,
            num_key_value_heads=hidden_size // 16,
            pad_token_id=0,
        )
    ).bfloat16()
    quantize_model(
        model,
        TorchAOQuantDType.nvfp4,
        activation_dtype=(
            TorchAOQuantDType.nvfp4
            if os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1"
            else None
        ),
    )
    save_quantized_model(model, path)


def _model_with_lora(base, device):
    torch.manual_seed(19)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(
        device
    )
    model = get_peft_model(
        model,
        LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, target_modules=["q_proj"]),
    )
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data.normal_(mean=0.0, std=0.05)
    return model


def _config(cpu_ram_efficient, cpu_offload):
    fsdp_config = {
        "fsdp_version": 2,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "state_dict_type": "FULL_STATE_DICT",
        "reshard_after_forward": True,
        "cpu_ram_efficient_loading": cpu_ram_efficient,
        "cpu_offload": cpu_offload,
    }
    return DictDefault(
        {
            "adapter": "lora",
            "fsdp_version": 2,
            "fsdp_config": fsdp_config,
            "nvfp4_merge_aware": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_parallel_size": 1,
        }
    )


def _train(model, output, cfg, resume=None):
    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.training_args import AxolotlTrainingArguments
    from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
        NativeNVFP4MergeMetadataCallback,
    )

    rows = torch.arange(1, 25).reshape(6, 4).remainder(31).tolist()
    dataset = Dataset.from_dict(
        {"input_ids": rows, "labels": rows, "attention_mask": [[1] * 4] * 6}
    )
    args = AxolotlTrainingArguments(
        output_dir=str(output),
        max_steps=2,
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        lr_scheduler_type="linear",
        warmup_steps=0,
        bf16=True,
        report_to=[],
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        disable_tqdm=True,
        seed=29,
        fsdp="full_shard",
        fsdp_config=cfg.fsdp_config,
    )
    trainer = AxolotlTrainer(model=model, args=args, train_dataset=dataset)
    trainer.axolotl_cfg = cfg
    from torch.distributed.fsdp import CPUOffloadPolicy

    assert (
        isinstance(trainer.accelerator.state.fsdp_plugin.cpu_offload, CPUOffloadPolicy)
        == cfg.fsdp_config.cpu_offload
    )
    # This direct trainer fixture bypasses TrainerBuilderBase callback registration.
    trainer.add_callback(NativeNVFP4MergeMetadataCallback())
    assert any(
        isinstance(callback, NativeNVFP4MergeMetadataCallback)
        for callback in trainer.callback_handler.callbacks
    )
    trainer.train(resume_from_checkpoint=resume)
    _request_payload_assertions(model, prepared=True)
    return trainer


def _assert_close(actual, expected, label):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=label)


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
            f"NATIVE_NVFP4_FSDP2_MERGE_AWARE_FAILURE rank={dist.get_rank()} stage={stage}",
            file=sys.stderr,
            flush=True,
        )
        print(trace, file=sys.stderr, flush=True)
    _sync_stage_error(stage, trace)


def _merge_aware_assertions(model, calls):
    lora = next(module for module in model.modules() if hasattr(module, "lora_A"))
    if os.environ.get("NVFP4_FSDP2_DYNAMIC_ACTIVATION") == "1":
        weight = lora.get_base_layer().weight
        weight = getattr(weight, "_local_tensor", weight)
        assert weight.act_quant_kwargs.use_dynamic_per_tensor_scale
        assert model._axolotl_native_nvfp4_dynamic_input_gradients
        non_target = next(
            module
            for name, module in model.named_modules()
            if name.endswith("mlp.gate_proj")
        )
        assert hasattr(non_target, "_axolotl_dynamic_nvfp4_ste_orig_forward")
    assert hasattr(lora, "_axolotl_fsdp_native_orig_forward")
    assert lora.forward.__func__.__name__ == "_fsdp_native_forward"
    assert calls[0] > 0
    assert not getattr(model, "_axolotl_merge_aware_unsupported", False)
    assert not any(
        getattr(module, "_axolotl_merge_aware_unsupported", False)
        for module in model.modules()
    )


def _request_payload_assertions(model, prepared=False):
    assert getattr(model, "_axolotl_native_nvfp4_merge_aware_requested", False)
    assert getattr(model, "_axolotl_native_nvfp4_metadata_requested", False)
    if not prepared:
        return
    metadata = getattr(model, "_axolotl_native_nvfp4_metadata", None)
    assert isinstance(metadata, dict)
    assert set(metadata["targets"]) == {TARGET}


def _checkpoint_header(path, original_weight):
    config = json.loads((path / "adapter_config.json").read_text())
    metadata = config["nvfp4_merge_aware"]
    assert validate_native_merge_aware_header(metadata)
    assert set(metadata["targets"]) == {TARGET}
    assert validate_native_merge_aware_target(metadata, TARGET, original_weight)
    tensors = load_file(path / "adapter_model.safetensors")
    assert tensors and all("lora_" in name for name in tensors)
    return metadata


def _saved_lora_values(path):
    values = load_file(path / "adapter_model.safetensors")
    assert values and all("lora_" in name for name in values)
    return _canonical_lora_values(values)


def _factor(values, kind):
    matches = [value for name, value in values.items() if f"lora_{kind}" in name]
    assert len(matches) == 1, f"expected one LoRA {kind} factor, got {len(matches)}"
    return matches[0]


def _assert_nvfp4_bytes(actual, expected):
    for name in ("qdata", "scale"):
        actual_bytes = (
            getattr(actual, name).detach().contiguous().view(torch.uint8).cpu()
        )
        expected_bytes = (
            getattr(expected, name).detach().contiguous().view(torch.uint8).cpu()
        )
        torch.testing.assert_close(
            actual_bytes,
            expected_bytes,
            rtol=0,
            atol=0,
            msg=f"merged NVFP4 {name} bytes",
        )


def main():
    faulthandler.dump_traceback_later(180, repeat=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    rank = dist.get_rank()
    failure = None
    try:
        import axolotl.monkeypatch.torchao_nvfp4_fsdp_lora as bridge
        from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2

        patch_accelerate_fsdp2()
        root = Path(os.environ["NVFP4_FSDP2_RESUME_ROOT"])
        base = root / "base"
        cpu_ram_efficient = os.environ["NVFP4_FSDP2_CPU_RAM_EFFICIENT"] == "1"
        cpu_offload = os.environ["NVFP4_FSDP2_CPU_OFFLOAD"] == "1"
        if rank == 0:
            _make_base(base)
        _run_stage("base checkpoint creation", dist.barrier)

        calls = [0]
        original_forward = bridge.native_nvfp4_merge_aware_linear

        def counted_forward(*args, **kwargs):
            calls[0] += 1
            return original_forward(*args, **kwargs)

        bridge.native_nvfp4_merge_aware_linear = counted_forward
        model = _model_with_lora(base, torch.device("cuda", local_rank))
        original_base = (
            next(
                parameter
                for parameter in model.parameters()
                if type(parameter).__name__ == "NVFP4Tensor"
            )
            .detach()
            .clone()
        )
        assert original_base.is_swizzled_scales
        cfg = _config(cpu_ram_efficient, cpu_offload)
        configure_native_merge_aware(cfg, model, sharded_backend="FSDP")
        from axolotl.monkeypatch.torchao_lora import enable_native_nvfp4_lora_training

        assert enable_native_nvfp4_lora_training(model)
        _request_payload_assertions(model)

        uninterrupted = [None]

        def train_uninterrupted():
            uninterrupted[0] = _train(model, root / "uninterrupted", cfg)

        _run_stage("uninterrupted merge-aware FSDP2 training", train_uninterrupted)
        trainer = uninterrupted[0]
        _run_stage(
            "merge-aware bridge verification",
            lambda: _merge_aware_assertions(model, calls),
        )
        checkpoint = root / "uninterrupted" / "checkpoint-1"
        _run_stage(
            "checkpoint metadata and adapter verification",
            lambda: _checkpoint_header(checkpoint, original_base),
        )
        reference_factors = [None]
        reference_moments = [None]
        reference_scheduler = [None]
        reference_logits = [None]

        def snapshot_uninterrupted():
            reference_factors[0] = _lora_values(model)
            reference_moments[0] = _optimizer_moments(trainer, model)
            reference_scheduler[0] = _scheduler_state(trainer)
            with torch.no_grad():
                reference_logits[0] = (
                    model(input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda"))
                    .logits.detach()
                    .float()
                    .cpu()
                )

        _run_stage("uninterrupted final snapshot", snapshot_uninterrupted)
        assert trainer.state.global_step == 2

        resumed = _model_with_lora(base, torch.device("cuda", local_rank))
        resumed_cfg = _config(cpu_ram_efficient, cpu_offload)
        configure_native_merge_aware(resumed_cfg, resumed, sharded_backend="FSDP")
        assert enable_native_nvfp4_lora_training(resumed)
        _request_payload_assertions(resumed)
        resumed_trainer = [None]

        def train_resumed():
            resumed_trainer[0] = _train(
                resumed, root / "resumed", resumed_cfg, str(checkpoint)
            )

        _run_stage("checkpoint-1 merge-aware FSDP2 resume", train_resumed)
        _run_stage(
            "resumed merge-aware bridge verification",
            lambda: _merge_aware_assertions(resumed, calls),
        )

        def compare_resume():
            for name, value in _lora_values(resumed).items():
                _assert_close(value, reference_factors[0][name], name)
            for name, value in _optimizer_moments(resumed_trainer[0], resumed).items():
                _assert_close(value, reference_moments[0][name], name)
            assert _scheduler_state(resumed_trainer[0]) == reference_scheduler[0]
            assert (
                resumed_trainer[0].state.global_step == trainer.state.global_step == 2
            )
            with torch.no_grad():
                logits = (
                    resumed(input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda"))
                    .logits.detach()
                    .float()
                    .cpu()
                )
            _assert_close(logits, reference_logits[0], "final quantized logits")

        _run_stage(
            "resume factor optimizer scheduler and quantized parity", compare_resume
        )
        export = root / "export"

        def export_adapter():
            from axolotl.train import save_trained_model

            resumed_cfg.output_dir = str(export)
            save_trained_model(resumed_cfg, resumed_trainer[0], resumed)

        _run_stage("final gathered adapter export", export_adapter)
        _run_stage(
            "final gathered adapter metadata verification",
            lambda: _checkpoint_header(export, original_base),
        )
        exported_factors = [None]

        def verify_exported_factors():
            exported_factors[0] = _saved_lora_values(export)
            assert set(exported_factors[0]) == set(reference_factors[0])
            for name, value in exported_factors[0].items():
                _assert_close(value, reference_factors[0][name], name)

        _run_stage(
            "final gathered adapter factor verification", verify_exported_factors
        )

        def merge_export_and_verify():
            if rank != 0:
                return
            from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient

            merged = root / "merged"
            merge_lora_sharded_efficient(base, export, merged, device="cpu")
            merged_model = AutoModelForCausalLM.from_pretrained(
                merged, torch_dtype=torch.bfloat16
            ).to(torch.device("cuda", local_rank))
            merged_model.eval()
            actual_weight = merged_model.model.layers[0].self_attn.q_proj.weight
            assert type(actual_weight).__name__ == "NVFP4Tensor"
            expected_weight = quantize_native_effective_weight(
                original_base,
                _factor(exported_factors[0], "A").to(original_base.device),
                _factor(exported_factors[0], "B").to(original_base.device),
                1.0,
            )
            _assert_nvfp4_bytes(actual_weight, expected_weight)
            with torch.no_grad():
                merged_logits = (
                    merged_model(input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda"))
                    .logits.detach()
                    .float()
                    .cpu()
                )
            _assert_close(merged_logits, reference_logits[0], "merged export logits")

        _run_stage(
            "format-preserving merged export verification", merge_export_and_verify
        )
        _run_stage("completion barrier", dist.barrier)
        print("NATIVE_NVFP4_FSDP2_MERGE_AWARE_RESUME_PASS", rank, flush=True)
    except BaseException:
        failure = traceback.format_exc()
        print(
            f"NATIVE_NVFP4_FSDP2_MERGE_AWARE_FAILURE rank={rank}",
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
                                "NATIVE_NVFP4_FSDP2_MERGE_AWARE_RANK_FAILURE "
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
