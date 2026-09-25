"""Two-rank Trainer lifecycle coverage for static native NVFP4 DeepSpeed LoRA."""

import datetime
import faulthandler
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

SOURCE = Path(__file__).with_name("_torchao_lora_deepspeed_zero3_checkpoint.py")
spec = importlib.util.spec_from_file_location("native_ds_checkpoint", SOURCE)
assert spec is not None and spec.loader is not None
checkpoint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checkpoint
spec.loader.exec_module(checkpoint)

TARGET = "model.layers.0.self_attn.q_proj.weight"


def _sync_failure(stage, trace):
    traces = [None] * dist.get_world_size()
    dist.all_gather_object(traces, trace)
    if any(traces):
        details = "\n".join(
            f"rank {rank}:\n{item}"
            for rank, item in enumerate(traces)
            if item is not None
        )
        raise RuntimeError(f"{stage} failed:\n{details}")


def _stage(stage, action):
    trace = None
    try:
        action()
    except BaseException:
        trace = traceback.format_exc()
        print(
            f"NATIVE_NVFP4_DEEPSPEED_LIFECYCLE_FAILURE rank={dist.get_rank()} "
            f"stage={stage}",
            file=sys.stderr,
            flush=True,
        )
        print(trace, file=sys.stderr, flush=True)
    _sync_failure(stage, trace)


def _metadata(path, original_weight):
    from safetensors.torch import load_file

    from axolotl.monkeypatch.torchao_nvfp4_merge_metadata import (
        validate_native_merge_aware_header,
        validate_native_merge_aware_target,
    )

    config = json.loads((path / "adapter_config.json").read_text())
    metadata = config["nvfp4_merge_aware"]
    assert validate_native_merge_aware_header(metadata)
    assert set(metadata["targets"]) == {TARGET}
    assert validate_native_merge_aware_target(metadata, TARGET, original_weight)
    tensors = load_file(path / "adapter_model.safetensors")
    assert tensors and all("lora_" in name for name in tensors)
    return metadata


def _make_model(base, device):
    from axolotl.integrations.kernels.merge_aware_setup import (
        configure_native_merge_aware,
    )
    from axolotl.utils.dict import DictDefault

    torch.manual_seed(19)
    model = checkpoint.model_with_lora(base, device, False)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data.normal_(mean=0.0, std=0.05)
            assert parameter.count_nonzero()
    configure_native_merge_aware(
        DictDefault(adapter="lora", nvfp4_merge_aware=True),
        model,
        sharded_backend="DeepSpeed",
    )
    assert model._axolotl_native_nvfp4_deepspeed_merge_aware_requested
    assert model._axolotl_native_nvfp4_metadata_requested
    return model


def _train(model, output, resume=None):
    from datasets import Dataset

    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.training_args import AxolotlTrainingArguments
    from axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora import (
        DeepSpeedNativeNVFP4MergeAwareCallback,
    )
    from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
        NativeNVFP4MergeMetadataCallback,
    )
    from axolotl.utils.dict import DictDefault

    rows = torch.arange(1, 25).reshape(6, 4).remainder(31).tolist()
    dataset = Dataset.from_dict(
        {"input_ids": rows, "labels": rows, "attention_mask": [[1] * 4] * 6}
    )
    deepspeed = {
        "train_batch_size": "auto",
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
    }
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
        seed=29,
        deepspeed=deepspeed,
    )
    trainer = AxolotlTrainer(model=model, args=args, train_dataset=dataset)
    trainer.axolotl_cfg = DictDefault(
        adapter="lora",
        nvfp4_merge_aware=True,
        deepspeed=deepspeed,
        output_dir=str(output),
    )
    trainer.add_callback(DeepSpeedNativeNVFP4MergeAwareCallback(trainer))
    trainer.add_callback(NativeNVFP4MergeMetadataCallback())
    assert any(
        isinstance(callback, DeepSpeedNativeNVFP4MergeAwareCallback)
        for callback in trainer.callback_handler.callbacks
    )
    assert any(
        isinstance(callback, NativeNVFP4MergeMetadataCallback)
        for callback in trainer.callback_handler.callbacks
    )
    trainer.train(resume_from_checkpoint=resume)
    wrapped = trainer.model_wrapped.module
    assert wrapped._axolotl_native_nvfp4_deepspeed_merge_aware_installed
    assert not getattr(wrapped, "_axolotl_merge_aware_unsupported", False)
    assert not any(
        getattr(module, "_axolotl_merge_aware_unsupported", False)
        for module in wrapped.modules()
    )
    assert getattr(wrapped, "_axolotl_native_nvfp4_metadata_valid", False)
    return trainer


def _logits(model, device):
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        return (
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            .logits.detach()
            .float()
            .cpu()
        )


def _assert_frozen_base_omitted(model, trainer, checkpoint_path):
    engine_state_path = Path(
        trainer.model_wrapped._get_ckpt_name(checkpoint_path, "global_step1")
    )
    assert engine_state_path.is_file()
    engine_state = torch.load(engine_state_path, map_location="cpu", weights_only=True)
    frozen = {
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    saved = set(engine_state["module"])
    assert frozen and not (saved & frozen)
    assert trainable <= saved


def _merge_and_compare(base, export, merged, expected, device):
    from transformers import AutoModelForCausalLM
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config

    from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient

    unset_hf_deepspeed_config()
    merge_lora_sharded_efficient(base, export, merged, device="cpu")
    model = (
        AutoModelForCausalLM.from_pretrained(merged, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    assert type(model.model.layers[0].self_attn.q_proj.weight).__name__ == "NVFP4Tensor"
    torch.testing.assert_close(_logits(model, device), expected, rtol=0, atol=0)


def main():
    faulthandler.dump_traceback_later(180, repeat=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    rank = dist.get_rank()
    failure = None
    try:
        root = Path(os.environ["TORCHAO_LORA_DEEPSPEED_CHECKPOINT_TMP"])
        base = root / "base"
        phase = os.environ["TORCHAO_LORA_DEEPSPEED_PHASE"]
        if rank == 0 and not base.exists():
            checkpoint.make_base_with_non_bf16_per_tensor_scale(base)
        _stage("base checkpoint creation", dist.barrier)

        import axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora as bridge

        calls = [0]
        original_forward = bridge.native_nvfp4_merge_aware_linear

        def counted_forward(*args, **kwargs):
            calls[0] += 1
            return original_forward(*args, **kwargs)

        bridge.native_nvfp4_merge_aware_linear = counted_forward
        try:
            model = _make_model(base, torch.device("cuda", local_rank))
            original_weight = (
                model.base_model.model.model.layers[0]
                .self_attn.q_proj.weight.detach()
                .clone()
            )
            if phase == "reference":
                trainer = [None]
                _stage(
                    "uninterrupted merge-aware DeepSpeed training",
                    lambda: trainer.__setitem__(0, _train(model, root / "run")),
                )
                assert calls[0] > 0
                checkpoint_path = root / "run" / "checkpoint-1"
                _stage(
                    "checkpoint metadata verification",
                    lambda: _metadata(checkpoint_path, original_weight),
                )
                _stage(
                    "frozen base checkpoint omission",
                    lambda: _assert_frozen_base_omitted(
                        model, trainer[0], checkpoint_path
                    ),
                )
                expected = {
                    "adapters": checkpoint.adapters(model),
                    "optimizer": checkpoint.optimizer_snapshot(
                        trainer[0].optimizer.state_dict()
                    ),
                    "scheduler": checkpoint.optimizer_snapshot(
                        trainer[0].lr_scheduler.state_dict()
                    ),
                    "logits": _logits(
                        trainer[0].model_wrapped, torch.device("cuda", local_rank)
                    ),
                }
                torch.save(expected, root / f"expected-{rank}.pt")
                export = root / "export"
                trainer[0].axolotl_cfg.output_dir = str(export)
                _stage(
                    "final application adapter export",
                    lambda: __import__(
                        "axolotl.train", fromlist=["save_trained_model"]
                    ).save_trained_model(
                        trainer[0].axolotl_cfg, trainer[0], trainer[0].model
                    ),
                )
                _stage(
                    "final metadata verification",
                    lambda: _metadata(export, original_weight),
                )
                _stage(
                    "format-preserving merged inference",
                    lambda: (
                        _merge_and_compare(
                            base,
                            export,
                            root / "merged",
                            expected["logits"],
                            torch.device("cuda", local_rank),
                        )
                        if rank == 0
                        else None
                    ),
                )
            else:
                trainer = [None]
                _stage(
                    "checkpoint-1 merge-aware DeepSpeed resume",
                    lambda: trainer.__setitem__(
                        0,
                        _train(
                            model, root / "resume", str(root / "run" / "checkpoint-1")
                        ),
                    ),
                )
                assert calls[0] > 0
                expected = torch.load(
                    root / f"expected-{rank}.pt", map_location="cpu", weights_only=True
                )
                for name, value in checkpoint.adapters(model).items():
                    torch.testing.assert_close(
                        value, expected["adapters"][name], rtol=0, atol=0
                    )
                checkpoint.assert_optimizer_snapshot(
                    checkpoint.optimizer_snapshot(trainer[0].optimizer.state_dict()),
                    expected["optimizer"],
                )
                checkpoint.assert_optimizer_snapshot(
                    checkpoint.optimizer_snapshot(trainer[0].lr_scheduler.state_dict()),
                    expected["scheduler"],
                )
                torch.testing.assert_close(
                    _logits(trainer[0].model_wrapped, torch.device("cuda", local_rank)),
                    expected["logits"],
                    rtol=0,
                    atol=0,
                )
        finally:
            bridge.native_nvfp4_merge_aware_linear = original_forward
        _stage("completion barrier", dist.barrier)
        print(f"NATIVE_NVFP4_DEEPSPEED_LIFECYCLE_PASS rank={rank}", flush=True)
    except BaseException:
        failure = traceback.format_exc()
        print(
            f"NATIVE_NVFP4_DEEPSPEED_LIFECYCLE_FAILURE rank={rank}",
            file=sys.stderr,
            flush=True,
        )
        print(failure, file=sys.stderr, flush=True)
    finally:
        try:
            if dist.is_initialized():
                failures = [None] * dist.get_world_size()
                dist.all_gather_object(failures, failure)
                for item in failures:
                    if item is not None:
                        print(item, file=sys.stderr, flush=True)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            faulthandler.cancel_dump_traceback_later()
    if failure is not None:
        raise RuntimeError(f"rank {rank} failed; original traceback was logged")


if __name__ == "__main__":
    main()
