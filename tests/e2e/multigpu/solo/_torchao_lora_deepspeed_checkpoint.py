"""Two-rank native TorchAO NVFP4 + ordinary LoRA DeepSpeed checkpoint smoke."""

import os
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parents[3] / "monkeypatch"))

from _torchao_lora_ddp import barrier, make_base, model_with_lora
from datasets import Dataset


def _local(value):
    return value.to_local() if type(value).__name__ == "DTensor" else value


def adapters(model):
    return {
        name: _local(parameter).detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


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
            "zero_optimization": {"stage": int(os.environ["ZERO_STAGE"])},
        },
    )
    trainer = AxolotlTrainer(model=model, args=args, train_dataset=dataset)
    trainer.axolotl_cfg = DictDefault(deepspeed=args.deepspeed)
    trainer.train(resume_from_checkpoint=resume)
    return trainer


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    root = Path(os.environ["TORCHAO_LORA_DEEPSPEED_CHECKPOINT_TMP"])
    base = root / "base"
    if dist.get_rank() == 0 and not base.exists():
        make_base(base)
    barrier()
    phase = os.environ["TORCHAO_LORA_DEEPSPEED_PHASE"]
    model = model_with_lora(base, device)
    if phase == "reference":
        before = adapters(model)
        trainer = train(model, root / "run")
        after = adapters(model)
        assert any(not torch.equal(before[name], after[name]) for name in before)
        expected = {
            "adapters": after,
            "optimizer": optimizer_snapshot(trainer.optimizer.state_dict()),
            "scheduler": optimizer_snapshot(trainer.lr_scheduler.state_dict()),
        }
        expected_path = root / f"expected-{dist.get_rank()}.pt"
        torch.save(expected, expected_path)
        assert_optimizer_snapshot(
            torch.load(expected_path, map_location="cpu", weights_only=True), expected
        )
        checkpoint = root / "run" / "checkpoint-1"
        assert checkpoint.is_dir()
        engine_state = torch.load(
            checkpoint / "global_step1" / "mp_rank_00_model_states.pt",
            map_location="cpu",
            weights_only=True,
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
    print(f"TORCHAO_LORA_DEEPSPEED_CHECKPOINT_OK rank={dist.get_rank()}", flush=True)
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
