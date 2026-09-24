"""Two-rank native TorchAO NVFP4 + ordinary LoRA FSDP2 checkpoint smoke."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from _torchao_lora_ddp import barrier, make_base, model_with_lora, native
from datasets import Dataset


def _local(value):
    return value.to_local() if type(value).__name__ == "DTensor" else value


def adapters(model):
    return {
        name: _local(parameter).detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


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
        fsdp="full_shard",
        fsdp_config={
            "fsdp_version": 2,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "state_dict_type": "FULL_STATE_DICT",
            "reshard_after_forward": True,
            "cpu_ram_efficient_loading": False,
        },
    )
    trainer = AxolotlTrainer(model=model, args=args, train_dataset=dataset)
    trainer.axolotl_cfg = DictDefault(fsdp_config=args.fsdp_config)
    trainer.train(resume_from_checkpoint=resume)
    return trainer


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2

    patch_accelerate_fsdp2()
    root = Path(os.environ["TORCHAO_LORA_FSDP2_TMP"])
    base = root / "base"
    if dist.get_rank() == 0:
        make_base(base)
    barrier()
    model = model_with_lora(base, device)
    before = adapters(model)
    train(model, root / "first")
    after = adapters(model)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    assert all(not parameter.requires_grad for parameter in native(model))
    checkpoint = root / "first" / "checkpoint-1"
    assert checkpoint.is_dir()
    fresh = model_with_lora(base, device)
    resumed = train(fresh, root / "resume", str(checkpoint))
    for name, value in adapters(fresh).items():
        torch.testing.assert_close(value, after[name], rtol=0, atol=0)
    assert resumed.state.global_step == 2
    print(f"TORCHAO_LORA_FSDP2_OK rank={dist.get_rank()}", flush=True)
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
