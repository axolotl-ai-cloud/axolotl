"""Two-rank native TorchAO NVFP4 + ordinary LoRA DeepSpeed smoke."""

import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from _torchao_lora_ddp import (
    adapters,
    barrier,
    identical_native,
    make_base,
    model_with_lora,
    native,
)


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend="nccl")
    root = Path(os.environ["TORCHAO_LORA_DEEPSPEED_TMP"])
    base = root / "base"
    if dist.get_rank() == 0:
        make_base(base)
    barrier()
    model = model_with_lora(base, device)
    baseline = {
        f"{index}.{name}": getattr(parameter, name).detach().clone()
        for index, parameter in enumerate(native(model))
        for name in ("qdata", "scale", "per_tensor_scale")
        if getattr(parameter, name, None) is not None
    }
    if dist.get_rank() == 1:
        with torch.no_grad():
            native(model)[0].qdata.add_(1)
    from axolotl.monkeypatch.torchao_deepspeed import prepare_native_nvfp4_deepspeed

    assert prepare_native_nvfp4_deepspeed(model, device, int(os.environ["ZERO_STAGE"]))
    before = adapters(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": int(os.environ["ZERO_STAGE"])},
        },
    )
    inputs = torch.arange(1, 9, device=device).reshape(2, 4).remainder(31)
    loss = engine(input_ids=inputs, labels=inputs).loss
    engine.backward(loss)
    engine.step()
    identical_native(engine.module)
    assert all(
        torch.equal(
            getattr(native(engine.module)[int(key.split(".")[0])], key.split(".")[1]),
            value,
        )
        for key, value in baseline.items()
    )
    after = adapters(engine.module)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    print(f"TORCHAO_LORA_DEEPSPEED_OK rank={dist.get_rank()}", flush=True)
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
