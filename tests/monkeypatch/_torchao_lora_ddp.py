"""Two-rank native TorchAO NVFP4 + ordinary PEFT LoRA validation worker."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from axolotl.utils.dict import DictDefault


def rank():
    return dist.get_rank()


def barrier():
    dist.barrier()


def native(model):
    return [p for p in model.parameters() if type(p).__name__ == "NVFP4Tensor"]


def make_base(path):
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
    save_quantized_model(model, path)


def model_with_lora(base, device):
    from axolotl.monkeypatch.torchao_lora import enable_native_nvfp4_lora_training

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(
        device
    )
    model = get_peft_model(
        model, LoraConfig(r=2, target_modules=["q_proj"], lora_alpha=2)
    )
    assert native(model) and enable_native_nvfp4_lora_training(model)
    return model


def adapters(model):
    return {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def identical_native(model):
    for p in native(model):
        for name in ("qdata", "scale", "per_tensor_scale"):
            value = getattr(p, name, None)
            if value is None:
                continue
            component = (
                value.detach().contiguous().reshape(-1).view(torch.uint8).to("cuda")
            )
            gathered = [
                torch.empty_like(component) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered, component)
            assert all(torch.equal(gathered[0], item) for item in gathered[1:])


def train(model, output, resume=None):
    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.training_args import AxolotlTrainingArguments

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
        eval_strategy="no",
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = AxolotlTrainer(model=model, args=args, train_dataset=dataset)
    trainer.axolotl_cfg = DictDefault(
        tensor_parallel_size=1, context_parallel_size=1, expert_parallel_size=1
    )
    trainer.train(resume_from_checkpoint=resume)
    return trainer


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    root = Path(os.environ["TORCHAO_LORA_DDP_TMP"])
    base = root / "base"
    if rank() == 0:
        make_base(base)
    barrier()
    model = model_with_lora(base, device)
    baseline = {
        f"{index}.{name}": getattr(param, name).detach().clone()
        for index, param in enumerate(native(model))
        for name in ("qdata", "scale", "per_tensor_scale")
        if getattr(param, name, None) is not None
    }
    if rank() == 1:
        with torch.no_grad():
            native(model)[0].qdata.add_(1)
    before = adapters(model)
    train(model, root / "first")
    identical_native(model)
    assert all(
        torch.equal(
            getattr(native(model)[int(key.split(".")[0])], key.split(".")[1]), value
        )
        for key, value in baseline.items()
    )
    after = adapters(model)
    assert any(not torch.equal(before[n], after[n]) for n in before)
    checkpoint = root / "first" / "checkpoint-1"
    assert checkpoint.is_dir()
    if rank() == 0:
        model.save_pretrained(root / "export")
        assert not (root / "export" / "model.safetensors").exists()
        assert (root / "export" / "adapter_model.safetensors").exists()
    barrier()
    reference = adapters(model)
    resumed = model_with_lora(base, device)
    train(resumed, root / "resume", str(checkpoint))
    for name, value in adapters(resumed).items():
        assert torch.equal(value, reference[name])
    identical_native(resumed)
    assert all(
        torch.equal(
            getattr(native(resumed)[int(key.split(".")[0])], key.split(".")[1])
            .detach()
            .reshape(-1)
            .view(torch.uint8),
            value.reshape(-1).view(torch.uint8),
        )
        for key, value in baseline.items()
    )
    print(f"TORCHAO_LORA_DDP_OK rank={rank()}", flush=True)
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
