"""Opt-in real-kernel FLA Mamba CP parity, including nonaligned packed boundaries."""

import os

import torch
import torch.distributed as dist

from axolotl.model_support.mamba.loading import MambaModelLoader

from tests.monkeypatch.test_fla_mamba import _config


def main():
    import ringmaster as rm
    from ringmaster.runtime import CPRuntime, set_runtime

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("gloo")
    try:
        for family in ("mamba", "mamba2"):
            for packed in (False, True):
                torch.manual_seed(31)
                model = MambaModelLoader(_config(family)).cuda().to(torch.bfloat16)
                if os.environ.get("RM_LORA") == "1":
                    from peft import LoraConfig, get_peft_model

                    model = get_peft_model(
                        model,
                        LoraConfig(
                            task_type="CAUSAL_LM",
                            r=4,
                            target_modules=["in_proj", "out_proj"],
                        ),
                    )
                    for name, parameter in model.named_parameters():
                        if "lora_B" in name:
                            torch.nn.init.normal_(parameter, std=0.05)
                ids = torch.randint(0, 64, (1, 96), device="cuda")
                positions = (
                    torch.cat(
                        [torch.arange(17), torch.arange(43), torch.arange(36)]
                    ).cuda()[None]
                    if packed
                    else torch.arange(96, device="cuda")[None]
                )
                set_runtime(None)
                embeddings = model.get_input_embeddings()(ids).detach().requires_grad_()
                reference = model(
                    inputs_embeds=embeddings, position_ids=positions
                ).logits
                reference.float().square().sum().div(reference.numel()).backward()
                grads = {
                    name: p.grad.clone()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
                input_gradient = embeddings.grad.clone()
                model.zero_grad(set_to_none=True)
                config = rm.RingmasterConfig(size=world)
                config.normalize(num_kv_heads=world, intra_node_size=world)
                runtime = CPRuntime(config=config, cp_group=dist.group.WORLD)
                if packed:
                    runtime.varlen = (
                        torch.tensor([0, 17, 60, 96], device="cuda", dtype=torch.int32),
                        43,
                    )
                set_runtime(runtime)
                wiring = rm.wire_recurrent_layers(model, group=dist.group.WORLD)
                start, end = rank * 96 // world, (rank + 1) * 96 // world
                local_embeddings = embeddings[:, start:end].detach().requires_grad_()
                actual = model(
                    inputs_embeds=local_embeddings, position_ids=positions[:, start:end]
                ).logits
                torch.testing.assert_close(
                    actual, reference[:, start:end], atol=0.02, rtol=0.03
                )
                actual.float().square().sum().div(reference.numel()).backward()
                torch.testing.assert_close(
                    local_embeddings.grad,
                    input_gradient[:, start:end],
                    atol=0.002,
                    rtol=0.03,
                )
                input_error = (
                    local_embeddings.grad.float() - input_gradient[:, start:end].float()
                ).norm() / input_gradient[:, start:end].float().norm().clamp_min(1e-6)
                assert input_error < 0.03
                errors = []
                for name, p in model.named_parameters():
                    if name not in grads:
                        continue
                    dist.all_reduce(p.grad)
                    error = (p.grad.float() - grads[name].float()).norm() / grads[
                        name
                    ].float().norm().clamp_min(1e-6)
                    errors.append(float(error))
                    torch.testing.assert_close(
                        p.grad,
                        grads[name],
                        atol=0.003,
                        rtol=0.04,
                        msg=f"{family} {name}",
                    )
                wiring.restore()
                set_runtime(None)
                print(
                    f"rank={rank} family={family} packed={packed} max_parameter_relative_l2={max(errors):.6f}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
