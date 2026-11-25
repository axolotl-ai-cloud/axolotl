"""torchrun worker: optmoe ``nvfp4_lora`` against the SonicMoE merge-aware path.

EP2 x dp_shard (world // 2) FSDP2 on a tiny Qwen3-MoE whose experts are packed NVFP4Tensors,
one PEFT ``target_parameters`` LoRA on both expert projections, two AdamW steps of two
micro-batches each. Checks the loss and the adapter gradients of both paths agree within the
W4A4 budget, that the adapters move, and that the requant cache is rebuilt exactly once per
step and layer.
"""

import json
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

TOKENS = 64
HIDDEN, INTER, EXPERTS, TOP_K, LAYERS = 256, 256, 8, 2, 2
RANK, ALPHA = 8, 16
STEPS, MICRO = 2, 2


def _pack(dense):
    """``[E, N, K]`` bf16 -> one NVFP4Tensor with a per-expert per-tensor scale."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    qdata, scale, pts = [], [], []
    for expert in dense:
        per_tensor_scale = per_tensor_amax_to_scale(expert.abs().max())
        native = NVFP4Tensor.to_nvfp4(
            expert.contiguous(), block_size=16, per_tensor_scale=per_tensor_scale
        )
        qdata.append(native.qdata)
        scale.append(native.scale)
        pts.append(per_tensor_scale)
    return NVFP4Tensor(
        torch.stack(qdata),
        torch.stack(scale),
        16,
        torch.bfloat16,
        per_tensor_scale=torch.stack(pts).reshape(-1, 1, 1),
    )


def _quantize_experts(model):
    """Pack the experts and leave them on the CPU, where a checkpoint loader leaves them."""
    from axolotl.integrations.expert_parallel.shard import _detect_experts_modules

    for _name, experts in _detect_experts_modules(model):
        for attr in ("gate_up_proj", "down_proj"):
            dense = getattr(experts, attr).detach().to(torch.bfloat16)
            packed = _pack(dense)
            packed = type(packed)(
                packed.qdata.cpu(),
                packed.scale.cpu(),
                packed.block_size,
                packed.orig_dtype,
                per_tensor_scale=packed.per_tensor_scale.cpu(),
            )
            setattr(experts, attr, torch.nn.Parameter(packed, requires_grad=False))


def _relative(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)).item()


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def main():
    from optmoe import DEFAULT_RESERVED_SMS, check_nccl_env, nccl_env
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from axolotl.integrations.expert_parallel import (
        experts_fn,
        nvfp4_lora,
        torch_dispatch,
    )
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
    from axolotl.integrations.expert_parallel.shard import (
        shard_expert_lora,
        shard_expert_weights,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        set_merge_aware_enabled,
    )
    from axolotl.monkeypatch import selective_checkpointing as sac
    from axolotl.monkeypatch.moe_quant import patch_peft_target_parameters_matching

    for key, value in nccl_env(reserved_sms=DEFAULT_RESERVED_SMS).items():
        os.environ.setdefault(key, value)
    check_nccl_env(reserved_sms=DEFAULT_RESERVED_SMS)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    world = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world // 2, 2), mesh_dim_names=("dp_shard", "ep"))
    dense_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("world",))
    ep_group = mesh["ep"].get_group()
    torch_dispatch.set_ep_group(ep_group)
    experts_fn.register_all()
    register_sonicmoe_experts()
    patch_peft_target_parameters_matching()
    experts_fn.set_local_implementation("sonicmoe")
    ExpertParallelPlugin._resolve_ep_group = staticmethod(lambda cfg: ep_group)
    # the model is synthetic; the config-level check reads a checkpoint's quantization config
    nvfp4_lora.base_is_nvfp4_modelopt = lambda cfg: True
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    def make_cfg(backend):
        return SimpleNamespace(
            expert_parallel_backend=backend,
            expert_parallel_optmoe_mode="auto",
            expert_parallel_optmoe_precision="auto",
            expert_parallel_fallback_on_unsupported=False,
            expert_parallel_save_dispatch=False,
            adapter="lora",
            use_sonicmoe=True,
            lora_dropout=0,
            lora_target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            nvfp4_merge_aware=False,
            base_model=None,
            sequence_len=TOKENS,
            micro_batch_size=1,
            eval_batch_size=None,
            context_parallel_size=None,
        )

    def run(backend):
        torch.manual_seed(123)
        config = Qwen3MoeConfig(
            vocab_size=256,
            hidden_size=HIDDEN,
            intermediate_size=512,
            moe_intermediate_size=INTER,
            num_hidden_layers=LAYERS,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            norm_topk_prob=True,
            use_cache=False,
            hidden_act="silu",
        )
        config._experts_implementation = "expert_parallel"
        config._attn_implementation = "sdpa"
        model = Qwen3MoeForCausalLM(config).cuda().train()
        _quantize_experts(model)
        sac.clear_registered_saves()
        cfg = make_cfg(backend)
        plugin = ExpertParallelPlugin()
        if backend == "optmoe":
            assert plugin._resolve_optmoe_model(cfg, model) == "optmoe"
            assert cfg.expert_parallel_optmoe_mode == "fused", cfg
            assert cfg.expert_parallel_optmoe_precision == "nvfp4_lora", cfg
        assert shard_expert_weights(model, ep_group) == LAYERS
        experts_fn.set_backend(backend)
        model = get_peft_model(
            model,
            LoraConfig(
                r=RANK,
                lora_alpha=ALPHA,
                target_modules=[],
                target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                lora_dropout=0.0,
                bias="none",
            ),
        )
        generator = torch.Generator(device="cuda").manual_seed(7)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                # PEFT sized them on the CPU experts; FSDP would move them anyway
                parameter.data = parameter.data.float().cuda()
                parameter.data.normal_(std=0.02, generator=generator)
        assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
        model._ep_lora_group = ep_group
        shard_expert_lora(model, 2)
        ExpertParallelPlugin._register_checkpoint_saves(cfg, model)
        state = None
        if backend == "optmoe":
            plugin._build_optmoe_runtime(cfg, model)
            state = nvfp4_lora.get_state()
            assert state is not None
        else:
            set_merge_aware_enabled(True)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "context_fn": sac.build_sac_context_fn(save=[]),
            }
        )
        kwargs = {"mp_policy": policy, "reshard_after_forward": True}
        ExpertParallelPlugin.fully_shard_experts(model, mesh["dp_shard"], kwargs)
        base = model.base_model.model
        for layer in base.model.layers:
            fully_shard(layer, mesh=dense_mesh, **kwargs)
        fully_shard(model, mesh=dense_mesh, **kwargs)
        if backend == "optmoe":
            # the frozen base stays a plain NVFP4Tensor outside FSDP
            for _n, module in base.named_modules():
                if type(module).__name__.endswith("Experts") and hasattr(
                    module, "gate_up_proj"
                ):
                    assert nvfp4_lora.is_nvfp4_tensor(module.gate_up_proj), type(
                        module.gate_up_proj
                    )
        # FSDP swapped the parameters for DTensors: collect the adapters after wrapping
        adapters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert adapters and all("lora_" in n for n in adapters), sorted(adapters)
        optimizer = torch.optim.AdamW(adapters.values(), lr=1e-3)
        starts = {n: p.to_local().detach().clone() for n, p in adapters.items()}
        losses, grads, builds = [], [], []
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for micro in range(MICRO):
                # dp_shard peers see different batches, EP peers the same one
                generator = torch.Generator(device="cuda").manual_seed(
                    1000
                    + 10 * step
                    + 100 * micro
                    + 7 * dist.get_rank(mesh["dp_shard"].get_group())
                )
                tokens = torch.randint(
                    0, 256, (1, TOKENS), device="cuda", generator=generator
                )
                loss = model(input_ids=tokens, labels=tokens).loss / MICRO
                assert torch.isfinite(loss), loss
                loss.backward()
                total += loss.item()
            losses.append(total)
            step_grads = {}
            for name, parameter in adapters.items():
                grad = parameter.grad
                assert grad is not None, name
                local = grad.to_local() if hasattr(grad, "to_local") else grad
                assert torch.isfinite(local).all(), name
                assert local.abs().sum() > 0, name
                step_grads[name] = local.detach().float().clone()
            grads.append(step_grads)
            if state is not None:
                builds.append(state.cache.builds)
                # every entry carries the current version: nothing stale served this step
                assert state.cache.entries and all(
                    entry.version == state.cache.version
                    for entry in state.cache.entries.values()
                ), (
                    state.cache.version,
                    [e.version for e in state.cache.entries.values()],
                )
            optimizer.step()
        if state is not None:
            if state.cache.mode == "all":
                # one rebuild per layer and step: micro-batches and recomputes reuse the merged weights
                assert builds == [LAYERS * (step + 1) for step in range(STEPS)], builds
            else:
                # just in time: the forward, the checkpoint recompute and the backward of every
                # layer each build their forms, per micro-batch; the recompute of the last
                # `capacity` layers of a micro-batch still finds its forward forms in the LRU
                forward_per_micro = LAYERS + max(LAYERS - state.cache.capacity, 0)
                per_step = (forward_per_micro + LAYERS) * MICRO
                assert builds == [per_step * (step + 1) for step in range(STEPS)], (
                    builds
                )
                assert (
                    state.cache.forward_builds == forward_per_micro * MICRO * STEPS
                ), state.stats()
                assert state.cache.backward_builds == LAYERS * MICRO * STEPS, (
                    state.stats()
                )
                assert state.cache.max_entries <= 2, state.stats()
            assert state.cache.version == STEPS
        moved = {
            n: _relative(p.to_local().detach(), starts[n]) for n, p in adapters.items()
        }
        assert all(v > 0 for v in moved.values()), moved
        if backend == "optmoe":
            plugin.post_train_unload(cfg)
            assert nvfp4_lora.get_state() is None
        else:
            set_merge_aware_enabled(False)
        del optimizer, model
        torch.cuda.synchronize()
        return losses, grads

    ref_losses, ref_grads = run("torch")
    losses, grads = run("optmoe")
    worst_rel, worst_cos, worst_loss = 0.0, 1.0, 0.0
    for step in range(STEPS):
        worst_loss = max(
            worst_loss, abs(losses[step] - ref_losses[step]) / abs(ref_losses[step])
        )
        for name, grad in grads[step].items():
            worst_rel = max(worst_rel, _relative(grad, ref_grads[step][name]))
            worst_cos = min(worst_cos, _cosine(grad, ref_grads[step][name]))
    stats = torch.tensor([worst_loss, worst_rel, -worst_cos], device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    worst_loss, worst_rel, worst_cos = (
        stats[0].item(),
        stats[1].item(),
        -stats[2].item(),
    )
    result = {
        "world_size": world,
        "losses": {"optmoe": losses, "sonicmoe_merge_aware": ref_losses},
        "max_loss_relative_diff": worst_loss,
        "max_adapter_grad_relative_l2": worst_rel,
        "min_adapter_grad_cosine": worst_cos,
    }
    if dist.get_rank() == 0:
        print(json.dumps(result))
    # W4A4 persistent kernels against W4 fake-quant with BF16 activations: the budget of the
    # package's own validation (adapter gradients ~0.2 relative to an FP32 reference)
    assert worst_loss < 0.1, result
    assert worst_rel < 0.5, result
    assert worst_cos > 0.9, result
    sac.clear_registered_saves()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
