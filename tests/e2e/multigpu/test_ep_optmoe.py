"""EP2 x FSDP2 gradient and optimizer parity of the fused optmoe backend against
torch dispatch with grouped_mm run the same way, eager and under torch.compile."""

import json
import os
import subprocess
import sys
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.gpu

TOKENS = 17


def training_checks():
    from optmoe import DEFAULT_RESERVED_SMS, check_nccl_env, nccl_env
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from axolotl.integrations.expert_parallel import experts_fn, torch_dispatch
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
    from axolotl.integrations.expert_parallel.shard import shard_expert_weights
    from axolotl.monkeypatch import selective_checkpointing as sac

    # must precede the first NCCL communicator
    for key, value in nccl_env(reserved_sms=DEFAULT_RESERVED_SMS).items():
        os.environ.setdefault(key, value)
    check_nccl_env(reserved_sms=DEFAULT_RESERVED_SMS)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    world = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world // 2, 2), mesh_dim_names=("dp_shard", "ep"))
    dense_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("world",))
    ep_group = mesh["ep"].get_group()
    torch_dispatch.set_ep_group(ep_group)
    experts_fn.register_all()
    experts_fn.set_local_implementation("grouped_mm")
    ExpertParallelPlugin._resolve_ep_group = staticmethod(lambda cfg: ep_group)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    cfg = SimpleNamespace(
        expert_parallel_backend="optmoe",
        expert_parallel_optmoe_precision="bf16",
        expert_parallel_fallback_on_unsupported=False,
        expert_parallel_save_dispatch=False,
        sequence_len=TOKENS,
        micro_batch_size=1,
        eval_batch_size=None,
        context_parallel_size=None,
    )
    plugin = ExpertParallelPlugin()

    def run(backend, checkpointing, compiled, starting_weights=None):
        torch.manual_seed(123)
        config = Qwen3MoeConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=8,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            use_cache=False,
            hidden_act="silu",
        )
        config._experts_implementation = "expert_parallel"
        config._attn_implementation = "sdpa"
        model = Qwen3MoeForCausalLM(config).cuda().train()
        sac.clear_registered_saves()
        if backend == "optmoe":
            assert plugin._resolve_optmoe_model(cfg, model) == "optmoe"
        assert shard_expert_weights(model, ep_group) == 2
        experts_fn.set_backend(backend)
        ExpertParallelPlugin._register_checkpoint_saves(cfg, model)
        if backend == "optmoe":
            plugin._build_optmoe_runtime(cfg)
        if checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                    "context_fn": sac.build_sac_context_fn(save=[]),
                }
            )
        kwargs = {"mp_policy": policy, "reshard_after_forward": True}
        ExpertParallelPlugin.fully_shard_experts(model, mesh["dp_shard"], kwargs)
        for layer in model.model.layers:
            fully_shard(layer, mesh=dense_mesh, **kwargs)
        fully_shard(model, mesh=dense_mesh, **kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        forward = torch.compile(model) if compiled else model
        history, starts = [], []
        for step in range(2):
            if starting_weights is not None:
                # identical pre-step weights keep near-tie top-k routing from diverging
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.to_local().copy_(starting_weights[step][name])
            starts.append(
                {n: p.to_local().detach().clone() for n, p in model.named_parameters()}
            )
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                generator = torch.Generator(device="cuda").manual_seed(
                    1000 + dist.get_rank() + 10 * step + 100 * micro
                )
                tokens = torch.randint(
                    0, 256, (1, TOKENS), device="cuda", generator=generator
                )
                loss = forward(input_ids=tokens, labels=tokens).loss / 2
                assert torch.isfinite(loss)
                loss.backward()
            grads = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                local = parameter.grad.to_local().detach()
                assert torch.isfinite(local).all(), name
                grads[name] = local.clone()
            optimizer.step()
            history.append(grads)
        weights = {
            n: p.to_local().detach().clone() for n, p in model.named_parameters()
        }
        plugin.post_train_unload(cfg)
        del optimizer, model, forward
        torch.cuda.synchronize()
        return history, weights, starts

    # Inductor's bf16 kernels round the router logits differently from eager, and a token
    # one bf16 ulp from the top-k boundary then routes to another expert, so compiled runs
    # are held to a compiled torch reference
    references = {}
    result = {}
    for checkpointing, compiled in ((False, False), (True, False), (False, True)):
        if compiled not in references:
            torch._dynamo.reset()
            references[compiled] = run("torch", False, compiled)
        reference, ref_weights, starting_weights = references[compiled]
        torch._dynamo.reset()
        history, weights, _ = run("optmoe", checkpointing, compiled, starting_weights)
        worst = 0.0
        for grads, expected in zip(history, reference, strict=True):
            for name, grad in grads.items():
                rel = (grad.float() - expected[name].float()).norm() / expected[
                    name
                ].float().norm().clamp_min(1e-8)
                worst = max(worst, rel.item())
        for name, weight in weights.items():
            torch.testing.assert_close(weight, ref_weights[name], atol=2e-4, rtol=0.02)
        error = torch.tensor(worst, device="cuda")
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert error.item() < 0.03, (
            dist.get_rank(),
            checkpointing,
            compiled,
            error.item(),
        )
        result[f"checkpointing={checkpointing},compile={compiled}"] = {
            "max_gradient_relative_l2": error.item(),
            "world_size": world,
            "expert_parallel_size": 2,
        }
    if dist.get_rank() == 0:
        print(json.dumps(result))
    sac.clear_registered_saves()
    dist.destroy_process_group()


@pytest.mark.slow
def test_optmoe_fsdp2_parity():
    world_size = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA GPUs")
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("optmoe requires SM90 or SM100")
    if find_spec("optmoe") is None or find_spec("optmoe_deepgemm") is None:
        pytest.skip("requires the optmoe package and its kernel extension")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            __file__,
            "--worker",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=1500,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__" and "--worker" in sys.argv:
    training_checks()
