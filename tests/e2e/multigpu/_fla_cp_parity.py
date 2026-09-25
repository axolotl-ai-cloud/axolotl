"""Opt-in four-rank native FLA forward/backward parity with a real Qwen mixer."""

import copy
import os
from types import MethodType

import torch
import torch.distributed as dist
from ringmaster.recurrent import (
    _rebind_globals,
    require_fla_cp,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


def assert_bf16_close(actual, expected, name, scale=None):
    # Bound BF16 accumulation error without dividing by near-zero individual entries.
    epsilon = 2 * torch.finfo(torch.bfloat16).eps
    delta = (actual.float() - expected.float()).abs()
    norm, peak = (
        scale
        if scale is not None
        else (expected.float().norm(), expected.float().abs().max())
    )
    relative = delta.norm() / norm.clamp_min(1e-8)
    maximum = delta.max() / peak.clamp_min(1e-8)
    assert relative <= epsilon, (name, "relative L2", relative.item())
    assert maximum <= epsilon, (name, "relative maximum", maximum.item())


class Hybrid(torch.nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer
        self.qkv = torch.nn.Linear(64, 192, bias=False).cuda().to(torch.bfloat16)
        self.output = torch.nn.Linear(64, 64, bias=False).cuda().to(torch.bfloat16)
        self.group = None

    def forward(self, x):
        from ringmaster.config import RotateMethod
        from ringmaster.ring.loop import ring_attention

        recurrent = self.mixer(x)
        if isinstance(recurrent, tuple):
            recurrent = recurrent[0]
        hidden = x + recurrent
        q, k, v = (
            value.reshape(1, -1, 2, 32).transpose(1, 2)
            for value in self.qkv(hidden).chunk(3, dim=-1)
        )
        from ringmaster.ring.loop import varlen_ring_attention
        from ringmaster.runtime import maybe_runtime

        runtime = maybe_runtime()
        if (
            self.group is not None
            and runtime is not None
            and runtime.varlen is not None
        ):
            attention = varlen_ring_attention(
                q, k, v, group=self.group, scaling=None, cu_seqlens=runtime.varlen[0]
            )
            return hidden + self.output(attention.reshape_as(hidden))
        attention = ring_attention(
            q,
            k,
            v,
            group=self.group,
            causal=True,
            scaling=None,
            dropout=0.0,
            provider="math",
            rotate_method=RotateMethod.ALLGATHER,
            attn_implementation="math",
        )
        return hidden + self.output(attention.reshape_as(hidden))


def main():
    import faulthandler

    if os.environ.get("RM_DEBUG_STACKS") == "1":
        faulthandler.dump_traceback_later(120, repeat=True)
    rank = int(os.environ["RANK"])
    shared = os.environ.get("AXOLOTL_FLA_CP_SHARED_GPU") == "1"
    torch.cuda.set_device(rank % torch.cuda.device_count() if shared else rank)
    dist.init_process_group("gloo" if shared else "nccl")
    if shared:
        original_gather = dist.all_gather_into_tensor

        def gather(output, input, **kwargs):
            # Gloo requires concatenated output; NCCL also accepts FLA's stacked layout.
            return original_gather(output.view(-1, *input.shape[1:]), input, **kwargs)

        dist.all_gather_into_tensor = gather
    try:
        torch.manual_seed(123)
        family = os.environ.get("RM_FLA_FAMILY", "gdn")
        if family == "kimi":
            from axolotl.model_support.kimi_linear.configuration_kimi import (
                KimiLinearConfig,
            )
            from axolotl.model_support.kimi_linear.modeling_kimi import (
                KimiDeltaAttention,
            )

            config = KimiLinearConfig(
                hidden_size=64,
                num_attention_heads=2,
                linear_attn_config={
                    "kda_layers": [0],
                    "full_attn_layers": [],
                    "head_dim": 32,
                    "num_heads": 2,
                    "short_conv_kernel_size": 4,
                },
            )
            model = KimiDeltaAttention(config, 0).cuda().to(torch.bfloat16)
            with torch.no_grad():
                model.A_log.fill_(-6)
                model.dt_bias.zero_()
            reference = copy.deepcopy(model)
        elif family == "kda":
            from axolotl.model_support.bailing_hybrid.configuration_bailing_moe_v3 import (
                BailingMoeV3Config,
            )
            from axolotl.model_support.bailing_hybrid.modeling_bailing_moe_v3 import (
                BailingMoeV3KimiDeltaAttention,
            )

            config = BailingMoeV3Config(
                hidden_size=64, num_attention_heads=2, head_dim=32, kda_safe_gate=False
            )
            model = BailingMoeV3KimiDeltaAttention(config, 0).cuda().to(torch.bfloat16)
            with torch.no_grad():
                model.A_log.fill_(-6)
            reference = copy.deepcopy(model)
        else:
            config = Qwen3_5TextConfig(
                hidden_size=64,
                num_hidden_layers=1,
                layer_types=["linear_attention"],
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
            )
            model = Qwen3_5GatedDeltaNet(config, 0).cuda().to(torch.bfloat16)
            with torch.no_grad():
                model.A_log.fill_(-6)
            reference = copy.deepcopy(model)
            _, chunk_gdn, causal_conv = require_fla_cp()

            def reference_conv(x, weight, bias=None, activation=None, **kwargs):
                output, _ = causal_conv(
                    x.transpose(1, 2).contiguous(), weight, bias, activation=activation
                )
                return output.transpose(1, 2)

            reference.forward = MethodType(
                _rebind_globals(
                    reference.forward.__func__,
                    {
                        "torch_chunk_gated_delta_rule": chunk_gdn,
                        "causal_conv1d_fn": reference_conv,
                    },
                ),
                reference,
            )
        if os.environ.get("RM_FLA_HYBRID") == "1":
            model = Hybrid(model)
            hybrid_reference = copy.deepcopy(model)
            hybrid_reference.mixer = reference
            reference = hybrid_reference
        x = torch.randn(1, 512, 64).cuda().to(torch.bfloat16)
        ref_x = x.clone().requires_grad_()
        print(f"rank={rank} reference forward", flush=True)
        packed = os.environ.get("RM_PACKED") == "1"
        lengths = [71, 57, 163, 9, 212]
        if packed:
            outputs = [reference(chunk) for chunk in ref_x.split(lengths, dim=1)]
            expected = torch.cat(
                [out[0] if isinstance(out, tuple) else out for out in outputs], dim=1
            )
        else:
            expected = reference(ref_x)
        if isinstance(expected, tuple):
            expected = expected[0]
        grad = torch.randn(expected.shape).cuda().to(torch.bfloat16)
        print(f"rank={rank} reference backward", flush=True)
        gradient_scales = {}
        if packed:
            handles = []
            for name, parameter in reference.named_parameters():
                if name.endswith("A_log"):

                    def capture(contribution, name=name):
                        # Bound reduction error before independent documents cancel.
                        norm, peak = gradient_scales.get(name, (0, 0))
                        value = contribution.float()
                        gradient_scales[name] = (
                            norm + value.norm(),
                            peak + value.abs().max(),
                        )

                    handles.append(parameter.register_hook(capture))
            for out, dy in zip(outputs, grad.split(lengths, dim=1), strict=True):
                if isinstance(out, tuple):
                    out = out[0]
                (out.float() * dy).sum().backward()
            for handle in handles:
                handle.remove()
        else:
            (expected.float() * grad).sum().backward()
        world = dist.get_world_size()
        from ringmaster import wire_recurrent_layers

        if packed:
            from types import SimpleNamespace

            from ringmaster.runtime import set_runtime
            from ringmaster.shard import varlen_meta

            positions = torch.cat(
                [torch.arange(n, device=x.device) for n in lengths]
            ).unsqueeze(0)
            set_runtime(
                SimpleNamespace(
                    varlen=varlen_meta(positions, 512), shard_load_balance="contiguous"
                )
            )
        restore = wire_recurrent_layers(model, group=dist.group.WORLD).restore
        if isinstance(model, Hybrid):
            model.group = dist.group.WORLD
        local_x = x.chunk(world, dim=1)[rank].contiguous().requires_grad_()
        print(f"rank={rank} CP forward", flush=True)
        actual = model(local_x)
        if isinstance(actual, tuple):
            actual = actual[0]
        target = expected.chunk(world, dim=1)[rank]
        if family == "gdn":
            torch.testing.assert_close(actual, target, atol=5e-3, rtol=3e-2)
        else:
            assert_bf16_close(actual, target, "output")
        print(f"rank={rank} CP backward", flush=True)
        (actual.float() * grad.chunk(world, dim=1)[rank]).sum().backward()
        target_grad = ref_x.grad.chunk(world, dim=1)[rank]
        if family == "gdn":
            torch.testing.assert_close(local_x.grad, target_grad, atol=1e-2, rtol=5e-2)
        else:
            assert_bf16_close(local_x.grad, target_grad, "input gradient")
        for (name, param), (_, ref) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            dist.all_reduce(param.grad)
            error = (
                param.grad.float() - ref.grad.float()
            ).norm() / ref.grad.float().norm().clamp_min(1e-8)
            if family == "gdn":
                assert error < 0.04, (name, error.item())
            else:
                assert_bf16_close(param.grad, ref.grad, name, gradient_scales.get(name))
        restore()
        print(
            f"PASS FLA CP={world} rank={rank} family={family} forward/backward",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
