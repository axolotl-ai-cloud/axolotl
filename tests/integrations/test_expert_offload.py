"""Tests for the expert-offload integration (single-GPU CPU offload of frozen 4-bit MoE experts).

The scheduling/recompute correctness — the load-bearing, tricky part — is validated on CPU with a
fake MoE whose experts are plain ``F.linear`` layers carrying a ``Params4bit``-named weight: the
mechanism only moves ``weight.data``, so this exercises the full stage/evict + gradient-checkpoint
recompute path without a GPU or bitsandbytes. A CUDA-gated test then confirms it end-to-end against
real ``bitsandbytes.nn.Linear4bit`` experts.
"""

import copy

import pytest
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from pydantic import ValidationError
from torch import nn
from torch.utils.checkpoint import checkpoint

from axolotl.integrations.expert_offload import ExpertOffloadArgs
from axolotl.integrations.expert_offload.offload import (
    _PLACEHOLDERS,
    _base_layer,
    _BlockOffload,
    _is_linear4bit,
    find_moe_expert_blocks,
    find_parametrized_expert_stacks,
    install_expert_offload,
)


# --------------------------------------------------------------------------- #
# CPU fakes: the mechanism keys on the weight *type name* and only moves .data #
# --------------------------------------------------------------------------- #
class Params4bit(nn.Parameter):
    """Stand-in for bitsandbytes' packed weight — detection matches on the class name."""


class FakeLinear4bit(nn.Module):
    """Frozen "4-bit" expert: a plain F.linear whose weight is a (frozen) Params4bit."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = Params4bit(torch.randn(d_out, d_in) * 0.1, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


class FakeLoraWrap(nn.Module):
    """Mimics a PEFT ``lora.Linear4bit``: delegates ``weight`` to a frozen base_layer, adds a
    trainable A/B that must stay resident."""

    def __init__(self, base: FakeLinear4bit, r: int = 4):
        super().__init__()
        self.base_layer = base
        d_out, d_in = base.weight.shape
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


class FakeMoEBlock(nn.Module):
    def __init__(self, d: int, n_experts: int, lora: bool = False):
        super().__init__()
        experts = [FakeLinear4bit(d, d) for _ in range(n_experts)]
        if lora:
            experts = [FakeLoraWrap(e) for e in experts]
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Linear(d, n_experts)  # trainable router, stays resident

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=-1)
        out = torch.zeros_like(x)
        for i, e in enumerate(self.experts):
            out = out + w[..., i : i + 1] * e(x)
        return out


class FakeMoEModel(nn.Module):
    def __init__(
        self, d: int = 16, n_experts: int = 4, n_layers: int = 3, lora: bool = False
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeMoEBlock(d, n_experts, lora) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d, d)

    def forward(self, x, use_ckpt: bool = False):
        for blk in self.layers:
            x = x + (checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x))
        return self.head(x)


class Bnb4bitParametrization(nn.Module):
    """Stand-in for bitsandbytes' dequantizing parametrization — detection matches on the class
    name. Identity "dequant" over a float packed tensor keeps the math checkable on CPU; the real
    one also runs under ``no_grad``, so autograd never sees the packed tensor in either case."""

    @torch.no_grad()
    def forward(self, packed):
        return packed


class FakeGroupedExperts(nn.Module):
    """OLMoE-style fused experts: one frozen 3D stack per projection, parametrized "4-bit"."""

    def __init__(self, d: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(n_experts, d, d) * 0.1, requires_grad=False
        )
        self.down_proj = nn.Parameter(
            torch.randn(n_experts, d, d) * 0.1, requires_grad=False
        )
        for pname in ("gate_up_proj", "down_proj"):
            parametrize.register_parametrization(
                self, pname, Bnb4bitParametrization(), unsafe=True
            )

    def forward(self, x, w):
        out = torch.zeros_like(x)
        for i in range(self.n_experts):
            h = torch.tanh(F.linear(x, self.gate_up_proj[i]))
            out = out + w[..., i : i + 1] * F.linear(h, self.down_proj[i])
        return out


class FakeGroupedMoEBlock(nn.Module):
    def __init__(self, d: int, n_experts: int):
        super().__init__()
        self.router = nn.Linear(d, n_experts)  # trainable, stays resident
        self.experts = FakeGroupedExperts(d, n_experts)  # a module, NOT a ModuleList

    def forward(self, x):
        w = torch.softmax(self.router(x), dim=-1)
        return self.experts(x, w)


class FakeGroupedMoEModel(nn.Module):
    def __init__(self, d: int = 16, n_experts: int = 4, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeGroupedMoEBlock(d, n_experts) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d, d)

    def forward(self, x, use_ckpt: bool = False):
        for blk in self.layers:
            x = x + (checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x))
        return self.head(x)


@pytest.fixture(autouse=True)
def _reset_offload_globals():
    """The single-resident slot and placeholder cache are class/module globals — reset between
    tests so residency state does not leak across them."""
    _BlockOffload._resident = None
    _PLACEHOLDERS.clear()
    yield
    _BlockOffload._resident = None
    _PLACEHOLDERS.clear()


# --------------------------------------------------------------------------- #
# Args + config validation                                                    #
# --------------------------------------------------------------------------- #
class MergedConfig(ExpertOffloadArgs):
    """Mimics ``integrations/config.py::merge_input_args``, which folds plugin args into the full
    config by inheritance — so the schema validator sees base-config fields via ``self``."""

    load_in_4bit: bool = False
    adapter: str | None = None
    gradient_checkpointing: bool | str = False
    gradient_checkpointing_kwargs: dict | None = None
    fsdp: list | None = None
    fsdp_config: dict | None = None
    deepspeed: str | dict | None = None
    expert_parallel_size: int = 1


def _cfg(**overrides):
    base = dict(
        expert_offload=True,
        load_in_4bit=True,
        adapter="qlora",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    base.update(overrides)
    return MergedConfig(**base)


class TestArgs:
    def test_defaults(self):
        a = ExpertOffloadArgs()
        assert a.expert_offload is False
        assert a.expert_offload_pin_memory is True

    def test_enabled(self):
        assert _cfg().expert_offload is True


class TestValidation:
    def test_valid_config_passes(self):
        _cfg()  # must not raise

    def test_disabled_skips_validation(self):
        # The validator must be a no-op when the feature is off, even with an otherwise bad config.
        MergedConfig(expert_offload=False, load_in_4bit=False)

    @pytest.mark.parametrize(
        "override, needle",
        [
            (dict(load_in_4bit=False), "load_in_4bit"),
            (dict(adapter="lora", load_in_4bit=False), "load_in_4bit"),
            (dict(adapter="full"), "qlora"),
            (dict(gradient_checkpointing=False), "gradient_checkpointing"),
            (
                dict(gradient_checkpointing_kwargs={"use_reentrant": True}),
                "use_reentrant",
            ),
            # omitted kwargs must fail too: normalize_config later defaults them to
            # {"use_reentrant": True}, after this validator has already run.
            (dict(gradient_checkpointing_kwargs=None), "use_reentrant"),
            (dict(fsdp_config={"x": 1}), "FSDP"),
            (dict(fsdp=["full_shard"]), "FSDP"),
            (dict(deepspeed="ds.json"), "DeepSpeed"),
            (dict(expert_parallel_size=2), "expert_parallel"),
        ],
    )
    def test_invalid_configs_raise(self, override, needle):
        with pytest.raises(ValidationError, match=needle):
            _cfg(**override)

    def test_lora_with_4bit_is_accepted(self):
        _cfg(adapter="lora")  # lora + load_in_4bit is fine


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
# --------------------------------------------------------------------------- #
class TestDiscovery:
    def test_finds_blocks_and_experts(self):
        model = FakeMoEModel(n_experts=4, n_layers=3)
        blocks = find_moe_expert_blocks(model)
        assert len(blocks) == 3
        for _name, block, bases in blocks:
            assert isinstance(block, FakeMoEBlock)
            assert len(bases) == 4
            assert all(_is_linear4bit(b) for b in bases)

    def test_unwraps_peft_base_layer(self):
        model = FakeMoEModel(n_experts=4, n_layers=2, lora=True)
        blocks = find_moe_expert_blocks(model)
        assert len(blocks) == 2
        for _name, _block, bases in blocks:
            assert len(bases) == 4  # deduped to the 4 base layers, not the wraps
            assert all(isinstance(b, FakeLinear4bit) for b in bases)
            assert all(_base_layer(b) is b for b in bases)  # already the base

    def test_ignores_singleton_and_non_4bit(self):
        model = nn.Module()
        model.solo = nn.Module()
        model.solo.experts = nn.ModuleList([FakeLinear4bit(8, 8)])  # len 1 -> skipped
        model.dense = nn.Module()
        model.dense.experts = nn.ModuleList(
            [nn.Linear(8, 8), nn.Linear(8, 8)]
        )  # not 4bit
        assert find_moe_expert_blocks(model) == []

    def test_install_raises_when_no_offloadable_blocks(self):
        # expert_offload was explicitly enabled; a layout with nothing to offload is a
        # misconfiguration and must fail loudly, not proceed without the VRAM reduction.
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        with pytest.raises(RuntimeError, match="no 4-bit MoE expert weights"):
            install_expert_offload(model, device="cpu", pin=False)


class TestDDPIgnoreList:
    # DDP's initial module-state sync must never touch the evicted 0-element placeholders, so
    # install registers every offloaded weight on ``_ddp_params_and_buffers_to_ignore`` (read by
    # DistributedDataParallel at construction, which happens after post_model_load).

    def test_install_registers_offloaded_weights(self):
        model = FakeMoEModel(n_experts=4, n_layers=3)
        handles = install_expert_offload(model, device="cpu", pin=False)
        ignore = model._ddp_params_and_buffers_to_ignore
        offloaded = {id(p) for handle in handles for p in handle.params}
        named = dict(model.named_parameters(remove_duplicate=False))
        # exactly the offloaded expert weights, each resolvable by name on the model DDP wraps
        assert len(ignore) == len(offloaded) == 3 * 4
        assert all(name in named and id(named[name]) in offloaded for name in ignore)

    def test_ignore_list_appends_without_duplicating(self):
        model = FakeMoEModel(n_experts=2, n_layers=1)
        model._ddp_params_and_buffers_to_ignore = ["pre.existing"]
        install_expert_offload(model, device="cpu", pin=False)
        ignore = model._ddp_params_and_buffers_to_ignore
        assert ignore[0] == "pre.existing"
        assert len(ignore) == 1 + 2
        assert len(ignore) == len(set(ignore))

    def test_trainable_lora_params_not_ignored(self):
        model = FakeMoEModel(n_experts=2, n_layers=2, lora=True)
        install_expert_offload(model, device="cpu", pin=False)
        ignore = set(model._ddp_params_and_buffers_to_ignore)
        for name, param in model.named_parameters():
            if param.requires_grad:  # LoRA A/B must stay in DDP's gradient buckets
                assert name not in ignore


# --------------------------------------------------------------------------- #
# The core: gradient-checkpoint recompute correctness + single-resident       #
# --------------------------------------------------------------------------- #
class TestRecompute:
    def test_offloaded_grads_match_reference(self):
        """With gradient checkpointing, offloaded training must produce the *same* gradients as the
        non-offloaded reference — proof that the backward recompute reads correctly-staged expert
        weights (not the 0-element eviction placeholders)."""
        torch.manual_seed(0)
        model = FakeMoEModel(d=16, n_experts=4, n_layers=3, lora=True)
        reference = copy.deepcopy(model)
        x = torch.randn(2, 5, 16)

        reference.zero_grad()
        out_ref = reference(x, use_ckpt=True)
        out_ref.sum().backward()
        ref_grads = {
            n: p.grad.clone()
            for n, p in reference.named_parameters()
            if p.grad is not None
        }

        install_expert_offload(model, device="cpu", pin=False)
        model.zero_grad()
        out_off = model(x, use_ckpt=True)
        out_off.sum().backward()
        off_grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        assert torch.allclose(out_ref, out_off, atol=1e-6)
        assert set(ref_grads) == set(off_grads) and len(ref_grads) > 0
        # No expert base weight received a grad (they are frozen and offloaded).
        assert not any("base_layer.weight" in n for n in off_grads)
        for name, g in ref_grads.items():
            assert torch.allclose(g, off_grads[name], atol=1e-6), (
                f"grad mismatch for {name}"
            )

    def test_at_most_one_block_resident(self):
        torch.manual_seed(0)
        model = FakeMoEModel(d=16, n_experts=4, n_layers=4)
        handles = install_expert_offload(model, device="cpu", pin=False)
        max_resident = 0

        def probe(_module, _args):
            nonlocal max_resident
            max_resident = max(max_resident, sum(h.staged for h in handles))

        for _name, block, _bases in find_moe_expert_blocks(model):
            block.register_forward_pre_hook(
                probe
            )  # runs right after the stage pre-hook

        model(torch.randn(2, 5, 16), use_ckpt=True).sum().backward()
        assert max_resident == 1  # exactly one block staged during any block's forward
        assert sum(h.staged for h in handles) <= 1  # settled state after backward

    def test_evicted_weights_are_offloaded_between_forwards(self):
        model = FakeMoEModel(d=16, n_experts=4, n_layers=3)
        handles = install_expert_offload(model, device="cpu", pin=False)
        # Right after install, every block is evicted: each expert weight is a 0-element placeholder.
        for h in handles:
            for p in h.params:
                assert p.data.numel() == 0
        assert handles[0].bytes > 0  # but the homes hold the real data

    def test_state_dict_substitutes_homes_for_placeholders(self):
        """A full state_dict() while evicted must still contain the real expert weights (via the
        home-substitution hook), not the 0-element placeholders."""
        model = FakeMoEModel(d=16, n_experts=4, n_layers=2)
        install_expert_offload(model, device="cpu", pin=False)
        sd = model.state_dict()
        expert_keys = [k for k in sd if ".experts." in k and k.endswith(".weight")]
        assert len(expert_keys) == 8
        for k in expert_keys:
            assert sd[k].numel() > 0  # home substituted, not the placeholder


# --------------------------------------------------------------------------- #
# Grouped 3D stacks quantized via quantize_moe_experts (parametrized layout)  #
# --------------------------------------------------------------------------- #
class TestParametrizedStacks:
    def test_discovery_finds_parametrized_stacks(self):
        model = FakeGroupedMoEModel(n_experts=4, n_layers=3)
        blocks = find_parametrized_expert_stacks(model)
        assert len(blocks) == 3
        for _name, module, slots in blocks:
            assert isinstance(module, FakeGroupedExperts)  # the hook site
            assert len(slots) == 2  # gate_up_proj + down_proj stacks
            assert all(not s.param.requires_grad for s in slots)

    def test_moduleList_path_does_not_match_grouped_layout(self):
        model = FakeGroupedMoEModel(n_experts=4, n_layers=2)
        assert find_moe_expert_blocks(model) == []  # no double-discovery

    def test_install_evicts_stacks(self):
        model = FakeGroupedMoEModel(n_experts=4, n_layers=3)
        handles = install_expert_offload(model, device="cpu", pin=False)
        assert len(handles) == 3
        for h in handles:
            assert len(h.params) == 2
            for p in h.params:
                assert p.data.numel() == 0  # evicted to placeholder
            assert h.bytes > 0  # homes hold the real data

    def test_offloaded_grads_match_reference(self):
        """Same contract as the Linear4bit path: offloaded + checkpointed training must produce
        identical outputs and trainable gradients — the recompute pre-hook re-stages the packed
        stacks before the parametrization dequantizes them."""
        torch.manual_seed(0)
        model = FakeGroupedMoEModel(d=16, n_experts=4, n_layers=3)
        reference = copy.deepcopy(model)
        x = torch.randn(2, 5, 16)

        reference.zero_grad()
        out_ref = reference(x, use_ckpt=True)
        out_ref.sum().backward()
        ref_grads = {
            n: p.grad.clone()
            for n, p in reference.named_parameters()
            if p.grad is not None
        }

        install_expert_offload(model, device="cpu", pin=False)
        model.zero_grad()
        out_off = model(x, use_ckpt=True)
        out_off.sum().backward()
        off_grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        assert torch.allclose(out_ref, out_off, atol=1e-6)
        assert set(ref_grads) == set(off_grads) and len(ref_grads) > 0
        assert not any("parametrizations" in n for n in off_grads)  # stacks stay frozen
        for name, g in ref_grads.items():
            assert torch.allclose(g, off_grads[name], atol=1e-6), (
                f"grad mismatch for {name}"
            )

    def test_state_dict_substitutes_homes_for_placeholders(self):
        model = FakeGroupedMoEModel(d=16, n_experts=4, n_layers=2)
        install_expert_offload(model, device="cpu", pin=False)
        sd = model.state_dict()
        stack_keys = [k for k in sd if ".original" in k]
        assert len(stack_keys) == 4  # 2 layers x 2 stacks
        for k in stack_keys:
            assert sd[k].numel() > 0  # home substituted, not the placeholder

    def test_ddp_ignore_covers_parametrized_stacks(self):
        model = FakeGroupedMoEModel(n_experts=4, n_layers=3)
        install_expert_offload(model, device="cpu", pin=False)
        ignore = model._ddp_params_and_buffers_to_ignore
        named = dict(model.named_parameters(remove_duplicate=False))
        assert len(ignore) == 3 * 2
        assert all(
            ".parametrizations." in n and n.endswith(".original") for n in ignore
        )
        assert all(n in named and not named[n].requires_grad for n in ignore)
        for name, param in model.named_parameters():
            if param.requires_grad:  # router/head must stay in DDP's buckets
                assert name not in ignore


# --------------------------------------------------------------------------- #
# CUDA + bitsandbytes: the real Linear4bit path                               #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + bitsandbytes"
)
class TestCudaLinear4bit:
    @staticmethod
    def _build(d=512, n_experts=16, n_layers=4):
        bnb = pytest.importorskip("bitsandbytes")

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList(
                    [
                        bnb.nn.Linear4bit(
                            d,
                            d,
                            bias=False,
                            compute_dtype=torch.float16,
                            quant_type="nf4",
                        )
                        for _ in range(n_experts)
                    ]
                )
                self.gate = nn.Linear(d, n_experts)

            def forward(self, x):
                w = torch.softmax(self.gate(x), dim=-1)
                out = torch.zeros_like(x)
                for i, e in enumerate(self.experts):
                    out = out + w[..., i : i + 1] * e(x)
                return out

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

            def forward(self, x, use_ckpt=True):
                for blk in self.layers:
                    x = x + (
                        checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x)
                    )
                return x

        return Model().cuda().half(), d

    def test_install_frees_experts_from_gpu(self):
        """Installing the offload must move the packed 4-bit experts off the GPU: live GPU memory
        drops by (approximately) the experts' bytes, and every expert weight becomes a 0-element
        placeholder while evicted."""
        torch.manual_seed(0)
        model, _d = self._build()
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()

        handles = install_expert_offload(model, device="cuda", pin=True)
        torch.cuda.synchronize()
        alloc_after = torch.cuda.memory_allocated()

        expert_bytes = sum(h.bytes for h in handles)
        assert expert_bytes > 0
        # Everything is evicted between forwards -> no expert data resident on the GPU.
        resident = sum(
            p.data.numel() for h in handles for p in h.params if p.data.is_cuda
        )
        assert resident == 0
        # The freed GPU memory accounts for (most of) the experts' packed bytes.
        assert alloc_before - alloc_after >= 0.8 * expert_bytes

    def test_recompute_grads_match_reference(self):
        """With real bitsandbytes Linear4bit + gradient checkpointing, offloaded training produces
        the same trainable gradients as the non-offloaded reference — the recompute reads staged
        4-bit weights through the real matmul_4bit backward, not the eviction placeholders."""
        torch.manual_seed(0)
        model, d = self._build()
        reference = copy.deepcopy(model)
        x = torch.randn(2, 8, d, device="cuda", dtype=torch.float16)

        reference.zero_grad()
        reference(x).sum().backward()
        ref_grads = {
            n: p.grad.clone()
            for n, p in reference.named_parameters()
            if p.grad is not None
        }

        install_expert_offload(model, device="cuda", pin=True)
        model.zero_grad()
        model(x).sum().backward()
        off_grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        assert set(ref_grads) == set(off_grads) and len(ref_grads) > 0
        for name, g in ref_grads.items():
            assert torch.allclose(g, off_grads[name], atol=1e-3, rtol=1e-2), name


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + bitsandbytes"
)
class TestCudaParametrized:
    """The real ``quantize_moe_experts`` layout: grouped 3D stacks quantized in place through
    ``bitsandbytes.nn.parametrize.replace_parameter_4bit`` (what axolotl's on-load patch calls)."""

    @staticmethod
    def _build(d=256, n_experts=8, n_layers=3):
        pytest.importorskip("bitsandbytes")
        from bitsandbytes.nn.parametrize import replace_parameter_4bit

        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_experts = n_experts
                self.gate_up_proj = nn.Parameter(
                    torch.randn(n_experts, d, d, device="cuda", dtype=torch.float16)
                    * 0.02,
                    requires_grad=False,
                )
                self.down_proj = nn.Parameter(
                    torch.randn(n_experts, d, d, device="cuda", dtype=torch.float16)
                    * 0.02,
                    requires_grad=False,
                )

            def forward(self, x, w):
                out = torch.zeros_like(x)
                for i in range(self.n_experts):
                    h = torch.tanh(F.linear(x, self.gate_up_proj[i]))
                    out = out + w[..., i : i + 1] * F.linear(h, self.down_proj[i])
                return out

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(d, n_experts)
                self.experts = Experts()

            def forward(self, x):
                w = torch.softmax(self.router(x), dim=-1)
                return self.experts(x, w)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

            def forward(self, x, use_ckpt=True):
                for blk in self.layers:
                    x = x + (
                        checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x)
                    )
                return x

        model = Model().cuda().half()
        for blk in model.layers:
            for pname in ("gate_up_proj", "down_proj"):
                replace_parameter_4bit(blk.experts, pname, quant_type="nf4")
        return model, d

    def test_install_frees_stacks_from_gpu(self):
        torch.manual_seed(0)
        model, _d = self._build()
        blocks = find_parametrized_expert_stacks(model)
        assert len(blocks) == 3 and all(len(s) == 2 for _n, _m, s in blocks)
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()

        handles = install_expert_offload(model, device="cuda", pin=True)
        torch.cuda.synchronize()
        alloc_after = torch.cuda.memory_allocated()

        stack_bytes = sum(h.bytes for h in handles)
        assert stack_bytes > 0
        resident = sum(
            p.data.numel() for h in handles for p in h.params if p.data.is_cuda
        )
        assert resident == 0
        assert alloc_before - alloc_after >= 0.8 * stack_bytes

    def test_recompute_grads_match_same_quantized_weights(self):
        """Reference and offloaded runs share the same quantized model (dequantization is
        deterministic), so outputs and trainable grads must match across the install."""
        torch.manual_seed(0)
        model, d = self._build()
        x = torch.randn(2, 8, d, device="cuda", dtype=torch.float16)

        model.zero_grad()
        out_ref = model(x)
        out_ref.sum().backward()
        ref_grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        install_expert_offload(model, device="cuda", pin=True)
        model.zero_grad()
        out_off = model(x)
        out_off.sum().backward()
        off_grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        assert torch.allclose(out_ref, out_off, atol=1e-3, rtol=1e-2)
        assert set(ref_grads) == set(off_grads) and len(ref_grads) > 0
        for name, g in ref_grads.items():
            assert torch.allclose(g, off_grads[name], atol=1e-3, rtol=1e-2), name

    def test_state_dict_keeps_packed_stacks_while_evicted(self):
        torch.manual_seed(0)
        model, _d = self._build()
        packed_numels = {
            n: p.numel() for n, p in model.named_parameters() if n.endswith(".original")
        }
        install_expert_offload(model, device="cuda", pin=True)
        sd = model.state_dict()
        # bitsandbytes' own post-hook renames parametrizations.<p>.original -> <p>; our hook must
        # still substitute the full-size home wherever the packed tensor landed.
        for name, numel in packed_numels.items():
            clean = name.replace(".parametrizations", "").replace(".original", "")
            t = sd.get(name) if name in sd else sd.get(clean)
            assert t is not None and t.numel() == numel
