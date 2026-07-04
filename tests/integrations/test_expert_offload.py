"""Tests for the expert-offload integration (single-GPU CPU offload of frozen 4-bit MoE experts).

The scheduling/recompute correctness — the load-bearing, tricky part — is validated on CPU with a
fake MoE whose experts are plain ``F.linear`` layers carrying a ``Params4bit``-named weight: the
mechanism only moves ``weight.data``, so this exercises the full stage/evict + gradient-checkpoint
recompute path without a GPU or bitsandbytes. A CUDA-gated test then confirms it end-to-end against
real ``bitsandbytes.nn.Linear4bit`` experts.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from axolotl.integrations.expert_offload import ExpertOffloadArgs, ExpertOffloadPlugin
from axolotl.integrations.expert_offload.offload import (
    _PLACEHOLDERS,
    _base_layer,
    _BlockOffload,
    _is_linear4bit,
    find_moe_expert_blocks,
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
class TestArgs:
    def test_defaults(self):
        a = ExpertOffloadArgs()
        assert a.expert_offload is False
        assert a.expert_offload_pin_memory is True

    def test_enabled(self):
        assert ExpertOffloadArgs(expert_offload=True).expert_offload is True


def _cfg(**overrides):
    base = dict(
        expert_offload=True,
        load_in_4bit=True,
        adapter="qlora",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp_config=None,
        fsdp=None,
        deepspeed=None,
        expert_parallel_size=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestValidation:
    def test_valid_config_passes(self):
        ExpertOffloadPlugin._validate(_cfg())  # must not raise

    def test_disabled_skips_validation(self):
        # pre_model_load must be a no-op when the feature is off, even with an otherwise bad config.
        ExpertOffloadPlugin().pre_model_load(
            _cfg(expert_offload=False, load_in_4bit=False)
        )

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
            (dict(fsdp_config={"x": 1}), "FSDP"),
            (dict(fsdp=["full_shard"]), "FSDP"),
            (dict(deepspeed="ds.json"), "DeepSpeed"),
            (dict(expert_parallel_size=2), "expert_parallel"),
        ],
    )
    def test_invalid_configs_raise(self, override, needle):
        with pytest.raises(ValueError, match=needle):
            ExpertOffloadPlugin._validate(_cfg(**override))

    def test_lora_with_4bit_is_accepted(self):
        ExpertOffloadPlugin._validate(
            _cfg(adapter="lora")
        )  # lora + load_in_4bit is fine


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
            for base in h.base_layers:
                assert base.weight.data.numel() == 0
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
# CUDA + bitsandbytes: the real Linear4bit path                               #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + bitsandbytes"
)
class TestCudaLinear4bit:
    @staticmethod
    def _build(d=512, n_experts=16, n_layers=4):
        import bitsandbytes as bnb

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
            b.weight.data.numel()
            for h in handles
            for b in h.base_layers
            if b.weight.data.is_cuda
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
