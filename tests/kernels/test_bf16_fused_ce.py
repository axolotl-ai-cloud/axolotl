"""Correctness for the chunked bf16 lm_head + cross-entropy.

The chunked path must produce a loss and dL/dhidden that match the same-weight
materialized ``F.cross_entropy`` reference (bit-close, not approximate), and both
must be FINITE — it is the convergence-safe alternative to the fused CCE/Liger
paths that collapsed under NVFP4 stochastic-rounding grads.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    pytest.skip("CUDA + bf16 required for bf16 fused CE", allow_module_level=True)

from axolotl.integrations.nvfp4.kernels.bf16_fused_ce import (  # noqa: E402
    bf16_lm_head_cross_entropy,
)


@pytest.mark.parametrize("num_items", [None, 137.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bf16_fused_ce_matches_materialized(num_items, dtype):
    torch.manual_seed(0)
    M, H, V = 192, 256, 4096 + 512  # crosses a vocab-tile boundary
    lm_head = nn.Linear(H, V, bias=False).cuda().to(dtype)
    lm_head.weight.requires_grad_(False)

    hidden = torch.randn(M, H, device="cuda", dtype=dtype)
    labels = torch.randint(0, V, (M,), device="cuda")
    labels[::7] = -100  # mask some tokens

    # reference: full bf16 logits (upcast to fp32), standard CE
    h_ref = hidden.clone().requires_grad_(True)
    logits = (h_ref @ lm_head.weight.t()).float()
    reduction = "sum" if num_items is not None else "mean"
    ref = F.cross_entropy(logits, labels, ignore_index=-100, reduction=reduction)
    if num_items is not None:
        ref = ref / num_items
    ref.backward()

    # fused (shift=False to align with the un-shifted reference)
    h_fused = hidden.clone().requires_grad_(True)
    fused = bf16_lm_head_cross_entropy(
        h_fused, lm_head, labels, num_items_in_batch=num_items, shift=False
    )
    fused.backward()

    assert torch.isfinite(fused).all()
    assert torch.isfinite(h_fused.grad).all()

    loss_rel = (fused - ref).abs() / (ref.abs() + 1e-9)
    grad_rel = (h_fused.grad - h_ref.grad).float().norm() / (
        h_ref.grad.float().norm() + 1e-9
    )
    # fp32 is bit-tight; bf16 carries the GEMM's intrinsic rounding noise.
    loss_tol = 1e-6 if dtype == torch.float32 else 1e-4
    grad_tol = 1e-5 if dtype == torch.float32 else 5e-3
    assert loss_rel < loss_tol, (dtype, num_items, loss_rel.item())
    assert grad_rel < grad_tol, (dtype, num_items, grad_rel.item())


def test_bf16_fused_ce_all_masked_is_finite():
    """A fully-masked microbatch must give a finite zero loss / zero grad."""
    torch.manual_seed(0)
    M, H, V = 64, 128, 8192
    lm_head = nn.Linear(H, V, bias=False).cuda().bfloat16()
    lm_head.weight.requires_grad_(False)

    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.full((M,), -100, device="cuda")
    loss = bf16_lm_head_cross_entropy(hidden, lm_head, labels, shift=False)
    loss.backward()

    assert torch.isfinite(loss).all()
    assert float(loss) == 0.0
    assert torch.isfinite(hidden.grad).all()
    assert float(hidden.grad.abs().max()) == 0.0


def test_bf16_fused_ce_rejects_non_plain_head():
    """Trainable / biased lm_head -> None (caller falls back to materialized)."""
    H, V = 128, 4096
    labels = torch.randint(0, V, (32,), device="cuda")
    hidden = torch.randn(32, H, device="cuda", dtype=torch.bfloat16)

    trainable = nn.Linear(H, V, bias=False).cuda().bfloat16()  # requires_grad default
    assert bf16_lm_head_cross_entropy(hidden, trainable, labels) is None

    biased = nn.Linear(H, V, bias=True).cuda().bfloat16()
    biased.weight.requires_grad_(False)
    assert bf16_lm_head_cross_entropy(hidden, biased, labels) is None


def test_bf16_fused_ce_logit_scale_matches_materialized():
    """logit_scale must scale the logits before CE, matching the materialized
    reference in loss and dL/dhidden."""
    torch.manual_seed(0)
    scale = 0.0625
    M, H, V = 192, 256, 4096 + 512
    lm_head = nn.Linear(H, V, bias=False).cuda().bfloat16()
    lm_head.weight.requires_grad_(False)

    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (M,), device="cuda")
    labels[::7] = -100

    h_ref = hidden.clone().requires_grad_(True)
    logits = (h_ref @ lm_head.weight.t()).float() * scale
    ref = F.cross_entropy(logits, labels, ignore_index=-100)
    ref.backward()

    h_fused = hidden.clone().requires_grad_(True)
    fused = bf16_lm_head_cross_entropy(
        h_fused, lm_head, labels, shift=False, logit_scale=scale
    )
    fused.backward()

    loss_rel = (fused - ref).abs() / (ref.abs() + 1e-9)
    grad_rel = (h_fused.grad - h_ref.grad).float().norm() / (
        h_ref.grad.float().norm() + 1e-9
    )
    assert loss_rel < 1e-4, loss_rel.item()
    assert grad_rel < 5e-3, grad_rel.item()


def test_config_logit_scale_variants():
    from axolotl.integrations.nvfp4.kernels.bf16_fused_ce import _config_logit_scale

    assert _config_logit_scale(SimpleNamespace(final_logit_softcapping=30.0)) is None
    assert _config_logit_scale(SimpleNamespace(logit_scale=0.0625)) == 0.0625
    assert _config_logit_scale(SimpleNamespace(logits_scaling=8.0)) == 0.125
    assert _config_logit_scale(SimpleNamespace()) == 1.0


def test_bf16_fused_ce_rejects_out_of_range_labels():
    """Labels outside [0, vocab) that aren't ignore_index must raise (the clamped
    tile gather would otherwise produce a finite garbage loss)."""
    torch.manual_seed(0)
    M, H, V = 64, 128, 4096
    lm_head = nn.Linear(H, V, bias=False).cuda().bfloat16()
    lm_head.weight.requires_grad_(False)
    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (M,), device="cuda")

    too_big = labels.clone()
    too_big[3] = V
    with pytest.raises(ValueError, match="out of range"):
        bf16_lm_head_cross_entropy(hidden, lm_head, too_big, shift=False)

    negative = labels.clone()
    negative[5] = -7
    with pytest.raises(ValueError, match="out of range"):
        bf16_lm_head_cross_entropy(hidden, lm_head, negative, shift=False)

    # ignore_index rows are exempt from the range check.
    masked = labels.clone()
    masked[::3] = -100
    loss = bf16_lm_head_cross_entropy(hidden, lm_head, masked, shift=False)
    assert loss is not None and torch.isfinite(loss)


def test_fused_forward_falls_through_on_router_logits():
    """MoE aux-loss safety: output_router_logits (kwarg or config) must route to
    the original forward — the fused path computes pure CE only."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from axolotl.integrations.nvfp4.kernels.bf16_fused_ce import _make_fused_forward

    torch.manual_seed(0)
    H, V = 64, 512
    hidden = torch.randn(2, 8, H, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (2, 8), device="cuda")

    class Base(nn.Module):
        def __init__(self, h):
            super().__init__()
            self._h = h

        def forward(self, *args, **kwargs):
            return SimpleNamespace(last_hidden_state=self._h)

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(output_router_logits=False)
            self._axolotl_bf16_lm_head_ce_enabled = True
            self.model = Base(hidden)
            self.lm_head = nn.Linear(H, V, bias=False).cuda().bfloat16()
            self.lm_head.weight.requires_grad_(False)

        def get_output_embeddings(self):
            return self.lm_head

    calls = []

    def orig_forward(self, *args, **kwargs):
        calls.append(kwargs)
        return "orig"

    fused_forward = _make_fused_forward(orig_forward)
    mod = Fake()
    mod.train()

    # Control: the fused path runs (orig NOT called) and yields a real CE loss.
    out = fused_forward(mod, labels=labels)
    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits is None and torch.isfinite(out.loss)
    assert not calls

    # Kwarg: must fall through to the original forward.
    assert fused_forward(mod, labels=labels, output_router_logits=True) == "orig"
    assert len(calls) == 1

    # Config flag: must fall through too.
    mod.config.output_router_logits = True
    assert fused_forward(mod, labels=labels) == "orig"
    assert len(calls) == 2


def test_patch_refuses_final_logit_softcapping():
    """patch_model must refuse a softcapped architecture (tanh capping is not
    expressible in the streaming kernel) while otherwise-identical configs patch."""
    from axolotl.integrations.nvfp4.kernels.bf16_fused_ce import (
        _PATCHED_FORWARDS,
        patch_model_bf16_lm_head_cross_entropy,
    )

    def make_causal_cls(softcapping):
        class FakeCausal(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(final_logit_softcapping=softcapping)
                self.lm_head = nn.Linear(64, 128, bias=False).cuda().bfloat16()
                self.lm_head.weight.requires_grad_(False)

            def get_output_embeddings(self):
                return self.lm_head

            def forward(self, *args, **kwargs):
                return "orig"

        return FakeCausal

    capped_cls = make_causal_cls(30.0)
    orig_forward = capped_cls.forward
    assert patch_model_bf16_lm_head_cross_entropy(capped_cls()) is False
    assert capped_cls.forward is orig_forward
    assert capped_cls not in _PATCHED_FORWARDS

    # Identical module minus the softcap: the other preconditions do pass.
    plain_cls = make_causal_cls(None)
    assert patch_model_bf16_lm_head_cross_entropy(plain_cls()) is True
    assert plain_cls in _PATCHED_FORWARDS
