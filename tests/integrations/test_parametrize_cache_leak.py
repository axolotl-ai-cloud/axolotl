"""The bnb parametrize-cache leak and its expert_offload-side neutralization.

bitsandbytes' ``replace_parameter_4bit`` registers a forward_pre_hook/forward_hook
pair that increments/decrements torch's GLOBAL ``parametrize._cache_enabled`` and
clears ``parametrize._cache`` only when the counter reaches 0. torch's
``use_reentrant=False`` checkpointing aborts the backward recompute mid-forward by
design (early stop) — the post-hook of the module holding the last recomputed save
is SKIPPED, the counter leaks upward every step, and once it is stuck above zero
every dequantized weight is cached globally for the rest of training (pool x4
bytes). These tests reproduce the torch-level skip with bnb-shaped stub hooks (no
bitsandbytes import needed) and pin the stripper's behavior.
"""

import pytest
import torch
import torch.nn.utils.parametrize as P
from torch import nn
from torch.utils.checkpoint import checkpoint

from axolotl.integrations.expert_offload.offload import (
    _BNB_CACHE_HOOK_NAMES,
    _reset_parametrize_cache_state,
    _strip_bnb_parametrize_cache_hooks,
)


# bnb-shaped stubs: same names, same global side effects (bitsandbytes/nn/parametrize.py)
def _enable_parametrization_cache(module, inputs):
    P._cache_enabled += 1


def _disable_parametrization_cache(module, inputs, output):
    P._cache_enabled -= 1
    if not P._cache_enabled:
        P._cache = {}


class _Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.tail = (nn.Linear(8, 8) for _ in range(3))

    def forward(self, x):
        return self.tail(self.b(self.a(x)))


def _register_bnb_shaped_hooks(module):
    module.register_forward_pre_hook(_enable_parametrization_cache)
    module.register_forward_hook(_disable_parametrization_cache)


@pytest.fixture(autouse=True)
def _clean_global_cache():
    _reset_parametrize_cache_state()
    yield
    _reset_parametrize_cache_state()


def test_checkpoint_early_stop_leaks_the_counter():
    """The torch-level mechanism: the tail module's post-hook is skipped during the
    early-stopped backward recompute, so the bnb-shaped counter never returns to 0."""
    m = _Chain()
    _register_bnb_shaped_hooks(m.tail)
    x = torch.randn(2, 8, requires_grad=True)
    checkpoint(m, x, use_reentrant=False).sum().backward()
    assert P._cache_enabled > 0, (
        "expected the early-stopped recompute to skip the forward_hook and leak the "
        "counter; if this now passes with 0, torch fixed the skip and the strip "
        "workaround can be retired"
    )


def test_strip_removes_exactly_the_bnb_pair_and_keeps_others():
    m = _Chain()
    _register_bnb_shaped_hooks(m.tail)
    seen = []
    m.tail.register_forward_hook(lambda mod, i, o: seen.append("kept"))

    removed = _strip_bnb_parametrize_cache_hooks(m.tail)
    assert removed == 2
    names = [getattr(fn, "__name__", "") for fn in m.tail._forward_pre_hooks.values()]
    names += [getattr(fn, "__name__", "") for fn in m.tail._forward_hooks.values()]
    assert not set(names) & set(_BNB_CACHE_HOOK_NAMES)

    m.tail(torch.randn(2, 8))
    assert seen == ["kept"]  # unrelated hooks survive
    assert _strip_bnb_parametrize_cache_hooks(m.tail) == 0  # idempotent


def test_stripped_module_cannot_leak_under_checkpointing():
    m = _Chain()
    for mod in (m.a, m.b, m.tail):
        _register_bnb_shaped_hooks(mod)
        _strip_bnb_parametrize_cache_hooks(mod)
    _reset_parametrize_cache_state()
    for _ in range(3):  # multi-step training shape
        x = torch.randn(2, 8, requires_grad=True)
        checkpoint(m, x, use_reentrant=False).sum().backward()
    assert P._cache_enabled == 0
    assert P._cache == {}
