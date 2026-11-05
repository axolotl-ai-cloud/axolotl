"""The DoRA weight-norm cache on the LoRA-kernel path lives in process memory
only (``magnitude._dora_cache`` / ``proj._dora_norm_cache``) and is keyed on
``lora_A``/``lora_B`` ``_version``. These CPU unit tests pin the behaviors the
resume-divergence fix (issue #4239) relies on:

- the cached norm is served when ``_version`` matches and recomputed when it
  changes (e.g. after an optimizer step that mutates A/B in place);
- ``clear_dora_norm_caches`` removes both cache flavors so a run resumed from
  a checkpoint and an unbroken run recompute the same norms;
- simulating save -> load: a freshly loaded (cache-free) module recomputes
  norms equal to the original module's, and clearing the original's cache
  makes its next-step output bitwise equal to the resumed module's.
"""

import pytest
import torch
from torch import nn

from axolotl.kernels.lora import (
    _compute_dora_scale,
    _compute_dora_scale_cached,
    clear_dora_norm_caches,
)

OUT, IN, RANK = 16, 32, 4


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "W": torch.randn(OUT, IN, generator=g) * 0.1,
        "W_quant": None,
        "A": torch.randn(RANK, IN, generator=g) * 0.1,
        "B": torch.randn(OUT, RANK, generator=g) * 0.1,
        "s": 0.5,
        "magnitude": torch.randn(OUT, generator=g).abs(),
    }


def _reference(fx, dtype=torch.float32):
    combined = fx["W"] + fx["s"] * (fx["B"] @ fx["A"])
    return fx["magnitude"].to(dtype) / torch.linalg.norm(combined, dim=1)


def test_cache_hit_and_version_invalidation():
    """Cache is served on matching _version and recomputed when A/B mutate."""
    fx = _fixture()
    out1 = _compute_dora_scale(**fx, dtype=torch.float32)
    assert torch.equal(out1, _reference(fx))
    assert hasattr(fx["magnitude"], "_dora_cache")

    # serve from cache: bump A/B without changing values would break equality,
    # so instead verify the cache is actually used by storing a poisoned norm
    cached_ver = fx["magnitude"]._dora_cache
    assert cached_ver[0] == fx["A"]._version and cached_ver[1] == fx["B"]._version
    poisoned = cached_ver[2] + 1.0
    fx["magnitude"]._dora_cache = (cached_ver[0], cached_ver[1], poisoned)
    out_hit = _compute_dora_scale(**fx, dtype=torch.float32)
    assert torch.equal(out_hit, fx["magnitude"].to(torch.float32) / poisoned), (
        "cache hit must serve the cached norm without recomputation"
    )

    # in-place update of A/B (optimizer step) changes _version -> recompute
    with torch.no_grad():
        fx["A"].add_(0.01)
        fx["B"].add_(0.01)
    out2 = _compute_dora_scale(**fx, dtype=torch.float32)
    assert fx["magnitude"]._dora_cache[0] == fx["A"]._version
    assert torch.allclose(out2, _reference(fx)), (
        "cache must be invalidated when lora_A/lora_B _version changes"
    )


def test_cached_variant_recomputes_on_version_change():
    fx = _fixture()
    proj = nn.Linear(IN, OUT)

    # a magnitude parameter registered like peft's lora_magnitude_vector.weight
    mag_layer = nn.Linear(1, OUT, bias=False)
    mag_layer.weight.data = _fixture()["magnitude"].clone().abs()
    proj.lora_magnitude_vector = nn.ModuleDict({"default": mag_layer})

    out1 = _compute_dora_scale_cached(proj, **fx, dtype=torch.float32)
    assert torch.allclose(out1, _reference(fx))
    assert hasattr(proj, "_dora_norm_cache")

    # optimizer-step-like in-place updates change _version -> recompute
    with torch.no_grad():
        fx["A"].add_(0.05)
    out2 = _compute_dora_scale_cached(proj, **fx, dtype=torch.float32)
    assert torch.allclose(out2, _reference(fx))
    assert proj._dora_norm_cache[0] == fx["A"]._version


def test_clear_dora_norm_caches_removes_both_cache_flavors():
    fx = _fixture()
    proj = nn.Linear(IN, OUT)
    mag_layer = nn.Linear(1, OUT, bias=False)
    mag_layer.weight.data = fx["magnitude"].clone().abs()
    proj.lora_magnitude_vector = nn.ModuleDict({"default": mag_layer})

    # both cache flavors live on tensors/modules reachable from the module
    # tree: the magnitude parameter (via _compute_dora_scale) and the proj
    # module itself (via _compute_dora_scale_cached)
    _compute_dora_scale(
        fx["W"], fx["W_quant"], fx["A"], fx["B"], fx["s"], mag_layer.weight,
        torch.float32,
    )
    fx_no_mag = {k: v for k, v in fx.items() if k != "magnitude"}
    _compute_dora_scale_cached(
        proj, fx_no_mag["W"], fx_no_mag["W_quant"], fx_no_mag["A"], fx_no_mag["B"],
        fx_no_mag["s"], mag_layer.weight, torch.float32,
    )
    assert hasattr(mag_layer.weight, "_dora_cache")
    assert hasattr(proj, "_dora_norm_cache")

    cleared = clear_dora_norm_caches(proj)
    assert cleared == 2
    assert not hasattr(mag_layer.weight, "_dora_cache")
    assert not hasattr(proj, "_dora_norm_cache")
    # idempotent
    assert clear_dora_norm_caches(proj) == 0


def test_resume_equivalence_clearing_cache_matches_fresh_module():
    """Simulated checkpoint boundary: an unbroken run that clears its caches
    at save time must be bitwise equal, at the next step, to a run resumed
    from the checkpoint (fresh module, no caches)."""
    fx = _fixture()
    dtype = torch.float32

    # Register the magnitude as a module parameter, like peft's
    # lora_magnitude_vector.weight, so clear_dora_norm_caches can reach it.
    mag = nn.Linear(1, OUT, bias=False)
    mag.weight.data = fx["magnitude"].clone().abs()
    root = nn.Module()
    root.mag = mag

    unbroken = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in fx.items()}
    unbroken["magnitude"] = mag.weight  # cache lives on this Parameter
    resumed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in fx.items()}

    # unbroken leg: cache populated at step 0
    _compute_dora_scale(**unbroken, dtype=dtype)
    stale_norm = unbroken["magnitude"]._dora_cache[2]

    # 3 fused-optimizer-style updates that bump nothing observable: A/B values
    # move in place while the cache key keeps matching the (unchanged-looking)
    # version counters, so the unbroken leg keeps serving the stale norm
    for _ in range(3):
        with torch.no_grad():
            unbroken["A"].add_(0.01)
            resumed["A"].add_(0.01)
            unbroken["B"].add_(0.01)
            resumed["B"].add_(0.01)
        unbroken["magnitude"]._dora_cache = (
            unbroken["A"]._version, unbroken["B"]._version, stale_norm,
        )

    stale_unbroken = _compute_dora_scale(**unbroken, dtype=dtype)
    fresh_resumed = _compute_dora_scale(**resumed, dtype=dtype)
    if stale_unbroken.equal(fresh_resumed):
        pytest.skip("emulated stale cache did not diverge from recomputation")

    # the fix: clear caches at the checkpoint boundary on the unbroken leg
    assert hasattr(mag.weight, "_dora_cache")
    assert clear_dora_norm_caches(root) == 1
    assert not hasattr(mag.weight, "_dora_cache")

    cleared_unbroken = _compute_dora_scale(**unbroken, dtype=dtype)
    assert torch.equal(cleared_unbroken, fresh_resumed), (
        "after clearing caches the unbroken leg must be bitwise equal to the resumed leg"
    )
