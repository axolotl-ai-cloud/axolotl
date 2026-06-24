"""Generic large-head-dim attention capability (head_dim > 256).

Flash/cuDNN SDPA cap at head_dim 256; at larger head_dim PyTorch falls to the memory-efficient or
math backend. The Triton ``flash_d512`` kernel fills that gap. This module exposes the routing as a
MODEL-AGNOSTIC capability: any attention path can call :func:`flash_d512_route`, and a generic
``sdpa`` wrapper (:func:`patch_sdpa_large_head`) lets plain SDPA models opt in via the
``large_head_attention`` config. Gemma-4's hybrid global layers reuse the same router.

Policy (``large_head_attention``):
  - ``sdpa``         : never use the Triton kernel (stock SDPA at large head_dim)
  - ``auto``         : Triton flash for genuinely packed rows (its proven win), SDPA otherwise
                       (single-document large-head attention is faster on SDPA is_causal)
  - ``triton_flash`` : prefer the Triton kernel for head_dim > 256 whenever possible
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_LARGE_HEAD_MIN_DIM = 256
_POLICY = "sdpa"
_SDPA_ORIG_ATTR = "_axolotl_large_head_sdpa_original"


def set_large_head_policy(policy: str | None) -> None:
    global _POLICY
    _POLICY = str(policy).lower() if policy else "sdpa"


def get_large_head_policy() -> str:
    return _POLICY


def resolve_large_head_policy(cfg) -> str:
    """Resolve the intent ``large_head_attention`` (auto/sdpa/triton_flash); fall back to the
    deprecated ``flash_attn_d512`` bool (True -> auto, since flash only wins on packed rows)."""
    policy = cfg.get("large_head_attention")
    if policy:
        return str(policy).lower()
    if cfg.get("flash_attn_d512"):
        return "auto"
    return "sdpa"


def _multidoc_position_ids(position_ids):
    """Return [B,S] position_ids iff they encode genuine (multi-document) packing, else None."""
    if position_ids is None:
        return None
    p = position_ids if position_ids.dim() > 1 else position_ids[None]
    return p if int((p == 0).sum()) > p.shape[0] else None


def flash_d512_route(module, query, key, value, scaling, position_ids, policy=None):
    """Route a large-head attention call through the Triton flash_d512 kernel, or return None to
    signal the caller to fall back to SDPA. Inputs are ``[B, H, S, D]``; on success returns
    ``(attn_output [B, S, Hq, D], None)`` matching ``sdpa_attention_forward``'s contract."""
    policy = policy or _POLICY
    # Allowlist: only the Triton policies route to the kernel, so a config typo can't enable it.
    if policy not in ("auto", "triton_flash") or query.shape[-1] <= _LARGE_HEAD_MIN_DIM:
        return None
    pid = _multidoc_position_ids(position_ids)
    # auto: kernel only for packed rows (single-doc large-head is faster on SDPA is_causal)
    if policy == "auto" and pid is None:
        return None
    try:
        from axolotl.monkeypatch.attention.flash_attn_d512 import flash_d512

        ng = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
        k = key.repeat_interleave(ng, dim=1) if ng > 1 else key
        v = value.repeat_interleave(ng, dim=1) if ng > 1 else value
        out = flash_d512(query, k, v, True, position_ids=pid, scale=scaling)
        return out.transpose(1, 2).contiguous(), None
    except Exception:  # pragma: no cover - any kernel issue falls back to SDPA
        return None


def patch_sdpa_large_head(policy: str | None = None) -> bool:
    """Wrap the ``sdpa`` attention interface so head_dim>256 maskless calls route through the Triton
    kernel per ``policy`` (idempotent). Generic — any SDPA model opts in via config; explicit
    attention masks always fall through to stock SDPA."""
    if policy is not None:
        set_large_head_policy(policy)
    if get_large_head_policy() == "sdpa":
        return False
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(current, _SDPA_ORIG_ATTR, None) is not None:
        return True
    original = current

    def sdpa_large_head_forward(module, query, key, value, attention_mask, **kwargs):
        if attention_mask is None:
            routed = flash_d512_route(
                module,
                query,
                key,
                value,
                kwargs.get("scaling"),
                kwargs.get("position_ids"),
            )
            if routed is not None:
                return routed
        return original(module, query, key, value, attention_mask, **kwargs)

    setattr(sdpa_large_head_forward, _SDPA_ORIG_ATTR, original)
    ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa_large_head_forward)
    LOG.info(
        "large_head_attention: wrapped sdpa to route head_dim>256 through Triton flash (%s)",
        get_large_head_policy(),
    )
    return True


def unpatch_sdpa_large_head() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    original = getattr(current, _SDPA_ORIG_ATTR, None)
    if original is not None:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
