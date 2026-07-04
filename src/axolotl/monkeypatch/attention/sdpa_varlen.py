"""Variable-length (cu_seqlens) SDPA path for sample packing.

With sample packing the model concatenates many documents into one row and
encodes the boundaries in ``position_ids`` (which reset to 0 at each document
start). The default SDPA path turns this into an explicit 4D block-diagonal
mask: O(S^2) compute even though cross-document blocks are masked out, plus the
mask tensor itself.

When PyTorch exposes ``torch.nn.attention.varlen.varlen_attn`` (>= 2.10) and the
head_dim is within Flash-Attention's limit (<= 256), we can instead run the
attention as variable-length with ``cu_seqlens`` derived from ``position_ids``,
which skips the cross-document blocks entirely — faster and lower memory — with
no dependency on the ``flash_attn`` package. It only activates for genuinely
multi-document (packed) rows; everything else falls back to the stock SDPA
implementation. Auto-enabled for ``sdpa`` + ``sample_packing`` when the kernel
can serve the model (see ``PatchManager._apply_sdpa_varlen_patch``); ``sdpa_varlen``
overrides the choice.

To make varlen actually engage during training (``use_cache=False``), patching
also overrides the ``sdpa`` mask builder to return ``None`` for packed rows whose
2D padding mask has been dropped. Otherwise transformers materializes a 4D
block-diagonal mask from ``position_ids`` and hands it to the attention interface,
which would keep the stock O(S^2) SDPA path (and, before the padding mask was
dropped for sdpa packing, silently leak across documents on padded rows).
"""

from __future__ import annotations

from typing import Any, Callable

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCH_APPLIED = False
_MASK_PATCH_APPLIED = False
# head_dim limit of the Flash-Attention kernel backing varlen_attn.
_VARLEN_MAX_HEAD_DIM = 256


def varlen_available() -> bool:
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _is_packed(position_ids) -> bool:
    """More document starts (position resets to 0) than rows -> packed."""
    pid = position_ids if position_ids.dim() > 1 else position_ids[None]
    return int((pid == 0).sum()) > pid.shape[0]


def _block_diagonal_causal_mask(position_ids, seq_len: int):
    """Bool (B, 1, S, S) block-diagonal causal mask (True = attend) built from
    per-document-resetting position_ids. Used only on the non-varlen fallback so
    stock SDPA still isolates documents when the 2D padding mask has been dropped."""
    import torch

    pid = position_ids if position_ids.dim() > 1 else position_ids[None]
    doc = (pid == 0).cumsum(-1)  # document id per token
    same_doc = doc[:, :, None] == doc[:, None, :]
    idx = torch.arange(seq_len, device=pid.device)
    causal = idx[:, None] >= idx[None, :]
    return (same_doc & causal[None])[:, None]


def _build_varlen_forward(original_sdpa: Callable) -> Callable:
    import inspect

    import torch
    from torch.nn.attention.varlen import varlen_attn
    from transformers.modeling_flash_attention_utils import (
        prepare_fa_kwargs_from_position_ids,
    )

    # varlen_attn's causal API differs across the supported torch range (>=2.10): torch 2.11 takes
    # window_size (causal = (-1, 0): unlimited left, no right; sliding = (W-1, 0)), earlier builds
    # take is_causal (causal only, no sliding). Detect which the installed build accepts. Scale stays
    # default (1/sqrt(d)) — the use_varlen guard below already restricts to standard scaling.
    try:
        _varlen_params = set(inspect.signature(varlen_attn).parameters)
    except (TypeError, ValueError):
        _varlen_params = set()
    # window_size present -> use it; only an is_causal-only build lacks sliding support.
    _supports_window = (
        "window_size" in _varlen_params or "is_causal" not in _varlen_params
    )

    def sdpa_varlen_forward(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ):
        position_ids = kwargs.get("position_ids")
        sliding_window = kwargs.get("sliding_window", None) or getattr(
            module, "sliding_window", None
        )
        head_dim = query.shape[-1]
        # Fast-path conditions; anything else falls back to stock SDPA.
        # - attention_mask must be None: packing carries structure via position_ids; a real mask
        #   (e.g. left padding) isn't expressible to the causal/sliding varlen kernel here.
        # - dropout unsupported by varlen_attn.
        # - head_dim within the Flash limit.
        # - scaling: varlen_attn only applies 1/sqrt(head_dim); a custom scale can't be honored.
        standard_scale = scaling is None or abs(scaling - head_dim**-0.5) < 1e-9
        use_varlen = (
            attention_mask is None
            and not dropout
            and head_dim <= _VARLEN_MAX_HEAD_DIM
            and position_ids is not None
            and standard_scale
        )
        packed = position_ids is not None and _is_packed(position_ids)
        use_varlen = use_varlen and packed  # single-doc rows -> stock SDPA
        if not use_varlen:
            # Rebuild isolation the dropped mask no longer provides, else stock SDPA leaks.
            if attention_mask is None and packed:
                attention_mask = _block_diagonal_causal_mask(
                    position_ids, query.shape[2]
                )
            return original_sdpa(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )

        # A sliding window needs varlen_attn's window_size arg; an is_causal-only build can't express
        # it, so refuse loudly there rather than silently running full causal attention (wrong).
        if sliding_window and not _supports_window:
            raise NotImplementedError(
                "sdpa_varlen: sliding-window attention needs varlen_attn(window_size=...), absent in "
                f"this torch build (requested window={sliding_window}); disable sdpa_varlen for this model."
            )

        B, Hq, S, D = query.shape
        Hkv = key.shape[1]
        if Hq != Hkv:  # GQA -> repeat (varlen_attn has no GQA mode)
            n = Hq // Hkv
            key = key.repeat_interleave(n, dim=1)
            value = value.repeat_interleave(n, dim=1)
        pid = position_ids if position_ids.dim() > 1 else position_ids[None]
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pid)
        qf = query.transpose(1, 2).reshape(B * S, Hq, D)
        kf = key.transpose(1, 2).reshape(B * S, Hq, D)
        vf = value.transpose(1, 2).reshape(B * S, Hq, D)
        if _supports_window:
            # (left, right): (-1, 0) = causal full; (W-1, 0) = causal sliding window of W.
            window = (sliding_window - 1, 0) if sliding_window else (-1, 0)
            causal_kw: dict = {"window_size": window}
        else:
            causal_kw = {
                "is_causal": True
            }  # is_causal-only build (sliding already refused above)
        out = varlen_attn(
            qf,
            kf,
            vf,
            cu_q.to(torch.int32),
            cu_k.to(torch.int32),
            int(max_q),
            int(max_k),
            **causal_kw,
        )
        if isinstance(out, tuple):
            out = out[0]
        # match sdpa_attention_forward's return contract: (attn_output [B,S,Hq,D], None)
        return out.reshape(B, S, Hq, D), None

    return sdpa_varlen_forward


def _build_varlen_mask(original_mask: Callable) -> Callable:
    def sdpa_varlen_mask(*args: Any, **kwargs: Any):
        # None for packed (mask-dropped) rows lets the varlen wrapper own isolation via
        # cu_seqlens, mirroring transformers' `flash_attention_mask`; else defer to stock.
        if kwargs.get("attention_mask") is None:
            return None
        return original_mask(*args, **kwargs)

    return sdpa_varlen_mask


def patch_sdpa_varlen() -> bool:
    """Replace the registered ``sdpa`` attention with a varlen-aware wrapper (idempotent).

    Also overrides the ``sdpa`` mask builder to return ``None`` for packed rows whose
    padding mask was dropped, so the varlen wrapper actually engages during training
    (``use_cache=False``) instead of transformers building a 4D block-diagonal mask that
    would otherwise force the stock O(S^2) SDPA path. Only call this for models the varlen
    path can serve (head_dim <= 256, no sliding window); other cases keep stock SDPA which
    decontaminates via the dropped-mask block-diagonal path.
    """
    global _PATCH_APPLIED, _MASK_PATCH_APPLIED
    if _PATCH_APPLIED:
        return True
    if not varlen_available():
        LOG.warning(
            "sdpa_varlen: torch.nn.attention.varlen.varlen_attn unavailable (needs torch >= 2.10); "
            "leaving stock SDPA in place."
        )
        return False
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    original_mask = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    wrapper = _build_varlen_forward(original)
    wrapper._axolotl_sdpa_original = original  # type: ignore[attr-defined]
    mask_wrapper = _build_varlen_mask(original_mask)
    mask_wrapper._axolotl_sdpa_mask_original = original_mask  # type: ignore[attr-defined]

    # Register both or neither, so a mid-way failure can't leave sdpa half-patched.
    ALL_ATTENTION_FUNCTIONS.register("sdpa", wrapper)
    try:
        ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", mask_wrapper)
    except Exception:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
        raise
    _PATCH_APPLIED = True
    _MASK_PATCH_APPLIED = True

    LOG.info(
        "sdpa_varlen: patched 'sdpa' to use cu_seqlens varlen_attn for packed rows "
        "(head_dim <= %d), falling back to stock SDPA otherwise",
        _VARLEN_MAX_HEAD_DIM,
    )
    return True


def unpatch_sdpa_varlen() -> None:
    global _PATCH_APPLIED, _MASK_PATCH_APPLIED
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _PATCH_APPLIED:
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
        original = getattr(current, "_axolotl_sdpa_original", None)
        if original is not None:
            ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
        _PATCH_APPLIED = False

    if _MASK_PATCH_APPLIED:
        current_mask = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
        original_mask = getattr(current_mask, "_axolotl_sdpa_mask_original", None)
        if original_mask is not None:
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", original_mask)
        _MASK_PATCH_APPLIED = False
