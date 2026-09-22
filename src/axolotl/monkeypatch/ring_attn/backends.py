"""Flash attention kernel backends for ring attention.

Ring attention needs the *split* kernels (a forward that returns the log-sum-exp and a
backward that accumulates into caller-owned dq/dk/dv) rather than the public
`flash_attn_func` entry points, and every flash attention flavour exposes those under a
different private API:

- FA2 (`flash_attn`, `kernels-community/flash-attn2`): separate batch and varlen
  kernels, `_flash_attn[_varlen]_forward` / `_flash_attn[_varlen]_backward`.
- FA3 (`flash-attn-3`, `kernels-community/flash-attn3`,
  `kernels-community/vllm-flash-attn3`): one `_flash_attn_forward` / `_flash_attn_backward`
  taking optional `cu_seqlens`.
- FA4 (`flash-attn-4`, `kernels-community/flash-attn4`): CuTe DSL `_flash_attn_fwd` /
  `_flash_attn_bwd`.

`resolve_flash_backend` maps a transformers `attn_implementation` onto one of these,
mirroring transformers' own resolution order (installed package first, then the
kernels-hub fallback repo).
"""

import importlib
from types import ModuleType

import torch

from axolotl.monkeypatch.ring_attn.utils import get_default_args
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import attn_impl_base

LOG = get_logger(__name__)

NO_WINDOW = (-1, -1)

# Installed-package module holding each flavour's private kernels.
_PIP_MODULES = {
    "flash_attention_2": "flash_attn.flash_attn_interface",
    "flash_attention_3": "flash_attn_interface",
    "flash_attention_4": "flash_attn.cute.interface",
}


def _call(func, overrides: dict):
    """Call `func` by keyword with every parameter filled, ignoring overrides it lacks.

    The private kernels have no defaults for most parameters and their names drift
    between builds (`window_size` vs `window_size_left/right`, `causal` vs
    `is_causal`), so callers pass every spelling and the signature decides.
    """
    params = get_default_args(func)
    params.update({name: value for name, value in overrides.items() if name in params})
    return func(**params)


def _copy_grads_if_needed(result, dq, dk, dv):
    """Some kernels hand back fresh dq/dk/dv instead of writing into the given buffers."""
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        return
    for buffer, grad in zip((dq, dk, dv), result[:3], strict=False):
        if (
            isinstance(grad, torch.Tensor)
            and grad is not buffer
            and grad.data_ptr() != buffer.data_ptr()
        ):
            buffer.copy_(grad)


class FlashAttnBackend:
    """Uniform split forward/backward over one flash attention private API.

    Tensors use the flash-attn layout: `(batch, seqlen, nheads, head_dim)` in batch mode
    and `(total_tokens, nheads, head_dim)` in varlen mode (`cu_seqlens_q` given).
    `forward` returns `(out, lse)` with `lse` fp32 shaped `(batch, nheads, seqlen)` or
    `(nheads, total_tokens)`; `backward` writes into the `dq` / `dk` / `dv` buffers.
    """

    name = ""

    def __init__(self, module: ModuleType, source: str):
        self.module = module
        self.source = source

    def forward(  # pylint: disable=too-many-arguments
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int] = NO_WINDOW,
        softcap: float = 0.0,
        dropout_p: float = 0.0,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def backward(  # pylint: disable=too-many-arguments
        self,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int] = NO_WINDOW,
        softcap: float = 0.0,
        dropout_p: float = 0.0,
        deterministic: bool = False,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
    ) -> None:
        raise NotImplementedError


class FA2Backend(FlashAttnBackend):
    """`flash_attn.flash_attn_interface` and the `kernels-community/flash-attn2` build."""

    name = "flash_attention_2"

    def forward(self, q, k, v, *, softmax_scale, causal, window_size=NO_WINDOW, softcap=0.0, dropout_p=0.0, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):  # fmt: skip
        overrides = {
            "q": q,
            "k": k,
            "v": v,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": tuple(window_size),
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "softcap": softcap,
            "alibi_slopes": None,
            "return_softmax": False,
        }
        if cu_seqlens_q is None:
            outputs = _call(self.module._flash_attn_forward, overrides)
        else:
            overrides.update(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            outputs = _call(self.module._flash_attn_varlen_forward, overrides)
        # flash-attn < 2.7 returned (out, q, k, v, out_padded, lse, S_dmask, rng_state)
        if len(outputs) == 8:
            return outputs[0], outputs[5]
        return outputs[0], outputs[1]

    def backward(self, dout, q, k, v, out, lse, dq, dk, dv, *, softmax_scale, causal, window_size=NO_WINDOW, softcap=0.0, dropout_p=0.0, deterministic=False, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):  # fmt: skip
        overrides = {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": lse,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": tuple(window_size),
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "softcap": softcap,
            "alibi_slopes": None,
            "deterministic": deterministic,
            "rng_state": None,
        }
        if cu_seqlens_q is None:
            result = _call(self.module._flash_attn_backward, overrides)
        else:
            overrides.update(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            result = _call(self.module._flash_attn_varlen_backward, overrides)
        _copy_grads_if_needed(result, dq, dk, dv)


class FA3Backend(FlashAttnBackend):
    """Hopper `flash_attn_interface` and the FA3 kernels-hub builds (Dao and vLLM)."""

    name = "flash_attention_3"

    def forward(self, q, k, v, *, softmax_scale, causal, window_size=NO_WINDOW, softcap=0.0, dropout_p=0.0, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):  # fmt: skip
        overrides = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": tuple(window_size),
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "softcap": softcap,
        }
        out, lse, *_ = _call(self.module._flash_attn_forward, overrides)
        return out, lse

    def backward(self, dout, q, k, v, out, lse, dq, dk, dv, *, softmax_scale, causal, window_size=NO_WINDOW, softcap=0.0, dropout_p=0.0, deterministic=False, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):  # fmt: skip
        overrides = {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": lse,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "is_causal": causal,
            "window_size": tuple(window_size),
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "softcap": softcap,
            "deterministic": deterministic,
        }
        result = _call(self.module._flash_attn_backward, overrides)
        _copy_grads_if_needed(result, dq, dk, dv)


def _fa4_window(size: int) -> int | None:
    # FA4 spells "no window" as None rather than -1.
    return None if size is None or size < 0 else size


class FA4Backend(FlashAttnBackend):
    """`flash_attn.cute.interface` and the `kernels-community/flash-attn4` build."""

    name = "flash_attention_4"

    def forward(self, q, k, v, *, softmax_scale, causal, window_size=NO_WINDOW, softcap=0.0, dropout_p=0.0, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):  # fmt: skip
        overrides = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "softcap": softcap if softcap else None,
            "window_size_left": _fa4_window(window_size[0]),
            "window_size_right": _fa4_window(window_size[1]),
            "return_lse": True,
        }
        out, lse, *_ = _call(self.module._flash_attn_fwd, overrides)
        return out, lse

    def backward(self, dout, q, k, v, out, lse, dq, dk, dv, *, softmax_scale, causal, window_size=NO_WINDOW, softcap=0.0, dropout_p=0.0, deterministic=False, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):  # fmt: skip
        overrides = {
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "dout": dout,
            "lse": lse,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "softcap": softcap or 0.0,
            "window_size_left": _fa4_window(window_size[0]),
            "window_size_right": _fa4_window(window_size[1]),
            "deterministic": deterministic,
        }
        result = _call(self.module._flash_attn_bwd, overrides)
        _copy_grads_if_needed(result, dq, dk, dv)


def _candidate_modules(module: ModuleType):
    yield module
    # Hub builds re-export only the public API at the top level; the private kernels
    # live one submodule down.
    for attr in ("flash_attn_interface", "interface", "cute"):
        sub = getattr(module, attr, None)
        if isinstance(sub, ModuleType):
            yield sub


def backend_from_module(module: ModuleType, source: str) -> FlashAttnBackend | None:
    """Detect which private-API flavour `module` exposes, or None if it has none."""
    for candidate in _candidate_modules(module):
        if all(
            hasattr(candidate, name)
            for name in (
                "_flash_attn_forward",
                "_flash_attn_backward",
                "_flash_attn_varlen_forward",
                "_flash_attn_varlen_backward",
            )
        ):
            return FA2Backend(candidate, source)
        if hasattr(candidate, "_flash_attn_fwd") and hasattr(
            candidate, "_flash_attn_bwd"
        ):
            return FA4Backend(candidate, source)
        if hasattr(candidate, "_flash_attn_forward") and hasattr(
            candidate, "_flash_attn_backward"
        ):
            if "cu_seqlens_q" in get_default_args(candidate._flash_attn_forward):
                return FA3Backend(candidate, source)
    return None


def _import_optional(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def load_hub_kernel(attn_implementation: str) -> ModuleType:
    """Load `org/name[@rev][:kernel]` from the kernels hub the way transformers does."""
    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise ImportError(
            f"attn_implementation={attn_implementation!r} needs the `kernels` package: "
            "pip install kernels"
        ) from exc

    repo_id, _, revision = attn_implementation.split(":", 1)[0].partition("@")
    repo_id, revision = repo_id.strip(), revision.strip() or None
    if revision:
        return get_kernel(repo_id, revision=revision)

    try:
        from transformers.integrations.hub_kernels import get_attn_kernel_version

        return get_kernel(repo_id, version=get_attn_kernel_version(repo_id))
    except ImportError:
        return get_kernel(repo_id)


def resolve_flash_backend(attn_implementation: str | None) -> FlashAttnBackend:
    """Pick the split-kernel backend behind the model's `attn_implementation`."""
    impl = attn_impl_base(attn_implementation) or "flash_attention_2"

    if impl in _PIP_MODULES:
        module = _import_optional(_PIP_MODULES[impl])
        backend = backend_from_module(module, _PIP_MODULES[impl]) if module else None
        if backend is not None:
            return backend

        from transformers.modeling_flash_attention_utils import (
            FLASH_ATTN_KERNEL_FALLBACK,
        )

        repo = FLASH_ATTN_KERNEL_FALLBACK[impl]
        LOG.info(
            "%s is not installed as a package; loading ring attention kernels from %s",
            _PIP_MODULES[impl],
            repo,
        )
    else:
        repo = attn_implementation

    backend = backend_from_module(load_hub_kernel(repo), repo)
    if backend is None:
        raise ValueError(
            f"attn_implementation={attn_implementation!r} resolved to {repo!r}, which "
            "does not expose split flash attention kernels (forward returning the "
            "log-sum-exp plus an in-place backward), so ring attention cannot use it. "
            "Use flash_attention_2/3/4 or one of kernels-community/flash-attn2, "
            "flash-attn3, vllm-flash-attn3, flash-attn4."
        )
    return backend
