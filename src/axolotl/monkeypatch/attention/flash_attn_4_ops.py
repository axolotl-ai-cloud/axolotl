"""Dispatcher-visible FA4 attention, so selective checkpointing can save it.

``flash_attn.cute`` (FA4, CuTe DSL) registers no torch op: its forward is invisible to
dispatch modes, so selective activation checkpointing always recomputes it. Inside a SAC
region, the ``FlashAttnFunc`` / ``FlashAttnVarlenFunc`` that FA4's public entry points call
are swapped for ``torch.ops.axolotl.flash_attn_4_fwd`` / ``flash_attn_4_varlen_fwd``. The op
runs FA4's own autograd ``forward`` and its backward op runs FA4's own ``backward`` on the
saved ``out``/``lse``, so the kernels and numerics are FA4's. The window is a flat
``window_size_left`` int (-1 = unbounded), which SAC reads to recompute sliding-window calls.

Outside SAC regions, under ``torch.compile``, without grad, or with arguments the op does not
carry (learnable sinks, mask/score mods, paged KV), FA4 is called as before. If op
registration fails, nothing is routed.
"""

from __future__ import annotations

import inspect
import itertools
import json
from collections import OrderedDict
from typing import Any, Optional

import torch

from axolotl.kernels.op_registry import _UnregisteredOp, register_kernel_op
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

DENSE = "dense"
VARLEN = "varlen"
_CLASS_NAMES = {DENSE: "FlashAttnFunc", VARLEN: "FlashAttnVarlenFunc"}
_PUBLIC_NAMES = {DENSE: "flash_attn_func", VARLEN: "flash_attn_varlen_func"}
_TENSOR_SLOTS = {
    DENSE: ("q", "k", "v"),
    VARLEN: ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k"),
}
_PRIMITIVES = (bool, int, float, str, type(None))

# forward-call token -> how to rebuild FA4's ctx in backward; bounded because a forward
# whose graph is dropped (or a recomputed call) never reaches backward to pop its entry
_MAX_PENDING = 4096
_PENDING: OrderedDict[int, Optional[tuple]] = OrderedDict()
_TOKENS = itertools.count(1)
_ORIGINALS: dict[str, Any] = {}
_NEEDS_INPUT_GRAD: dict[str, tuple] = {}


class _RecordingCtx:
    """Stands in for the autograd ctx FA4's ``forward``/``backward`` receive."""

    def __init__(self, needs_input_grad: tuple = ()):
        object.__setattr__(self, "_saved", ())
        object.__setattr__(self, "_dirty", False)
        object.__setattr__(self, "_attrs", {})
        object.__setattr__(self, "needs_input_grad", needs_input_grad)

    def __setattr__(self, name, value):
        self._attrs[name] = value

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name) from None

    def save_for_backward(self, *tensors):
        object.__setattr__(self, "_saved", tensors)

    @property
    def saved_tensors(self):
        return self._saved

    def mark_non_differentiable(self, *args):
        return None

    def set_materialize_grads(self, value):
        return None

    def mark_dirty(self, *args):
        object.__setattr__(self, "_dirty", True)


def _forward_params(func: Any) -> list[inspect.Parameter] | None:
    try:
        params = list(inspect.signature(func.forward).parameters.values())[1:]
    except (TypeError, ValueError):
        return None
    if any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    ):
        return None
    return params


def _decode(static: str) -> dict[str, Any]:
    payload = json.loads(static)
    params = payload["params"]
    for name in payload["tuples"]:
        params[name] = tuple(params[name])
    return params


def _run_forward(kind: str, tensors: dict[str, Any], static: str):
    params = _decode(static)
    params.update({k: v for k, v in tensors.items() if v is not None})
    ctx = _RecordingCtx(_NEEDS_INPUT_GRAD[kind])
    result = _ORIGINALS[kind].forward(ctx, **params)
    if not (isinstance(result, (tuple, list)) and len(result) == 2):
        raise RuntimeError(
            f"flash_attn.cute {_CLASS_NAMES[kind]}.forward returned "
            f"{type(result).__name__}, expected (out, lse); this FA4 build cannot run "
            "through axolotl's FA4 op"
        )
    out, lse = result
    return ctx, out, lse


def _layout(ctx: _RecordingCtx, tensors: dict[str, Any], out, lse) -> tuple | None:
    """How to rebuild ``ctx.saved_tensors`` from the backward op's inputs, or None."""
    if ctx._dirty or any(torch.is_tensor(v) for v in ctx._attrs.values()):
        return None
    layout = []
    for saved in ctx.saved_tensors:
        if saved is None:
            layout.append(None)
        elif saved is out:
            layout.append("out")
        elif saved is lse:
            layout.append("lse")
        else:
            name = next((n for n, t in tensors.items() if t is saved), None)
            if name is None:
                return None
            layout.append(name)
    return tuple(layout), dict(ctx._attrs)


def _forward_impl(kind: str, tensors: dict[str, Any], static: str):
    ctx, out, lse = _run_forward(kind, tensors, static)
    entry = _layout(ctx, tensors, out, lse) if lse is not None else None
    if lse is None:
        lse = out.new_empty(0, dtype=torch.float32)
    token = next(_TOKENS)
    _PENDING[token] = entry
    while len(_PENDING) > _MAX_PENDING:
        _PENDING.popitem(last=False)
    return out, lse, torch.tensor(token, dtype=torch.int64)


def _backward_impl(kind, dout, dlse, tensors, out, lse, token, static):
    entry = _PENDING.pop(int(token.item()), None)
    func = _ORIGINALS[kind]
    if entry is None:
        # the forward's ctx cannot be rebuilt: re-run FA4's forward for one
        ctx, out, lse = _run_forward(kind, tensors, static)
    else:
        layout, attrs = entry
        available = {**tensors, "out": out, "lse": lse}
        ctx = _RecordingCtx(_NEEDS_INPUT_GRAD[kind])
        ctx._attrs.update(attrs)
        ctx.save_for_backward(*(None if n is None else available[n] for n in layout))
    if dlse is None:
        dlse = torch.zeros_like(lse)
    grads = func.backward(ctx, dout, dlse)
    dq, dk, dv = grads[0], grads[1], grads[2]
    q, k, v = tensors["q"], tensors["k"], tensors["v"]
    return (
        torch.zeros_like(q) if dq is None else dq,
        torch.zeros_like(k) if dk is None else dk,
        torch.zeros_like(v) if dv is None else dv,
    )


def _fake_forward(q, v, varlen: bool):
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
    if varlen:
        lse = q.new_empty((q.shape[-2], q.shape[0]), dtype=torch.float32)
    else:
        lse = q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)
    return out, lse, torch.empty((), dtype=torch.int64, device="cpu")


@register_kernel_op("flash_attn_4_fwd")
def flash_attn_4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size_left: int,
    window_size_right: int,
    static: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _forward_impl(DENSE, {"q": q, "k": k, "v": v}, static)


@flash_attn_4_fwd.register_fake
def _(q, k, v, window_size_left, window_size_right, static):
    return _fake_forward(q, v, varlen=False)


@register_kernel_op("flash_attn_4_varlen_fwd")
def flash_attn_4_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    window_size_left: int,
    window_size_right: int,
    static: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
    }
    return _forward_impl(VARLEN, tensors, static)


@flash_attn_4_varlen_fwd.register_fake
def _(q, k, v, cu_q, cu_k, used_q, used_k, window_size_left, window_size_right, static):
    return _fake_forward(q, v, varlen=True)


@register_kernel_op("flash_attn_4_bwd")
def flash_attn_4_bwd(
    dout: torch.Tensor,
    dlse: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    token: torch.Tensor,
    static: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = {"q": q, "k": k, "v": v}
    return _backward_impl(DENSE, dout, dlse, tensors, out, lse, token, static)


@flash_attn_4_bwd.register_fake
def _(dout, dlse, q, k, v, out, lse, token, static):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


@register_kernel_op("flash_attn_4_varlen_bwd")
def flash_attn_4_varlen_bwd(
    dout: torch.Tensor,
    dlse: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    out: torch.Tensor,
    lse: torch.Tensor,
    token: torch.Tensor,
    static: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
    }
    return _backward_impl(VARLEN, dout, dlse, tensors, out, lse, token, static)


@flash_attn_4_varlen_bwd.register_fake
def _(dout, dlse, q, k, v, cu_q, cu_k, used_q, used_k, out, lse, token, static):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_dense(ctx, inputs, output):
    q, k, v, _left, _right, static = inputs
    out, lse, token = output
    ctx.save_for_backward(q, k, v, out, lse, token)
    ctx.static = static


def _backward_dense(ctx, dout, dlse, _dtoken):
    q, k, v, out, lse, token = ctx.saved_tensors
    dq, dk, dv = flash_attn_4_bwd(dout, dlse, q, k, v, out, lse, token, ctx.static)
    return dq, dk, dv, None, None, None


def _setup_varlen(ctx, inputs, output):
    *tensors, _left, _right, static = inputs
    out, lse, token = output
    ctx.save_for_backward(*tensors, out, lse, token)
    ctx.static = static


def _backward_varlen(ctx, dout, dlse, _dtoken):
    *tensors, out, lse, token = ctx.saved_tensors
    dq, dk, dv = flash_attn_4_varlen_bwd(
        dout, dlse, *tensors, out, lse, token, ctx.static
    )
    return (dq, dk, dv) + (None,) * 7


_OPS = {DENSE: flash_attn_4_fwd, VARLEN: flash_attn_4_varlen_fwd}
_REGISTERED = not any(
    isinstance(op, _UnregisteredOp)
    for op in (*_OPS.values(), flash_attn_4_bwd, flash_attn_4_varlen_bwd)
)
if _REGISTERED:
    try:
        flash_attn_4_fwd.register_autograd(_backward_dense, setup_context=_setup_dense)
        flash_attn_4_varlen_fwd.register_autograd(
            _backward_varlen, setup_context=_setup_varlen
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOG.warning(
            "FA4 op autograd registration failed (%s: %s); FA4 stays invisible to "
            "selective checkpointing",
            type(exc).__name__,
            exc,
        )
        _REGISTERED = False


def _window(params: dict[str, Any]) -> tuple[int, int] | None:
    if "window_size" in params:
        window = params["window_size"]
        if window is None:
            left = right = None
        elif isinstance(window, (tuple, list)) and len(window) == 2:
            left, right = window
        else:
            return None
    else:
        left = params.get("window_size_left")
        right = params.get("window_size_right")
    flat = []
    for side in (left, right):
        if side is None:
            flat.append(-1)
        elif isinstance(side, int) and not isinstance(side, bool):
            flat.append(side)
        else:
            return None
    return flat[0], flat[1]


class _Route:
    """Replaces an FA4 autograd Function where FA4's entry points look it up."""

    def __init__(self, kind: str, func: Any, params: list[inspect.Parameter]):
        self.kind = kind
        self.func = func
        self.signature = inspect.Signature(params)
        self._axolotl_fa4_route = True

    def __getattr__(self, name):
        return getattr(self.func, name)

    def _prepare(self, args: tuple, kwargs: dict) -> tuple | None:
        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError:
            return None
        bound.apply_defaults()
        slots = _TENSOR_SLOTS[self.kind]
        tensors: dict[str, Any] = dict.fromkeys(slots)
        statics: dict[str, Any] = {}
        tuples = []
        for name, value in bound.arguments.items():
            if torch.is_tensor(value):
                if name not in slots:
                    return None
                tensors[name] = value
            elif isinstance(value, _PRIMITIVES):
                statics[name] = value
            elif isinstance(value, (tuple, list)) and all(
                isinstance(item, _PRIMITIVES) for item in value
            ):
                statics[name] = list(value)
                if isinstance(value, tuple):
                    tuples.append(name)
            else:
                return None
        if any(tensors[name] is None for name in ("q", "k", "v")):
            return None
        if not any(tensors[name].requires_grad for name in ("q", "k", "v")):
            return None
        window = _window(statics)
        if window is None:
            return None
        static = json.dumps({"params": statics, "tuples": tuples}, sort_keys=True)
        return (*(tensors[name] for name in slots), *window, static)

    def apply(self, *args, **kwargs):
        if (
            torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and _in_sac_region()
        ):
            call = self._prepare(args, kwargs)
            if call is not None:
                out, lse, _token = _OPS[self.kind](*call)
                return out, lse
        return self.func.apply(*args, **kwargs)


def _in_sac_region() -> bool:
    from axolotl.monkeypatch.selective_checkpointing import _in_sac_region as active

    return active()


def install_fa4_ops(module: Any = None) -> bool:
    """Route FA4's autograd Functions through the axolotl ops inside SAC regions.

    ``module`` is ``flash_attn.cute`` (imported when None). The Functions are swapped in the
    namespace FA4's public ``flash_attn_func``/``flash_attn_varlen_func`` resolve them from.
    Idempotent; returns whether any Function is routed.
    """
    if not _REGISTERED:
        return False
    if module is None:
        try:
            import flash_attn.cute as module  # type: ignore[no-redef]
        except ImportError:
            return False
    routed = False
    for kind, class_name in _CLASS_NAMES.items():
        public = getattr(module, _PUBLIC_NAMES[kind], None)
        namespace = getattr(inspect.unwrap(public), "__globals__", None)
        if public is None or namespace is None:
            continue
        current = namespace.get(class_name)
        if current is None:
            continue
        if getattr(current, "_axolotl_fa4_route", False):
            routed = True
            continue
        params = _forward_params(current)
        names = [p.name for p in params or ()]
        if (
            params is None
            or names[:3] != ["q", "k", "v"]
            or not callable(getattr(current, "backward", None))
            or ("window_size" not in names and "window_size_left" not in names)
        ):
            LOG.info(
                "flash_attn.cute %s has an unexpected signature; FA4 stays invisible to "
                "selective checkpointing",
                class_name,
            )
            continue
        _ORIGINALS[kind] = current
        _NEEDS_INPUT_GRAD[kind] = tuple(n in ("q", "k", "v") for n in names)
        namespace[class_name] = _Route(kind, current, params)
        routed = True
    if routed:
        LOG.debug(
            "FA4 attention routed through torch.ops.axolotl.flash_attn_4_* inside "
            "selective checkpointing regions"
        )
    return routed
